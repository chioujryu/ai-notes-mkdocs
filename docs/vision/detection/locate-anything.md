# LocateAnything

LocateAnything（Wang et al., 2026）是一個專門做**視覺定位（Visual Grounding）**的視覺語言模型（Vision-Language Model, VLM）。你給它一張影像和一句話，例如 `"the red button"`，它會用座標指出紅色按鈕在哪裡。它也能處理一般物件偵測、密集物件偵測、GUI 元件定位、[光學字元辨識（Optical Character Recognition, OCR）](../tasks/ocr-docai.md)文字定位、文件版面定位與 point-based localization。

它最重要的設計叫做 **Parallel Box Decoding (PBD)**：不再把一個框的四個座標拆開、逐 token 慢慢生成，而是把完整 bounding box 視為一個不可拆的 **atomic unit**，在同一個 decoding step 內平行預測。

![LocateAnything：影像與文字經 Moon-ViT、MLP projector、Qwen2.5 decoder 後，以 Parallel Box Decoding 產生完整 box block；Hybrid Mode 在不可靠時切換到 NTP](../assets/locate-anything/locate-anything-arch.svg)

> 先記住一句話：**LocateAnything 不是讓所有 bounding boxes 一次全部出現，而是讓「同一個框裡的 6 個 token」在一個 step 內一起預測；不同 block 之間仍維持 causal 順序。**

---

## 故事背景：GUI Agent 卡在「一個座標一個座標念」

想像一個 GUI agent 正在操作購物網站。使用者說：

> Click the red button.

模型先看懂畫面，也知道紅色按鈕在哪裡，但它仍要把位置轉成文字序列。如果按鈕的框是 \((120,200,420,650)\)，傳統 generative VLM 常用 [**Next-Token Prediction (NTP)**](../../llm/core/decoding.md)，也就是每次只接著猜一個 token，依序輸出：

```text
<box> → <120> → <200> → <420> → <650> → </box>
```

這就像導航員明明已經知道完整地址，卻每次只能說一個字，還要等上一個字說完才能說下一個字。它帶來三個問題：

1. **慢**：一個框需要多個 sequential decoding steps；畫面裡有幾百個物件時，延遲會一直累積。
2. **幾何關係被拆散**：\(x_1,y_1,x_2,y_2\) 共同描述同一個矩形，但 NTP 把它們當成先後出現的獨立 token。
3. **錯誤會往後傳**：前面座標選錯，後面只能在錯誤 prefix 上繼續生成。

一般的 **Multi-Token Prediction (MTP)** 雖然能一次猜多個 token，卻常按照固定長度任意切塊，可能把上一個框的結尾、下一個類別名稱和另一個框的座標塞進同一塊。LocateAnything 的回答是：**既然四個座標天生屬於同一個框，就讓 block 邊界和 box 邊界對齊。**

---

## 四種座標 Decoding 方法有什麼差別？

| 方法 | 如何輸出 `(120, 200, 420, 650)` | 主要問題或優點 |
| --- | --- | --- |
| **Textual Digit Decoding** | 逐字生成 `1`、`2`、`0`、`,`…… | token 最多，速度最慢，還可能出現格式錯誤。 |
| **Quantized Coordinate Decoding + NTP** | 依序生成 `<120>`、`<200>`、`<420>`、`<650>` | 比逐 digit 短，但四個 coordinate tokens 仍是 sequential。 |
| 一般 **MTP** | 固定每次猜 \(k\) 個 token | 加速但 block 可能跨越 box 或 category 邊界，結構沒有對齊。 |
| **Parallel Box Decoding (PBD)** | 一個長度固定的 box block 同步預測完整座標 | block 就是一個幾何單位，兼顧 parallelism 與 geometric coherence。 |

PBD 的「平行」發生在**同一個 block 內**。第 \(i\) 個 block 仍會依賴先前已確認的 blocks：

$$
P(B\mid Z,E)=\prod_{i=1}^{N}P(b_i\mid b_{<i},Z,E)
$$

- \(Z\)：影像經 vision encoder 得到的 visual tokens。
- \(E\)：文字 query。
- \(b_i\)：第 \(i\) 個 box-aligned block。
- \(b_{<i}\)：已經確認並寫進 [**Key-Value Cache（KV Cache）**](../../llm/core/kv-cache.md) 的先前 blocks；它像是把已讀過的上下文重點先記住，下一步不必全部重算。

---

## 推論架構：從影像與文字到 Pixel Box

LocateAnything-3B 建立在 native-resolution VLM 上；它會保留影像的原生長寬比例與較細的空間線索。推論的六個階段與上圖完全對齊：

| 編號 | 階段 | 輸入 → 輸出 |
| --- | --- | --- |
| 1 | **影像與文字 Query** | GUI screenshot 與 `"the red button"` → 影像張量與文字 tokens。 |
| 2 | [**Moon-ViT Visual Tokens**](../backbones/moonvit.md) | 影像張量 → 保留細小文字、GUI icon 與密集物件位置線索的 visual tokens。 |
| 3 | **MLP Projector 與 Query Context** | visual tokens 與 query embeddings → Qwen2.5 能共同讀取的 context。 |
| 4 | **Qwen2.5 + Parallel Box Decoding** | context 與已確認 blocks → 一次預測長度為 6 的 Box Block。 |
| 5 | **Hybrid Validation 與局部 NTP Fallback** | 檢查格式與空間信心；不可靠時只重解碼有問題的 block。 |
| 6 | **`[0,1000]` 座標還原成 Pixel Box** | 量化座標 → 原圖上的像素邊界框。 |

Qwen2.5 decoder 內部仍是 [Transformer](../../llm/core/transformer.md)：反覆經過 masked self-attention、FFN 與 LM head。[Attention](../../llm/core/attention.md) 可以把它想成「依目前問題，替不同視覺與文字線索分配注意力」。PBD 改變的重點不是 vision encoder，也不是把 Transformer 換掉，而是**輸出單位、attention mask、訓練目標與 inference loop**。

公開的 `nvidia/LocateAnything-3B` checkpoint 使用的真實主幹如下；這些是官方設定，不是後文為了手算縮小的 toy shape：

| 模組 | 官方設定 | 資料轉換重點 |
| --- | --- | --- |
| Moon-ViT | patch size 14、27 layers、hidden size 1152、16 attention heads | 原生解析度影像 → visual tokens。 |
| 2×2 spatial merger | 每四個相鄰 token 合併 | 每組特徵為 \(4\times1152=4608\) 維。 |
| [MLP Projector](../../vlm/connectors/projector-adapter.md) | [LayerNorm](../../foundations/normalization/layernorm.md) `(4608)` → Linear → [GELU](../../foundations/activation/gelu-silu.md) → Linear | LayerNorm 先把數值尺度整理穩定，GELU 再以平滑非線性挑選特徵；最後把 Moon-ViT 特徵投影到 Qwen2.5 的 hidden size。Projector 就像不同模組之間的轉接頭。 |
| Qwen2.5-3B-Instruct | causal language decoder | 讀取 visual tokens、query 與已確認 blocks，再產生 structured output。 |

訓練時則依下圖的 T1–T3，把同一份答案同時教成逐 token 與逐 block 兩種讀法：

![LocateAnything 訓練流程：T1 建立固定長度且對齊邊界框的 blocks，T2 套用 NTP 與 MTP 聯合 attention mask，T3 以雙交叉熵損失共同訓練](../assets/locate-anything/locate-anything-training.svg)

---

## T1 — Fixed-length Box-Aligned Blocks

LocateAnything 先把連續座標正規化到 \([0,1000]\)，再量化為 coordinate tokens。所有 block 都固定為 6 個位置；用不到的位置填 `<null>`，以維持一致的 tensor shape。

以 query `"the red button"` 為例：

| Block type | 長度為 6 的示意內容 | 用途 |
| --- | --- | --- |
| **Semantic Block** | `<ref>`, `red`, `button`, `</ref>`, `<null>`, `<null>` | 說明接下來定位的是哪個語意實體；太長的名稱可拆成多個連續 blocks。 |
| **Box Block** | `<box>`, `<120>`, `<200>`, `<420>`, `<650>`, `</box>` | 四個座標與兩個 structural tokens 恰好組成一個 atomic box。 |
| **Negative Block** | `<box>`, `none`, `</box>`, `<null>`, `<null>`, `<null>` | 明確表示 query 對應的物件不存在。 |
| **End Block** | `<eos>`, `<null>`, `<null>`, `<null>`, `<null>`, `<null>` | 宣告整段生成結束。 |

`Semantic Block → Box Block → End Block` 串起來後，對外可還原成：

```text
<ref>red button</ref><box><120><200><420><650></box>
```

`<null>` 只用來補齊 block，不會保留在最後答案。

---

## T2 — Joint NTP/MTP Attention Mask

只訓練平行輸出，可能破壞原本 language decoder 擅長的 causal reasoning；只訓練 NTP，又學不會一次完成整個 box。LocateAnything 因此把同一份 ground truth 做成兩種表示：

$$
x_{\text{all}}=x_{\text{vis}}\oplus x_q\oplus x_{\text{ntp}}\oplus x_{\text{blk}}
$$

- \(x_{\text{vis}}\)：影像 visual tokens。
- \(x_q\)：query tokens。
- \(x_{\text{ntp}}\)：標準 token-by-token 序列。
- \(x_{\text{blk}}\)：依 Semantic / Box / Negative / End 規則切好的 block-wise MTP 序列。
- \(\oplus\)：sequence concatenation，不是矩陣加法。

建立 \(x_{\text{blk}}\) 時，每個 block 保留第一個 token 當 prediction context，其餘位置換成 `[mask]`，讓模型一起還原該 block 的內容。三個 target blocks 可寫成：

```text
Semantic target: [<ref>, red, button, </ref>, <null>, <null>]
Box target:      [<box>, <120>, <200>, <420>, <650>, </box>]
End target:      [<eos>, <null>, <null>, <null>, <null>, <null>]
```

### T2.1 三種 Attention 可見範圍

以下矩陣都用 `1` 表示「可以 attend」，`0` 表示「不可 attend」。

**NTP token-level causal visibility \(M_{\text{ntp}}\)，shape \(6\times6\)**：每個位置只能看自己與前面的位置。

$$
M_{\text{ntp}}=
\begin{bmatrix}
1&0&0&0&0&0\\
1&1&0&0&0&0\\
1&1&1&0&0&0\\
1&1&1&1&0&0\\
1&1&1&1&1&0\\
1&1&1&1&1&1
\end{bmatrix}
\quad\text{shape}=(6,6)
$$

**Segment visibility \(M_{\text{segment}}\)，shape \(4\times4\)**：欄與列依序為 shared context \(C\)、NTP stream \(N\)、第一個 MTP block \(B_1\)、第二個 MTP block \(B_2\)。NTP 與 MTP streams 彼此隔離，兩者都能看 shared context；後面的 block 可以看前面的 committed block。

$$
M_{\text{segment}}=
\begin{array}{c|cccc}
 & C&N&B_1&B_2\\\hline
C   &1&0&0&0\\
N   &1&1&0&0\\
B_1 &1&0&1&0\\
B_2 &1&0&1&1
\end{array}
\quad\text{shape}=(4,4)
$$

**Block 內雙向可見矩陣 \(M_{\text{intra}}\)，shape \(6\times6\)**：同一個 block 的六個位置彼此都能溝通，這就是 **bidirectional intra-block attention**。

$$
M_{\text{intra}}=
\begin{bmatrix}
1&1&1&1&1&1\\
1&1&1&1&1&1\\
1&1&1&1&1&1\\
1&1&1&1&1&1\\
1&1&1&1&1&1\\
1&1&1&1&1&1
\end{bmatrix}
\quad\text{shape}=(6,6)
$$

這三個規則合起來就是 **block-causal attention**：block 之間維持 causal，block 內部則可以雙向互看。

## T3 — Dual Cross-Entropy Loss

最後同時最小化兩份[交叉熵損失（Cross-Entropy Loss）](../../foundations/losses/cross-entropy.md)：它會懲罰模型分給正確 token 的機率太低，讓 NTP 與 MTP 兩條訓練路徑一起變準。

$$
\mathcal{L}=\mathcal{L}_{\text{ntp}}+\mathcal{L}_{\text{mtp}}
$$

例如 toy batch 得到 \(\mathcal{L}_{\text{ntp}}=0.42\)、\(\mathcal{L}_{\text{mtp}}=0.31\)，總 loss 就是 \(0.42+0.31=0.73\)。


---

## Step 1 — 影像與文字 Query

我們用一張寬 \(W_{\text{img}}=1000\)、高 \(H_{\text{img}}=600\) 的 GUI screenshot，query 是 `"the red button"`。為了能手算，以下把真實模型縮成 4 個 visual tokens、hidden dimension 2 的 toy model。

> **重要：**以下所有 feature、embedding 與 probability 都是教學用 toy values，只為追蹤資料如何流動，不是 `nvidia/LocateAnything-3B` 的真實 hidden values 或輸出機率。真實 Transformer 維度和 vocabulary 都大得多。

把 screenshot 簡化成 \(2\times2\) 四個區域，每個區域用三個可追蹤的 toy 數值表示，得到 **影像區域矩陣 \(X_{\text{img}}\)**：

$$
X_{\text{img}}=
\begin{bmatrix}
0.0&1.0&1.0\\
1.0&0.0&1.0\\
1.0&1.0&0.0\\
1.0&0.0&0.0
\end{bmatrix}
\quad\text{shape}=(4,3)
$$

四列由左上、右上、左下到右下排列；三欄只是教學用視覺量測，不把它們冒充真實模型學到的語意。文字 Query token sequence 為 \(T_q=(\texttt{the},\texttt{red},\texttt{button})\)，sequence length 為 3；它在 Step 3 才會轉成有數值的 embedding matrix。

所以本階段輸出是 \(X_{\text{img}}\)、\(T_q\)、\(W_{\text{img}}=1000\) 與 \(H_{\text{img}}=600\)，供後續視覺編碼與座標還原使用。

## Step 2 — Moon-ViT Visual Tokens

真實 Moon-ViT 會經過 patch embedding 與多層 attention。為了讓每個數值都能驗算，這裡只以 **toy Moon-ViT weight \(W_{\text{vit}}\)** 模擬一次視覺特徵轉換：

$$
W_{\text{vit}}=
\begin{bmatrix}
0.0&1.0&0.0\\
1.0&0.0&0.0\\
0.0&0.0&1.0
\end{bmatrix}
\quad\text{shape}=(3,3)
$$

矩陣關係為：

$$
\underbrace{X_{\text{img}}}_{(4,3)}\cdot
\underbrace{W_{\text{vit}}}_{(3,3)}=
\underbrace{Z}_{(4,3)}
$$

得到 **visual tokens \(Z\)**：

$$
Z=
\begin{bmatrix}
1.0&0.0&1.0\\
0.0&1.0&1.0\\
1.0&1.0&0.0\\
0.0&1.0&0.0
\end{bmatrix}
\quad\text{shape}=(4,3)
$$

每一列仍對應一個 image region，每一欄是轉換後的 toy 視覺屬性。這個 \(3\times3\) 矩陣只是把龐大的真實 encoder 壓成可手算的代理；真實 feature 不是人工指定的「紅色欄」或「按鈕欄」，而是訓練學到的 dense representation。

## Step 3 — MLP Projector 與 Query Context

為了示範，toy MLP projector 簡化成一個線性矩陣 **projector weight \(W_{\text{proj}}\)**，shape \(3\times2\)：

$$
W_{\text{proj}}=
\begin{bmatrix}
0.5&0.2\\
0.1&0.7\\
0.4&0.1
\end{bmatrix}
\quad\text{shape}=(3,2)
$$

矩陣關係只需寫成：

$$
\underbrace{Z}_{(4,3)}\cdot
\underbrace{W_{\text{proj}}}_{(3,2)}=
\underbrace{V}_{(4,2)}
$$

得到 **projected visual tokens \(V\)**：

$$
V=
\begin{bmatrix}
0.9&0.3\\
0.5&0.8\\
0.6&0.9\\
0.1&0.7
\end{bmatrix}
\quad\text{shape}=(4,2)
$$

### 3.1 加入 Query Embeddings

`the`、`red`、`button` 三個 token 的 toy **query embeddings \(Q\)** 為：

$$
Q=
\begin{bmatrix}
0.2&0.1\\
0.9&0.2\\
0.3&0.8
\end{bmatrix}
\quad\text{shape}=(3,2)
$$

把四個 projected visual tokens 與三個 query tokens 串接，得到 language decoder 的 **shared context \(C=V\oplus Q\)**：

$$
C=
\begin{bmatrix}
0.9&0.3\\
0.5&0.8\\
0.6&0.9\\
0.1&0.7\\
0.2&0.1\\
0.9&0.2\\
0.3&0.8
\end{bmatrix}
\quad\text{shape}=(7,2)
$$

前四列來自 image，後三列來自 query。Qwen2.5 decoder 會以 attention 讓文字和視覺資訊互相作用，再根據 shared context、先前 committed blocks 與目前的 masked block 產生六個位置的 token probabilities。

## Step 4 — Qwen2.5 + Parallel Box Decoding

為了完整列出所有值，toy vocabulary 只保留本例需要的 10 個 tokens，欄順序是：

```text
[<box>, <120>, <200>, <390>, <405>, <420>, <470>, <501>, <650>, </box>]
```

目前要解碼的是 `[<box>, [mask], [mask], [mask], [mask], [mask]]`。加入 token 與 position embedding 後，用 **Box Block 輸入矩陣 \(B_{\text{in}}\)** 表示：

$$
B_{\text{in}}=
\begin{bmatrix}
0.6&0.2\\
0.1&0.1\\
0.2&0.1\\
0.3&0.1\\
0.4&0.1\\
0.5&0.1
\end{bmatrix}
\quad\text{shape}=(6,2)
$$

完整 Qwen2.5 有數十層，無法用兩維 toy model重現；這裡把 attention、FFN 與 LM head 壓成同一個可追蹤函式，輸入 \(C\) 與 \(B_{\text{in}}\)，輸出 **box token logits \(L_{\text{box}}\)**。為使後續機率可精確驗算，令 \(L_{\text{box}}=\ln(P_{\text{box}})\)，顯示至小數點後八位：

$$
\operatorname{ToyQwen}\!\left(
\underbrace{C}_{(7,2)},
\underbrace{B_{\text{in}}}_{(6,2)}
\right)=
\underbrace{L_{\text{box}}}_{(6,10)}
$$

$$
L_{\text{box}}=
\begin{bmatrix}
-0.09431068&-4.60517019&-4.60517019&-4.60517019&-4.60517019&-4.60517019&-4.60517019&-4.60517019&-4.60517019&-4.60517019\\
-4.60517019&-0.10536052&-3.91202301&-4.60517019&-4.60517019&-4.60517019&-4.60517019&-4.60517019&-4.60517019&-4.60517019\\
-4.60517019&-3.91202301&-0.11653382&-4.60517019&-4.60517019&-4.60517019&-4.60517019&-4.60517019&-3.91202301&-4.60517019\\
-4.60517019&-4.60517019&-4.60517019&-2.52572864&-2.12026354&-0.49429632&-2.40794561&-2.99573227&-4.60517019&-4.60517019\\
-4.60517019&-4.60517019&-4.60517019&-4.60517019&-4.60517019&-3.91202301&-4.60517019&-4.60517019&-0.10536052&-4.60517019\\
-4.60517019&-4.60517019&-4.60517019&-4.60517019&-4.60517019&-4.60517019&-4.60517019&-4.60517019&-4.60517019&-0.09431068
\end{bmatrix}
\quad\text{shape}=(6,10)
$$

接著對每一列套用 [Softmax](../../foundations/activation/softmax.md)，得到 **box token probability matrix \(P_{\text{box}}\)**。Softmax 會把每列 logits 轉成總和為 1 的機率：

$$
\operatorname{Softmax}\!\left(
\underbrace{L_{\text{box}}}_{(6,10)}
\right)=
\underbrace{P_{\text{box}}}_{(6,10)}
$$

$$
P_{\text{box}}=
\begin{bmatrix}
0.91&0.01&0.01&0.01&0.01&0.01&0.01&0.01&0.01&0.01\\
0.01&0.90&0.02&0.01&0.01&0.01&0.01&0.01&0.01&0.01\\
0.01&0.02&0.89&0.01&0.01&0.01&0.01&0.01&0.02&0.01\\
0.01&0.01&0.01&0.08&0.12&0.61&0.09&0.05&0.01&0.01\\
0.01&0.01&0.01&0.01&0.01&0.02&0.01&0.01&0.90&0.01\\
0.01&0.01&0.01&0.01&0.01&0.01&0.01&0.01&0.01&0.91
\end{bmatrix}
\quad\text{shape}=(6,10)
$$

六列同時取 argmax，得到：

```text
[<box>, <120>, <200>, <420>, <650>, </box>]
```

這正是一個完整 box。若用 NTP，要依序走 6 個 decoding steps；PBD 在這個 box block 內只用 1 個 parallel step。

## Step 5 — Hybrid Validation 與局部 NTP Fallback

雖然第 4 列的 argmax 是 `<420>`，但它的 top-1 probability 只有：

$$
p_{\text{top-1}}=0.61<0.7
$$

前五名 coordinate candidates 依序涵蓋 `420、405、470、390、501`。它們在 \([0,1000]\) 座標空間的跨度是：

$$
\max(420,405,470,390,501)-\min(420,405,470,390,501)
=501-390=111>80
$$

兩個條件同時成立，因此觸發 **spatial ambiguity**。Hybrid Mode 會：

$$
\text{fallback}
=\left(p_{\text{top-1}}<0.7\right)
\land\left(\text{top-5 span}>80\right)
=\text{true}
$$

1. 丟棄整個未確認的 Box Block，不把它寫入 KV cache。
2. 回到上一個 verified prefix，也就是已確認的 Semantic Block。
3. 暫時切換成 **NTP**，逐 token 重解碼這一個 block。
4. 得到格式正確、信心較穩定的 `<box><120><200><420><650></box>`。
5. 把確認過的 tokens 寫入 KV cache，下一個 block 再切回 **MTP**。

在本例中，局部 NTP fallback 得到的 **已驗證 token 信心向量 \(p_{\text{verified}}\)** 為：

$$
p_{\text{verified}}=
\begin{bmatrix}
0.99&0.94&0.93&0.88&0.96&0.99
\end{bmatrix}
\quad\text{shape}=(1,6)
$$

它依序對應 `[<box>, <120>, <200>, <420>, <650>, </box>]`，因此本階段輸出已確認的四個量化座標 \((120,200,420,650)\)。這組信心仍是教學用 toy values，不是官方 checkpoint 的實測輸出。

另一種 fallback 原因叫 **format irregularity**，例如 block 內混入 `</ref>`：

```text
<box><120></ref><420><650></box>
```

這種輸出即使 coordinate confidence 很高也不能構成合法 Box Block，因此同樣直接回退到 NTP。

## Step 6 — `[0,1000]` 座標還原成 Pixel Box

確認後的 **normalized box \(B_{\text{norm}}\)** 為：

$$
B_{\text{norm}}=
\begin{bmatrix}
120&200&420&650
\end{bmatrix}
\quad\text{shape}=(1,4)
$$

轉換公式是：

$$
x_{\text{pixel}}=\frac{x}{1000}W_{\text{img}},\qquad
y_{\text{pixel}}=\frac{y}{1000}H_{\text{img}}
$$

也可把四個縮放係數排成 **pixel scaling matrix \(S_{\text{pixel}}\)**：

$$
S_{\text{pixel}}=
\begin{bmatrix}
1.0&0.0&0.0&0.0\\
0.0&0.6&0.0&0.0\\
0.0&0.0&1.0&0.0\\
0.0&0.0&0.0&0.6
\end{bmatrix}
\quad\text{shape}=(4,4)
$$

其中 \(1.0=W_{\text{img}}/1000\)，\(0.6=H_{\text{img}}/1000\)。矩陣關係為：

$$
\underbrace{B_{\text{norm}}}_{(1,4)}\cdot
\underbrace{S_{\text{pixel}}}_{(4,4)}=
\underbrace{B_{\text{pixel}}}_{(1,4)}
$$

所以 **pixel box \(B_{\text{pixel}}\)** 為：

$$
B_{\text{pixel}}=
\begin{bmatrix}
120&120&420&390
\end{bmatrix}
\quad\text{shape}=(1,4)
$$

也就是左上角 \((120,120)\)、右下角 \((420,390)\)。至此，資料已依序走完 screenshot → visual tokens → projector → query fusion → parallel box probabilities → Hybrid validation → pixel box。

---

## 三種 Inference Mode 怎麼選？

| Mode | Decoding 行為 | 論文報告的 throughput | 適合情境 |
| --- | --- | ---: | --- |
| **Slow Mode (NTP)** | 每個 token autoregressive decoding | 4.3 BPS | 離線高精度標註、final-pass data curation。 |
| **Fast Mode (MTP)** | 每個 box-aligned block 內平行預測 | 15.3 BPS | 即時性優先、場景相對單純。 |
| **Hybrid Mode** | 預設 MTP，遇到 format / spatial ambiguity 才局部回退 NTP | 12.7 BPS | production pipeline 的預設折衷。 |

這些 BPS（Boxes Per Second）數字不是跨硬體都固定的速度。論文的 throughput benchmark 使用單張 NVIDIA H100、COCO、BF16、batch size 1；因此比較其他部署時，必須同時看 GPU、輸入解析度、batch size、attention backend 和輸出 box 數量。

論文也說明，每次 MTP step 後只保留真正 committed tokens 的 KV cache；`[mask]` 與重複的 anchor token 會被移除，確保下一步看到的 prefix 和 causal training history 一致。

---

## LocateAnything-Data：不只看一般照片

PBD 解決「怎麼輸出」，大量且多樣的資料則解決「模型看過哪些定位問題」。作者整理的 LocateAnything-Data 包含約：

- **12M unique images**
- **138M language queries**
- **785M bounding boxes**

| 任務 | Query 比例 | 學到的能力 |
| --- | ---: | --- |
| General Object Detection | 66.9% | 一般與密集物件的 box localization。 |
| GUI Element Grounding | 16.5% | 找按鈕、icon、輸入框，支援 embodied / GUI agents。 |
| Referring Comprehension | 7.3% | 理解「穿紅衣服的人」這類自然語言描述。 |
| Text Localization (OCR) | 3.6% | 找到影像中文字的位置。 |
| Layout Grounding | 3.5% | 文件、表格與版面區塊定位。 |
| Point-Based Localization | 2.2% | 只需回傳一個 point 的細粒度定位。 |

在作者報告的 **Hybrid Mode** 結果中，LocateAnything-3B 達到 12.7 BPS；同一份結果摘要也列出 LVIS mean F1 50.7、COCO mean F1 54.7、M6Doc mean F1 70.1，以及 ScreenSpot-Pro 平均 60.3。定位常先以 [Intersection over Union（IoU）](../../foundations/losses/dice-iou-loss.md) 衡量預測框與真實框的重疊比例，再依 benchmark 規則彙整成 F1 等指標。這些數字應解讀為特定 benchmark 與 evaluation protocol 下的論文結果，不等於每個自有資料集都會得到相同表現。


---

## 官方 `LocateAnythingWorker` Quick Start

官方程式碼位於 Eagle repository 的 `Embodied` 目錄：

```bash
git clone https://github.com/NVlabs/Eagle.git eagle
cd eagle/Embodied
pip install -e .
```

最小 Python 範例：

```python
from PIL import Image
from locateanything_worker import LocateAnythingWorker

worker = LocateAnythingWorker("nvidia/LocateAnything-3B")
image = Image.open("example.jpg").convert("RGB")

# Object Detection
print(worker.detect(image, ["person", "car", "bicycle"])["answer"])

# Phrase Grounding
print(worker.ground_multi(image, "people wearing red shirts")["answer"])

# Scene Text Detection
print(worker.detect_text(image)["answer"])

# GUI Grounding：輸出 point
print(worker.ground_gui(image, "the search button", output_type="point")["answer"])

# Pointing
print(worker.point(image, "the traffic light")["answer"])
```

### 官方輸出格式

| 類型 | 格式 |
| --- | --- |
| Bounding box | `<ref>label</ref><box><x1><y1><x2><y2></box>` |
| Point | `<box><x><y></box>` |
| No object | `<box>none</box>` |

座標是 \([0,1000]\) 的整數。Bounding box 轉回原圖時，\(x\) 乘上 `image_width / 1000`，\(y\) 乘上 `image_height / 1000`；point 也使用相同規則。

### 發布狀態（截至 2026-07-22）

- 論文已獲 **ECCV 2026** 接收。
- 官方程式庫已支援 **batch inference**；在 NVIDIA A100、RTX 4090 等非 Hopper／Blackwell GPU 上，可選用 `la_flash` attention backend。
- 公開的 `nvidia/LocateAnything-3B` 權重尚未直接支援 **visual prompt inference**。程式庫已釋出 visual prompt 與 LoRA fine-tuning 程式，但可直接做 visual prompt inference 的官方權重仍待後續發布。

---

## 和 Grounding DINO、一般 Generative VLM 有什麼不同？

| 面向 | LocateAnything | [Grounding DINO](grounding-dino.md) | 一般 generative VLM grounding |
| --- | --- | --- | --- |
| 核心骨架 | Moon-ViT + Qwen2.5 generative VLM | Swin/BERT + DETR/DINO-style detector | Vision encoder + language decoder |
| Box 產生方式 | `PBD` 生成 structured coordinate blocks | Detection queries 經 Box Head 直接回歸 boxes | 常用 NTP 逐 coordinate token 生成 |
| 語言能力 | 保留 instruction-following 與多任務介面 | 強項是 open-vocabulary detection / grounding | 通用問答強，但定位速度與幾何一致性不一定最佳 |
| 速度策略 | Fast / Slow / Hybrid on-demand decoding | detector forward pass | 通常只能調 generation 參數或換 runtime |
| 典型優勢 | GUI、OCR、layout、dense detection、pointing 統一在一個模型 | 純偵測 pipeline 成熟，能接 [SAM](../segmentation/sam.md) 等模組 | 任務彈性高、自然語言輸出方便 |

選型直覺：

- 需要統一處理 GUI、文件、OCR、自然語言 referring 與 dense detection，而且希望 generative VLM 更快輸出座標：考慮 **LocateAnything**。
- 主要需求是 open-vocabulary object detection，或要接既有 DETR / SAM pipeline：比較 **Grounding DINO**。
- 只要固定類別、極低 latency 的 production detector：也應比較 [RF-DETR](rf-detr.md)、YOLO 等 specialist。

---

## 限制與容易誤解的地方

1. **PBD 不是所有 boxes 一次平行完成。**它平行的是一個 atomic block 內的 tokens；blocks 之間仍遵守 causal order。
2. **Fast Mode 不是永遠和 Slow Mode 一樣準。**密集排列或 category transition 複雜時仍可能出現 spatial ambiguity / format irregularity。
3. **Hybrid Mode 的速度依 fallback 比例而變。**難例越多，越常切回 NTP，實際 throughput 就越接近 Slow Mode。
4. **座標是量化值。**從 \([0,1000]\) 映射回高解析度影像時仍有 quantization；實際邊界品質也受 vision encoder 與訓練資料影響。
5. **目前主要依賴 Supervised Fine-Tuning (SFT)。**論文把用 Reinforcement Learning 改善 block policy、降低 fallback frequency 列為後續方向。
6. **部署成本不只看參數量。**原生解析度、長 context、KV cache、attention backend 與密集輸出都會影響 VRAM 和 latency。

---

## 參考資料

- NVIDIA 專案頁：[LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding](https://research.nvidia.com/labs/lpr/locate-anything/)
- 原始論文：[LocateAnything.pdf](https://research.nvidia.com/labs/lpr/locate-anything/LocateAnything.pdf)
- 官方程式碼與使用說明：[NVlabs/Eagle — Embodied](https://github.com/NVlabs/Eagle/tree/main/Embodied)
- 模型：[nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)
- 論文索引：[arXiv:2605.27365](https://arxiv.org/abs/2605.27365)

