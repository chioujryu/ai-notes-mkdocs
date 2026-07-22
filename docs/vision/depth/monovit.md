# MonoViT：讓單眼相機從影片自己學會看深度

MonoViT（**Monocular Vision Transformer**）是 Zhao 等人在 3DV 2022 提出的**自監督單目深度估計（self-supervised monocular depth estimation）**模型。它在推論時只看一張 RGB 影像，就能為每個像素預測深度；訓練時則不需要昂貴的逐像素深度標註，而是利用相鄰影片影格能否互相重建，自己產生學習訊號。

它最關鍵的設計不是「把 CNN 全部換成 Transformer」，而是讓兩者分工：**卷積神經網路（Convolutional Neural Network, CNN）**像拿放大鏡看車輪、邊緣與紋理；[**Vision Transformer（ViT）**](../backbones/vit.md)則像退後幾步看整條道路，透過 [Transformer](../../llm/core/transformer.md) 建立遠距位置的關係，理解前景小車與遠方路面即使顏色相近，深度仍不相同。

> 本文先說明為什麼單眼深度需要自監督與全域視野，再用一個完整的 2×2 數值範例，依序走過 DepthNet、PoseNet、幾何投影、影像重建與訓練損失。真實 MonoViT 的尺寸放在對照表中；手算部分使用較小維度，但保留相同資料流。

![MonoViT 完整資料流：Step 1–4 從輸入影格經 Conv-stem、MPViT encoder 與 attention decoder 產生深度；Step 5–8 以 PoseNet、幾何投影、影像重建和自監督損失完成訓練](../assets/monovit/monovit-arch.svg)

> 圖與正文使用完全相同的 Step 1–8。部署推論只走 **Step 1–4**；訓練才會繼續走 **Step 5–8**。

---

## 1. 故事背景：只有一台相機，怎麼知道前方有多遠？

想像一台只有普通行車記錄器的機器人。照片中的小車究竟是真的很小，還是只是離相機很遠？單張影像把 3D 世界壓成 2D，同一張圖可能對應很多種深度，因此**單目深度估計（monocular depth estimation）**本來就是一個沒有唯一解的問題。

最直接的方法是拿 LiDAR 或雙目相機量出真實深度，再逐像素監督模型。然而，這會遇到三個實際痛點：

1. **深度標註昂貴。** LiDAR、校正與資料同步都有成本，而且稀疏雷射點不等於完整深度圖。
2. **只靠 CNN 容易只看局部。** 淺層卷積擅長邊緣與紋理，卻不容易立刻比較畫面兩端；遠方小車若和路面顏色相近，可能被當成路面的一部分。
3. **重建式自監督本身有雜訊。** 遮擋、移動物體、光線改變，以及相機停止，都會破壞「同一個 3D 點在相鄰影格應該長得一樣」的假設。

MonoViT 對應的解法是：

- 用沒有深度標註的單眼影片訓練，將深度學習改寫成**視角合成（view synthesis）**問題。
- 在 DepthNet 編碼器中並行使用 CNN 與 Transformer，同時保留局部細節與全域關係。
- 用多尺度解碼器保留細小物體；再以 minimum reprojection、auto-mask 與 edge-aware smoothness 降低自監督雜訊。

---

## 2. 先分清楚：推論與訓練不是同一條管線

MonoViT 有兩個網路，但部署時只需要其中一個：

| 階段 | 輸入 | 使用的網路 | 輸出 |
| --- | --- | --- | --- |
| **推論** | 單張目標影像 \(I_t\) | **DepthNet** | 每像素 inverse depth，再轉成 depth map |
| **自監督訓練** | \(I_{t-1}, I_t, I_{t+1}\) | **DepthNet + PoseNet** | 深度、相鄰影格相對姿態、重建影像與損失 |

完整流程的編號如下，後面的數值範例會依此順序逐步計算：

| 流程 | 階段名稱 | 主要輸出 | 推論時使用？ |
| --- | --- | --- | --- |
| Step 1 | **輸入影格與相機資料** | \(I_t\)、訓練用 \(I_s\) 與 \(K\) | 是，但只需要 \(I_t\) |
| Step 2 | **Conv-stem 局部特徵** | stem tokens \(X\) | 是 |
| Step 3 | **MPViT Joint CNN & Transformer Encoder** | 五尺度 encoder features；toy 為 \(F\) | 是 |
| Step 4 | **[Attention](../../llm/core/attention.md) Decoder 與深度輸出** | inverse depth 與 \(D_t\) | 是 |
| Step 5 | **PoseNet 相對姿態** | \(T_{t\rightarrow s}\) | 否 |
| Step 6 | **3D 反投影、座標變換與再投影** | sampling grid \(G_s\) | 否 |
| Step 7 | **Bilinear Sampling 影像重建** | \(\widetilde I_t\) | 否 |
| Step 8 | **Minimum Reprojection、Auto-mask、Smoothness 與總損失** | \(\mathcal L_{total}\) | 否 |

### 2.1 DepthNet

DepthNet 是 encoder-decoder：

1. **Conv-stem**：用兩層卷積先抽取淺層局部特徵。
2. **Multi-Scale Patch Embedding**：連續的 3×3 depthwise convolution 形成不同有效感受野的 token 路徑。
3. **Joint CNN & Transformer Layer**：局部 CNN 分支與多條 Transformer 分支並行。
4. **Feature Fusion**：串接各分支，再用 1×1 卷積融合。
5. **Depth Decoder**：以跨層、跨尺度 skip connection 逐步放大解析度，並用注意力調整不同通道／尺度的重要性。
6. **Disparity Heads**：在完整、1/2、1/4、1/8 解析度預測四張 normalized disparity map。

### 2.2 PoseNet

PoseNet 使用輕量的 ResNet-18。它接收目標影格和一張相鄰影格的串接結果，輸出 6 DoF（degrees of freedom，三個平移量與三個旋轉量），再組成 4×4 相機變換矩陣。

PoseNet 只是訓練用的「攝影機移動估算員」；訓練完成後，單張影像經過 DepthNet 就能產生深度，不需要相鄰影格。

### 2.3 官方 MPViT-small 編碼器的真實 shape

官方 MonoViT 實作使用 MPViT-small 設定：Transformer 層數為 \([1,3,6,3]\)，路徑數為 \([2,3,3,3]\)，通道數為 \([64,128,216,288]\)，每層使用 8 個 attention heads。輸入 640×192 影像時，編碼器輸出如下。

| 特徵名稱 | 來源 | `shape`（省略 batch） |
| --- | --- | --- |
| \(E_0\) | Conv-stem | \(64\times96\times320\) |
| \(E_1\) | Joint stage 1 | \(128\times48\times160\) |
| \(E_2\) | Joint stage 2 | \(216\times24\times80\) |
| \(E_3\) | Joint stage 3 | \(288\times12\times40\) |
| \(E_4\) | Joint stage 4 | \(288\times6\times20\) |

這五個尺度交給 decoder，最後輸出 `shape` 分別為 \(1\times192\times640\)、\(1\times96\times320\)、\(1\times48\times160\)、\(1\times24\times80\) 的 disparity maps。

> 論文以「stage 1 的 Conv-stem，加上 stage 2–5 的 Joint layers」描述同一組五尺度特徵。官方程式中的 `mpvit_small` 具體採用上表設定。

---

## Step 1 — 輸入影格與相機資料

我們使用一張 2×2 灰階目標影像。左上與右上較暗，想成近處車身；下排較亮，想成遠方路面。像素值已正規化到 \([0,1]\)。

**目標影像矩陣 \(I_t\)**，`shape = (2, 2)`：

$$
I_t=
\begin{bmatrix}
0.22 & 0.40\\
0.78 & 0.70
\end{bmatrix}
$$

訓練時會取前、後兩張相鄰影格。**下一張來源影像矩陣 \(I_{s^+}\)** 與**上一張來源影像矩陣 \(I_{s^-}\)** 都是 `shape = (2, 2)`：

$$
I_{s^+}=
\begin{bmatrix}
0.20&0.50\\
0.90&0.70
\end{bmatrix},\qquad
I_{s^-}=
\begin{bmatrix}
0.22&0.43\\
0.81&0.70
\end{bmatrix}
$$

Toy 相機的**內參矩陣 \(K\)** 與**逆內參矩陣 \(K^{-1}\)** 都是 `shape = (3, 3)`。為了讓焦點放在資料流，本例使用單位矩陣：

$$
K=K^{-1}=
\begin{bmatrix}
1&0&0\\0&1&0\\0&0&1
\end{bmatrix}
$$

Step 1 的輸出因此是 \(I_t\)、\(I_{s^+}\)、\(I_{s^-}\) 與 \(K\)。推論只把 \(I_t\) 送進 DepthNet；兩張來源影格和 \(K\) 會在 Step 5–8 建立訓練訊號。

依照 row-major 順序攤平，得到**目標像素向量 \(\mathbf{i}_t\)**，`shape = (4, 1)`：

$$
\mathbf{i}_t=
\begin{bmatrix}
0.22\\0.40\\0.78\\0.70
\end{bmatrix}
$$

## Step 2 — Conv-stem 局部特徵

真實 Conv-stem 使用兩層 3×3 convolution、BatchNorm 與 Hardswish。為了手算，我們用一個共享線性投影代表 stem 的通道轉換，再用 Layer Normalization 把每列正規化。

**Stem 權重矩陣 \(W_{stem}\)**，`shape = (1, 2)`，以及**偏置向量 \(b_{stem}\)**，`shape = (1, 2)`：

$$
W_{stem}=\begin{bmatrix}-4 & 4\end{bmatrix},\qquad
b_{stem}=\begin{bmatrix}2 & -2\end{bmatrix}
$$

**Stem 投影矩陣 \(Z_{stem}\)**，`shape = (4, 2)`：

$$
\mathbf{i}_t\cdot W_{stem}+b_{stem}=Z_{stem}
=\begin{bmatrix}
1.12 & -1.12\\
0.40 & -0.40\\
-1.12 & 1.12\\
-0.80 & 0.80
\end{bmatrix}
$$

每列做 [**Layer Normalization（LayerNorm）**](../../foundations/normalization/layernorm.md)，也就是把同一個 token 的通道調整到可比較的尺度，得到**stem 特徵矩陣 \(X\)**，`shape = (4, 2)`：

$$
X=\operatorname{LN}(Z_{stem})=
\begin{bmatrix}
1 & -1\\
1 & -1\\
-1 & 1\\
-1 & 1
\end{bmatrix}
$$

上兩列與下兩列方向相反，表示 stem 已把較暗與較亮區域分開，但還沒有理解它們在整張圖的關係。

## Step 3 — MPViT Joint CNN & Transformer Encoder

Step 3 是 MonoViT 的核心：先補回位置資訊，再讓 Transformer 分支看全域、CNN 分支看局部，最後融合兩者。以下子步驟都位於圖中的同一個 Step 3 節點。

> 後續矩陣顯示至小數點後八位；所有下游結果都用未四捨五入的內部值計算，再於展示時取位數，避免把中途顯示值反覆四捨五入。

### 3.1 Convolutional Position Encoding：先把鄰近位置放回 token

MonoViT 採用的 MPViT block 不靠固定 position embedding，而使用 **Convolutional Position Encoding（CPE）**：把 token 還原成特徵圖，做 depthwise 3×3 convolution，再加回原特徵。

Toy 的兩個**深度卷積核 \(K_{CPE}^{(1)}\)、\(K_{CPE}^{(2)}\)**，各自 `shape = (3, 3)`：

$$
K_{CPE}^{(1)}=K_{CPE}^{(2)}=
\begin{bmatrix}
0&0&0\\
0&0.10&0\\
0&0&0
\end{bmatrix}
$$

卷積結果是 \(0.1X\)，加回 residual 後得到**位置特徵矩陣 \(X_{cpe}\)**，`shape = (4, 2)`：

$$
X_{cpe}=X+\operatorname{DWConv}(X)=
\begin{bmatrix}
1.10 & -1.10\\
1.10 & -1.10\\
-1.10 & 1.10\\
-1.10 & 1.10
\end{bmatrix}
$$

### 3.2 Factorized Attention：不用先建立 4×4 token 關係表

[**Factorized Multi-Head Self-Attention（factorized MHSA）**](../../llm/core/multi-head_attention.md) 是一種較省計算的多頭自注意力：它把一般的
\(\operatorname{softmax}(QK^T)V\) 改寫成先算 \(\operatorname{softmax}_N(K)^TV\)，再讓 \(Q\) 查詢較小的通道摘要。[Softmax](../../foundations/activation/softmax.md) 會把一組分數轉成總和為 1 的權重；下標 \(N\) 表示沿 token 維度正規化。

本例只有一個 head、每個 head 兩維。LayerNorm 後的**注意力輸入矩陣 \(X_n\)**，`shape = (4, 2)`：

$$
X_n=\operatorname{LN}(X_{cpe})=
\begin{bmatrix}
1 & -1\\
1 & -1\\
-1 & 1\\
-1 & 1
\end{bmatrix}
$$

三個**投影權重矩陣 \(W_Q,W_K,W_V\)** 均為 `shape = (2, 2)`：

$$
W_Q=W_K=W_V=
\begin{bmatrix}
1&0\\0&1
\end{bmatrix}
$$

因此 **Query、Key、Value 矩陣 \(Q,K,V\)** 都是 `shape = (4, 2)`，且數值相同：

$$
Q=X_n\cdot W_Q=K=X_n\cdot W_K=V=X_n\cdot W_V=
\begin{bmatrix}
1 & -1\\
1 & -1\\
-1 & 1\\
-1 & 1
\end{bmatrix}
$$

沿四個 token 做 Softmax，得到**正規化 Key 矩陣 \(\widetilde K\)**，`shape = (4, 2)`：

$$
\widetilde K=\operatorname{softmax}_N(K)=
\begin{bmatrix}
0.44039854&0.05960146\\
0.44039854&0.05960146\\
0.05960146&0.44039854\\
0.05960146&0.44039854
\end{bmatrix}
$$

先彙整成**通道摘要矩陣 \(M\)**，`shape = (2, 2)`：

$$
\widetilde K^T\cdot V=M=
\begin{bmatrix}
0.76159416&-0.76159416\\
-0.76159416&0.76159416
\end{bmatrix}
$$

再讓每個 query 讀取摘要，得到**factorized attention 矩陣 \(A_f\)**，`shape = (4, 2)`：

$$
\frac{Q\cdot M}{\sqrt{2}}=A_f=
\begin{bmatrix}
1.07705678&-1.07705678\\
1.07705678&-1.07705678\\
-1.07705678&1.07705678\\
-1.07705678&1.07705678
\end{bmatrix}
$$

真實 block 還加入 **Convolutional Relative Position Encoding（CRPE）**，以局部卷積補回相對位置。Toy 的**相對位置矩陣 \(C_{rel}\)**，`shape = (4, 2)`：

$$
C_{rel}=
\begin{bmatrix}
0.10&-0.05\\
0.05&-0.10\\
-0.05&0.10\\
-0.10&0.05
\end{bmatrix}
$$

**輸出投影權重矩陣 \(W_O\)**，`shape = (2, 2)`：

$$
W_O=\begin{bmatrix}1&0\\0&1\end{bmatrix}
$$

注意力與位置訊號相加、投影後得到**全域特徵矩陣 \(H\)**，`shape = (4, 2)`：

$$
(A_f+C_{rel})\cdot W_O=H=
\begin{bmatrix}
1.17705678&-1.12705678\\
1.12705678&-1.17705678\\
-1.12705678&1.17705678\\
-1.17705678&1.12705678
\end{bmatrix}
$$

加上第一條 residual，得到**注意力殘差矩陣 \(R\)**，`shape = (4, 2)`：

$$
R=X_{cpe}+H=
\begin{bmatrix}
2.27705678&-2.22705678\\
2.22705678&-2.27705678\\
-2.22705678&2.27705678\\
-2.27705678&2.22705678
\end{bmatrix}
$$

這一步對應第一個痛點：每個位置不必等待很多層局部卷積，現在就能取得由全圖四個 token 彙整出的訊息。

### 3.3 Feed-Forward Network：逐 token 整理特徵

第二次 LayerNorm 產生**FFN 輸入矩陣 \(R_n\)**，`shape = (4, 2)`：

$$
R_n=\operatorname{LN}(R)=
\begin{bmatrix}
1&-1\\1&-1\\-1&1\\-1&1
\end{bmatrix}
$$

Toy FFN 的兩個**權重矩陣 \(W_1,W_2\)** 均為 `shape = (2, 2)`：

$$
W_1=\begin{bmatrix}0.2&0\\0&0.2\end{bmatrix},\qquad
W_2=\begin{bmatrix}1&0\\0&1\end{bmatrix}
$$

第一層得到**FFN 線性矩陣 \(Z_{ffn}\)**，`shape = (4, 2)`：

$$
R_n\cdot W_1=Z_{ffn}=
\begin{bmatrix}
0.2&-0.2\\0.2&-0.2\\-0.2&0.2\\-0.2&0.2
\end{bmatrix}
$$

經 [**GELU（Gaussian Error Linear Unit）**](../../foundations/activation/gelu-silu.md)，也就是平滑地決定保留多少訊號，再乘第二個權重：

**FFN 輸出矩陣 \(G_{ffn}\)**，`shape = (4, 2)`：

$$
\operatorname{GELU}(Z_{ffn})\cdot W_2=G_{ffn}=
\begin{bmatrix}
0.11585194&-0.08414806\\
0.11585194&-0.08414806\\
-0.08414806&0.11585194\\
-0.08414806&0.11585194
\end{bmatrix}
$$

加上第二條 residual，得到**Transformer 分支矩陣 \(T\)**，`shape = (4, 2)`：

$$
T=R+G_{ffn}=
\begin{bmatrix}
2.39290873&-2.31120484\\
2.34290873&-2.36120484\\
-2.31120484&2.39290873\\
-2.36120484&2.34290873
\end{bmatrix}
$$

### 3.4 CNN 局部分支與特徵融合

真實 Joint layer 另有 1×1 → 3×3 depthwise → 1×1 的 residual convolution branch。Toy 的兩個**depthwise kernel \(K_{DW}^{(1)},K_{DW}^{(2)}\)**，各自 `shape = (3, 3)`：

$$
K_{DW}^{(1)}=K_{DW}^{(2)}=
\begin{bmatrix}
0&0&0\\0&1&0\\0&0&0
\end{bmatrix}
$$

為了把焦點放在分支融合，kernel 只保留中心；接著用**局部通道權重矩陣 \(W_{local}\)** 混合通道，`shape = (2, 2)`：

$$
W_{local}=\begin{bmatrix}0.6&0.4\\0.4&0.6\end{bmatrix}
$$

得到**CNN 局部特徵矩陣 \(L\)**，`shape = (4, 2)`：

$$
\operatorname{DWConv}(X)\cdot W_{local}=L=
\begin{bmatrix}
0.2&-0.2\\0.2&-0.2\\-0.2&0.2\\-0.2&0.2
\end{bmatrix}
$$

真實 MPViT-small 在第一個 joint stage 有兩條 Transformer 路徑，後三個 stage 各有三條；每條路徑具有不同累積感受野。Toy 只保留一條 Transformer 路徑與一條 CNN 路徑，串接成**多路徑矩陣 \(C_{path}\)**，`shape = (4, 4)`：

$$
C_{path}=\operatorname{Concat}(T,L)=
\begin{bmatrix}
2.39290873&-2.31120484&0.20000000&-0.20000000\\
2.34290873&-2.36120484&0.20000000&-0.20000000\\
-2.31120484&2.39290873&-0.20000000&0.20000000\\
-2.36120484&2.34290873&-0.20000000&0.20000000
\end{bmatrix}
$$

以 1×1 convolution 表示的**融合權重矩陣 \(W_{fuse}\)**，`shape = (4, 2)`：

$$
W_{fuse}=
\begin{bmatrix}
0.7&0\\0&0.7\\0.3&0\\0&0.3
\end{bmatrix}
$$

得到**融合特徵矩陣 \(F\)**，`shape = (4, 2)`：

$$
C_{path}\cdot W_{fuse}=F=
\begin{bmatrix}
1.73503611&-1.67784339\\
1.70003611&-1.71284339\\
-1.67784339&1.73503611\\
-1.71284339&1.70003611
\end{bmatrix}
$$

這正是 MonoViT 的核心：\(T\) 提供全域關係，\(L\) 保留局部紋理，兩者不是二選一。

## Step 4 — Attention Decoder 與深度輸出

Encoder 已經取得局部與全域資訊，Step 4 接著把五個解析度的特徵逐步放大並融合，再輸出每個像素的 inverse depth。Toy 範例保留一個高解析度尺度與一個低解析度尺度。

### 4.1 跨尺度 decoder 與 channel attention

真實 decoder 接收五個 encoder 尺度。Toy 用 \(F\) 代表高解析度特徵，並以全域平均代表最深層的低解析度特徵。

**低解析度特徵矩陣 \(G\)**，`shape = (1, 2)`：

$$
G=\operatorname{MeanTokens}(F)=\begin{bmatrix}0.01109636&0.01109636\end{bmatrix}
$$

最近鄰上採樣後得到**上採樣矩陣 \(U\)**，`shape = (4, 2)`：

$$
U=
\begin{bmatrix}
0.01109636&0.01109636\\
0.01109636&0.01109636\\
0.01109636&0.01109636\\
0.01109636&0.01109636
\end{bmatrix}
$$

與高解析度特徵串接成**decoder 輸入矩陣 \(C_{dec}\)**，`shape = (4, 4)`：

$$
C_{dec}=\operatorname{Concat}(F,U)=
\begin{bmatrix}
1.73503611&-1.67784339&0.01109636&0.01109636\\
1.70003611&-1.71284339&0.01109636&0.01109636\\
-1.67784339&1.73503611&0.01109636&0.01109636\\
-1.71284339&1.70003611&0.01109636&0.01109636
\end{bmatrix}
$$

全域平均得到**通道摘要矩陣 \(p_{ca}\)**，`shape = (1, 4)`：

$$
p_{ca}=\begin{bmatrix}0.01109636&0.01109636&0.01109636&0.01109636\end{bmatrix}
$$

Toy 將兩層 channel-attention MLP 壓成一個線性門。其**權重矩陣 \(W_{ca}\)**，`shape = (4, 4)`，與**偏置向量 \(b_{ca}\)**，`shape = (1, 4)`：

$$
W_{ca}=\begin{bmatrix}
0&0&0&0\\0&0&0&0\\0&0&0&0\\0&0&0&0
\end{bmatrix},\qquad
b_{ca}=\begin{bmatrix}
\ln 4&\ln(7/3)&\ln(3/2)&0
\end{bmatrix}
=\begin{bmatrix}
1.38629436&0.84729786&0.40546511&0
\end{bmatrix}
$$

經 [**Sigmoid**](../../foundations/activation/sigmoid-tanh.md)，也就是把任意實數壓到 0–1 當成門控強度，得到**通道門控矩陣 \(g\)**，`shape = (1, 4)`：

$$
g=\sigma(p_{ca}\cdot W_{ca}+b_{ca})=
\begin{bmatrix}0.80000000&0.70000000&0.60000000&0.50000000\end{bmatrix}
$$

逐通道相乘得到**注意力加權矩陣 \(C_g\)**，`shape = (4, 4)`：

$$
C_g=C_{dec}\odot g=
\begin{bmatrix}
1.38802889&-1.17449037&0.00665782&0.00554818\\
1.36002889&-1.19899037&0.00665782&0.00554818\\
-1.34227471&1.21452528&0.00665782&0.00554818\\
-1.37027471&1.19002528&0.00665782&0.00554818
\end{bmatrix}
$$

**Decoder 融合權重矩陣 \(W_{dec}\)**，`shape = (4, 2)`：

$$
W_{dec}=\begin{bmatrix}
1&0\\0&1\\0.2&0\\0&0.2
\end{bmatrix}
$$

**Decoder 輸出矩陣 \(D_f\)**，`shape = (4, 2)`：

$$
C_g\cdot W_{dec}=D_f=
\begin{bmatrix}
1.38936045&-1.17338074\\
1.36136045&-1.19788074\\
-1.34094315&1.21563491\\
-1.36894315&1.19113491
\end{bmatrix}
$$

> 論文把 decoder 的 Atten Block 描述為空間與通道注意力。官方釋出程式中的 `Attention_Module` 主要啟用 `ChannelAttention`，而跨尺度融合位置另使用 feature squeeze-and-excitation（fSE）門控。

### 4.2 Disparity head：從 normalized disparity 轉成深度

**Disparity head 權重矩陣 \(W_{disp}\)**，`shape = (2, 1)`，以及**偏置 \(b_{disp}\)**，`shape = (1, 1)`：

$$
W_{disp}=\begin{bmatrix}0.5\\-0.5\end{bmatrix},\qquad
b_{disp}=\begin{bmatrix}-1\end{bmatrix}
$$

得到**disparity logits 矩陣 \(z\)**，`shape = (4, 1)`：

$$
D_f\cdot W_{disp}+b_{disp}=z=
\begin{bmatrix}
0.28137059\\0.27962059\\-2.27828903\\-2.28003903
\end{bmatrix}
$$

Sigmoid 後的**normalized disparity 矩陣 \(y\)**，`shape = (4, 1)`：

$$
y=\sigma(z)=
\begin{bmatrix}
0.56988221\\0.56945320\\0.09293709\\0.09278967
\end{bmatrix}
$$

真實實作通常把深度限制在 \([0.1,100]\) 公尺。為了讓 toy 數字容易讀，我們改用 \([1,10]\) 公尺，因此
\(d_{min}=1/10=0.1\)、\(d_{max}=1/1=1\)。

縮放公式與深度公式是：

$$
d= d_{min}+(d_{max}-d_{min})y,\qquad
\operatorname{depth}=\frac{1}{d}
$$

重排成 2×2 後，得到**inverse-depth 矩陣 \(D_{inv}\)**，`shape = (2, 2)`：

$$
D_{inv}=
\begin{bmatrix}
0.61289399&0.61250788\\
0.18364338&0.18351070
\end{bmatrix}
$$

最後的**深度矩陣 \(D_t\)**，`shape = (2, 2)`，單位為公尺：

$$
D_t=\frac{1}{D_{inv}}=
\begin{bmatrix}
1.63160353&1.63263205\\
5.44533655&5.44927351
\end{bmatrix}
$$

上排約 1.63 m、下排約 5.45 m；模型已把畫面分成近處物體與遠處路面。Step 4 的輸出是 \(D_t\)，到這裡就是**推論時的完整流程**。

---

## 自監督訓練：沿用 Step 1–4 的輸出

自監督的核心問題是：「如果深度與相機移動都猜對了，能不能用相鄰影格重建目標影格？」以下沿用 Step 1 的 \(I_t\)、\(I_{s^+}\)、\(I_{s^-}\)、\(K\)，以及 Step 4 的 \(D_t\)。

## Step 5 — PoseNet 相對姿態

先處理下一張影格。逐位置串接目標與來源強度，得到**下一張影格配對矩陣 \(P_{pair}^{+}\)**，`shape = (4, 2)`：

$$
P_{pair}^{+}=
\begin{bmatrix}
0.22&0.20\\0.40&0.50\\0.78&0.90\\0.70&0.70
\end{bmatrix}
$$

Toy 用全域平均代表 ResNet-18 聚合，得到**下一張配對摘要矩陣 \(\bar P^{+}\)**，`shape = (1, 2)`：

$$
\bar P^{+}=\begin{bmatrix}0.525&0.575\end{bmatrix}
$$

為了專注幾何運算，本例令 PoseNet head 的**權重矩陣 \(W_{pose}\)** 為 `shape = (2, 6)`，並由**偏置矩陣 \(b_{pose}\)**，`shape = (1, 6)`，指定一個沿 \(z\) 軸移動 1 m 的已知姿態：

$$
W_{pose}=\begin{bmatrix}
0&0&0&0&0&0\\
0&0&0&0&0&0
\end{bmatrix},\qquad
b_{pose}=\begin{bmatrix}0&0&1&0&0&0\end{bmatrix}
$$

**6 DoF 姿態矩陣 \(\xi\)**，`shape = (1, 6)`，欄位依序是 \([t_x,t_y,t_z,r_x,r_y,r_z]\)：

$$
\bar P^{+}\cdot W_{pose}+b_{pose}=\xi^{+}=
\begin{bmatrix}0&0&1&0&0&0\end{bmatrix}
$$

上一張影格也走同一個 PoseNet。其**影格配對矩陣 \(P_{pair}^{-}\)**，`shape = (4, 2)`，以及**配對摘要矩陣 \(\bar P^{-}\)**，`shape = (1, 2)`，為：

$$
P_{pair}^{-}=
\begin{bmatrix}
0.22&0.22\\0.40&0.43\\0.78&0.81\\0.70&0.70
\end{bmatrix},\qquad
\bar P^{-}=
\begin{bmatrix}0.525&0.540\end{bmatrix}
$$

同一組 toy pose head 也得到：

$$
\bar P^{-}\cdot W_{pose}+b_{pose}=\xi^{-}=
\begin{bmatrix}0&0&1&0&0&0\end{bmatrix}
\quad\text{shape}=(1,6)
$$

轉成**齊次變換矩陣 \(T_{t\rightarrow s}\)**，`shape = (4, 4)`：

$$
T_{t\rightarrow s^+}=T_{t\rightarrow s^-}=T_{t\rightarrow s}=
\begin{bmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&1\\
0&0&0&1
\end{bmatrix}
$$

真實 PoseNet 不會使用零權重；這裡只是固定一個可追蹤姿態，讓後面的反投影、變換與取樣都能完整手算。

## Step 6 — 3D 反投影、座標變換與再投影

Step 6 先把每個 2D 像素依深度拉回 3D，再套用 Step 5 的相機姿態，最後投影到來源影格，得到可供取樣的 2D 座標。

### 6.1 從像素反投影成 3D 點

四個像素的**齊次座標矩陣 \(P_t\)**，`shape = (3, 4)`：

$$
P_t=
\begin{bmatrix}
0&1&0&1\\
0&0&1&1\\
1&1&1&1
\end{bmatrix}
$$

內參矩陣 \(K\) 與逆矩陣 \(K^{-1}\) 已在 Step 1 定義，兩者都是 `shape = (3, 3)` 的單位矩陣。

把深度攤平為**深度列矩陣 \(\mathbf d_t\)**，`shape = (1, 4)`：

$$
\mathbf d_t=\begin{bmatrix}1.63160353&1.63263205&5.44533655&5.44927351\end{bmatrix}
$$

先計算 \(K^{-1}P_t\)，再讓每一欄乘上對應深度，得到**目標相機座標矩陣 \(X_t\)**，`shape = (3, 4)`：

$$
K^{-1}\cdot P_t\odot\mathbf d_t=X_t=
\begin{bmatrix}
0&1.63263205&0&5.44927351\\
0&0&5.44533655&5.44927351\\
1.63160353&1.63263205&5.44533655&5.44927351
\end{bmatrix}
$$

補上齊次座標，得到**齊次 3D 點矩陣 \(X_t^h\)**，`shape = (4, 4)`：

$$
X_t^h=
\begin{bmatrix}
0&1.63263205&0&5.44927351\\
0&0&5.44533655&5.44927351\\
1.63160353&1.63263205&5.44533655&5.44927351\\
1&1&1&1
\end{bmatrix}
$$

### 6.2 把 3D 點移到來源相機，再投影回 2D

**來源相機座標矩陣 \(X_s^h\)**，`shape = (4, 4)`：

$$
T_{t\rightarrow s}\cdot X_t^h=X_s^h=
\begin{bmatrix}
0&1.63263205&0&5.44927351\\
0&0&5.44533655&5.44927351\\
2.63160353&2.63263205&6.44533655&6.44927351\\
1&1&1&1
\end{bmatrix}
$$

用**延伸內參矩陣 \(K_{ext}\)** 投影，`shape = (3, 4)`：

$$
K_{ext}=\begin{bmatrix}
1&0&0&0\\0&1&0&0\\0&0&1&0
\end{bmatrix}
$$

得到**未正規化投影矩陣 \(P_s'\)**，`shape = (3, 4)`：

$$
K_{ext}\cdot X_s^h=P_s'=
\begin{bmatrix}
0&1.63263205&0&5.44927351\\
0&0&5.44533655&5.44927351\\
2.63160353&2.63263205&6.44533655&6.44927351
\end{bmatrix}
$$

每欄的前兩列除以第三列，得到**來源取樣座標矩陣 \(G_s\)**，`shape = (2, 4)`：

$$
G_s=
\begin{bmatrix}
0&0.62015201&0&0.84494378\\
0&0&0.84484906&0.84494378
\end{bmatrix}
$$

近處上排點因深度較小，投影位移比例和遠處下排不同。這正是重建損失可以反過來監督深度的原因。

## Step 7 — Bilinear Sampling 影像重建

先把下一張來源影像攤平為**來源像素向量 \(\mathbf i_{s^+}\)**，`shape = (4, 1)`：

$$
\mathbf i_{s^+}=\begin{bmatrix}0.20\\0.50\\0.90\\0.70\end{bmatrix}
$$

依 \(G_s\) 計算四鄰點的 bilinear weights，得到**取樣權重矩陣 \(B\)**，`shape = (4, 4)`；欄依序代表來源的左上、右上、左下、右下像素：

$$
B=
\begin{bmatrix}
1.00000000&0&0&0\\
0.37984799&0.62015201&0&0\\
0.15515094&0&0.84484906&0\\
0.02404243&0.13101379&0.13101379&0.71392998
\end{bmatrix}
$$

重建的**目標像素向量 \(\widetilde{\mathbf i}_t\)**，`shape = (4, 1)`：

$$
B\cdot\mathbf i_{s^+}=\widetilde{\mathbf i}_t^{+}=
\begin{bmatrix}
0.20000000\\0.38604560\\0.79139434\\0.68797878
\end{bmatrix}
$$

重排後的**下一張來源重建影像矩陣 \(\widetilde I_t^{+}\)**，`shape = (2, 2)`：

$$
\widetilde I_t^{+}=
\begin{bmatrix}
0.20000000&0.38604560\\
0.79139434&0.68797878
\end{bmatrix}
$$

上一張影格的 pose 與取樣座標相同，因此沿用同一個 \(B\)。其**上一張來源像素向量 \(\mathbf i_{s^-}\)**，`shape = (4, 1)`：

$$
\mathbf i_{s^-}=
\begin{bmatrix}0.22\\0.43\\0.81\\0.70\end{bmatrix}
$$

完整矩陣乘法為：

$$
\underbrace{B}_{(4,4)}\cdot
\underbrace{\mathbf i_{s^-}}_{(4,1)}=
\underbrace{\widetilde{\mathbf i}_t^{-}}_{(4,1)}
$$

得到**上一張來源重建向量 \(\widetilde{\mathbf i}_t^{-}\)**，`shape = (4, 1)`，以及**重建影像矩陣 \(\widetilde I_t^{-}\)**，`shape = (2, 2)`：

$$
\widetilde{\mathbf i}_t^{-}=
\begin{bmatrix}
0.22000000\\0.35023192\\0.71846095\\0.66749743
\end{bmatrix},\qquad
\widetilde I_t^{-}=
\begin{bmatrix}
0.22000000&0.35023192\\
0.71846095&0.66749743
\end{bmatrix}
$$

兩個重建結果都可與原本的 \(I_t\) 比較。由於 `grid_sample` 與上述投影都是可微分操作，誤差能一路反向傳回 disparity head、decoder 與 encoder。

---

## Step 8 — Minimum Reprojection、Auto-mask、Smoothness 與總損失

Step 8 同時回答兩個問題：哪些重建像素值得相信，以及深度圖應該在哪裡平滑。最後把兩部分合成可反向傳播的訓練目標。

### 8.1 Photometric reprojection loss

先算下一張來源重建的**逐像素 L1 誤差矩陣 \(E_{L1}^{+}\)**，`shape = (2, 2)`：

$$
E_{L1}^{+}=|\widetilde I_t^{+}-I_t|=
\begin{bmatrix}
0.02000000&0.01395440\\
0.01139434&0.01202122
\end{bmatrix}
$$

MonoViT 沿用 Monodepth2 的 photometric function：

$$
\mathcal F(\widetilde I,I)
=\alpha\frac{1-\operatorname{SSIM}(\widetilde I,I)}{2}
+(1-\alpha)|\widetilde I-I|,\qquad \alpha=0.85
$$

**SSIM（Structural Similarity Index Measure）**用局部平均、變異與共變異比較結構，而不只比較單一像素亮度。為了讓 2×2 toy 可手算，本例用整張 2×2 當共同視窗；真實程式使用帶 reflection padding 的 3×3 局部視窗。

本例的統計量依序為：

$$
\mu_{\widetilde I^+}=0.51635468,\quad \mu_I=0.52500000,\quad
\sigma^2_{\widetilde I^+}=0.05554060,\quad \sigma^2_I=0.05107500,\quad
\sigma_{\widetilde I^+,I}=0.05323654
$$

取 \(C_1=0.01^2\)、\(C_2=0.03^2\)，得到
\(\operatorname{SSIM}=0.99853675\)，所以 structural loss 為 \(0.00073162\)。將它廣播到四個像素後，得到由下一張影格重建的**photometric loss 矩陣 \(F_{+1}\)**，`shape = (2, 2)`：

$$
F_{+1}=
\begin{bmatrix}
0.00362188&0.00271504\\
0.00233103&0.00242506
\end{bmatrix}
$$

### 8.2 Minimum reprojection：前後影格選比較可信的那一張

上一張來源已在 Step 5–7 完整走過相同資料流。其**逐像素 L1 誤差矩陣 \(E_{L1}^{-}\)**，`shape = (2, 2)`：

$$
E_{L1}^{-}=|\widetilde I_t^{-}-I_t|=
\begin{bmatrix}
0.00000000&0.04976808\\
0.06153905&0.03250257
\end{bmatrix}
$$

其整張 2×2 視窗統計量為：

$$
\mu_{\widetilde I^-}=0.48904757,\quad \mu_I=0.52500000,\quad
\sigma^2_{\widetilde I^-}=0.04403281,\quad \sigma^2_I=0.05107500,\quad
\sigma_{\widetilde I^-,I}=0.04728515
$$

因此 \(\operatorname{SSIM}(\widetilde I_t^{-},I_t)=0.99190510\)，structural loss 為 \(0.00404745\)。代入同一個 photometric function，得到**上一張來源 photometric loss 矩陣 \(F_{-1}\)**，`shape = (2, 2)`：

$$
F_{-1}=
\begin{bmatrix}
0.00344033&0.01090554\\
0.01267119&0.00831572
\end{bmatrix}
$$

逐像素取較小值，得到**minimum reprojection 矩陣 \(F_{min}\)**，`shape = (2, 2)`：

$$
F_{min}=\min(F_{-1},F_{+1})=
\begin{bmatrix}
0.00344033&0.00271504\\
0.00233103&0.00242506
\end{bmatrix}
$$

若某個點在一側被遮住，另一側影格仍可能看得到；取 minimum 能降低遮擋造成的錯誤監督。

### 8.3 Auto-mask：不移動也能重建的像素，不拿來教深度

如果不做任何幾何 warp，直接把兩張來源影像各自和目標影像比較，再逐像素取 minimum，就得到 identity reprojection。這次沒有未展開的矩陣：直接把 Step 1 的 \(I_{s^+}\) 與 \(I_{s^-}\) 分別代入同一個 photometric function。

對下一張來源，\(\operatorname{SSIM}(I_{s^+},I_t)=0.96487365\)，structural loss 為 \(0.01756318\)，得到**下一張 identity loss 矩陣 \(F_{id}^{+}\)**，`shape = (2, 2)`：

$$
F_{id}^{+}=
\begin{bmatrix}
0.01792870&0.02992870\\
0.03292870&0.01492870
\end{bmatrix}
$$

對上一張來源，\(\operatorname{SSIM}(I_{s^-},I_t)=0.99746597\)，structural loss 為 \(0.00126701\)，得到**上一張 identity loss 矩陣 \(F_{id}^{-}\)**，`shape = (2, 2)`：

$$
F_{id}^{-}=
\begin{bmatrix}
0.00107696&0.00557696\\
0.00557696&0.00107696
\end{bmatrix}
$$

逐像素取 minimum，得到**identity photometric loss 矩陣 \(F_{id}\)**，`shape = (2, 2)`：

$$
F_{id}=
\min(F_{id}^{+},F_{id}^{-})=
\begin{bmatrix}
0.00107696&0.00557696\\
0.00557696&0.00107696
\end{bmatrix}
$$

只有「幾何重建比原地照抄更好」的像素才保留。**Auto-mask 矩陣 \(\mu\)**，`shape = (2, 2)`：

$$
\mu=[F_{min}<F_{id}]=
\begin{bmatrix}
0&1\\1&0
\end{bmatrix}
$$

左上與右下都是原地照抄的 loss 更小，因此它們不提供深度梯度；中間兩個像素才由幾何重建提供監督。平均遮罩後得到：

$$
\mathcal L_{ss}=\operatorname{Mean}(\mu\odot F_{min})
=\frac{0+0.0027150409+0.0023310329+0}{4}=0.0012615185
$$

### 8.4 Edge-aware smoothness：平坦區要平順，影像邊緣可以跳變

先將 inverse depth 除以全圖平均 \(0.39813899\)，得到**mean-normalized inverse-depth 矩陣 \(D_{inv}^*\)**，`shape = (2, 2)`：

$$
D_{inv}^*=\frac{D_{inv}}{\overline D_{inv}}=
\begin{bmatrix}
1.53939706&1.53842729\\
0.46125445&0.46092120
\end{bmatrix}
$$

其**水平深度梯度矩陣 \(|\partial_xD_{inv}^*|\)**，`shape = (2, 1)`，與**垂直深度梯度矩陣 \(|\partial_yD_{inv}^*|\)**，`shape = (1, 2)`：

$$
|\partial_xD_{inv}^*|=
\begin{bmatrix}0.00096978\\0.00033324\end{bmatrix},\qquad
|\partial_yD_{inv}^*|=
\begin{bmatrix}1.07814262&1.07750609\end{bmatrix}
$$

目標影像的**水平梯度矩陣 \(|\partial_xI_t|\)**，`shape = (2, 1)`，與**垂直梯度矩陣 \(|\partial_yI_t|\)**，`shape = (1, 2)`：

$$
|\partial_xI_t|=
\begin{bmatrix}0.18\\0.08\end{bmatrix},\qquad
|\partial_yI_t|=
\begin{bmatrix}0.56&0.30\end{bmatrix}
$$

轉成**水平邊緣權重矩陣 \(W_x\)**，`shape = (2, 1)`，與**垂直邊緣權重矩陣 \(W_y\)**，`shape = (1, 2)`：

$$
W_x=e^{-|\partial_xI_t|}=
\begin{bmatrix}0.83527021\\0.92311635\end{bmatrix},\qquad
W_y=e^{-|\partial_yI_t|}=
\begin{bmatrix}0.57120906&0.74081822\end{bmatrix}
$$

逐元素加權後，得到**水平平滑項矩陣 \(S_x\)**，`shape = (2, 1)`，與**垂直平滑項矩陣 \(S_y\)**，`shape = (1, 2)`：

$$
S_x=|\partial_xD_{inv}^*|\odot W_x=
\begin{bmatrix}0.00081002\\0.00030762\end{bmatrix}
$$

$$
S_y=|\partial_yD_{inv}^*|\odot W_y=
\begin{bmatrix}0.61584484&0.79823614\end{bmatrix}
$$

因此：

$$
\mathcal L_{smooth}=\operatorname{Mean}(S_x)+\operatorname{Mean}(S_y)
=0.70759931
$$

影像垂直方向本來就有很明顯的亮度邊界，所以 \(e^{-|\partial I|}\) 會降低「深度一定要平滑」的要求，保留近車與遠路面的深度跳變。

### 8.5 單一尺度的總損失

論文使用 \(\lambda=10^{-3}\)：

$$
\mathcal L=\mathcal L_{ss}+\lambda\mathcal L_{smooth}
=0.0012615185+10^{-3}\times0.70759931
=0.0019691178
$$

真實模型會把四個尺度的 disparity 都先放大回完整解析度，重複上述完整損失流程，再平均：

$$
\mathcal L_{total}=\frac{1}{4}\sum_{s=1}^{4}
\left(\mathcal L_{ss}^{(s)}+10^{-3}\mathcal L_{smooth}^{(s)}\right)
$$

本例只有 2×2，無法再建立有意義的 1/2、1/4、1/8 影像，因此以完整解析度的一個尺度結束；真實四尺度只是對不同 decoder head 重複同一條資料流，沒有額外的損失種類。

---

## 原論文結果該怎麼讀？

下表是 MonoViT 論文在 KITTI Eigen split 公布的部分結果。`M` 表示只用 monocular video 訓練，`MS` 表示 monocular + stereo；Abs Rel、Sq Rel、RMSE、RMSE log 越低越好，\(\delta\) 指標越高越好。

| 訓練 | 解析度 | Abs Rel ↓ | Sq Rel ↓ | RMSE ↓ | RMSE log ↓ | \(\delta<1.25\) ↑ | \(\delta<1.25^2\) ↑ | \(\delta<1.25^3\) ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M | 640×192 | 0.099 | 0.708 | 4.372 | 0.175 | 0.900 | 0.967 | 0.984 |
| M | 1024×320 | 0.096 | 0.714 | 4.292 | 0.172 | 0.908 | 0.968 | 0.984 |
| M | 1280×384 | 0.094 | 0.682 | 4.200 | 0.170 | 0.912 | 0.969 | 0.984 |
| MS | 1024×320 | 0.093 | 0.671 | 4.202 | 0.169 | 0.912 | 0.969 | 0.985 |

這些數字表示 MonoViT 在 **2022 年論文發表時**優於表中比較的自監督方法，不代表它一直是目前所有單目深度方法的最新最佳結果。論文也在 Make3D 與 DrivingStereo 做跨資料集測試，用來支持 CNN／Transformer 混合編碼器的泛化能力。

---

## 限制與實務注意事項

### 單目影片有尺度不確定性

只靠單眼影片，同一段相機平移與整張深度同時乘上一個常數，仍可能產生相似投影。因此純 `M` 模型主要學到**相對深度**；若要直接得到可靠公尺尺度，需要 stereo baseline、已知相機運動、IMU、相機高度或少量 metric depth 等額外訊號。KITTI 單目評估通常會做 median scaling，部署時不能把這一步誤當成模型已天然知道絕對尺度。

### 重建假設不是永遠成立

- 獨立移動的車、人與反光表面不符合靜態世界假設。
- 遮擋區在另一張影格可能根本看不到。
- 曝光、陰影與天候改變會造成 photometric error，即使幾何是對的。
- 相機完全停止時，影像重建幾乎不提供深度訊號。

Minimum reprojection 與 auto-mask 能緩解問題，但不能徹底解決。

### 全域視野不是免費的

Factorized attention 比完整 \(N\times N\) attention 更省，但混合 Transformer encoder 仍比單純的輕量 CNN 複雜。若目標是低功耗即時裝置，應實測延遲、記憶體與輸入解析度，而不是只看 KITTI 誤差。

### 官方環境較舊

官方 README 列出的參考環境包含 Python 3.7、PyTorch 1.9、CUDA 11.1、舊版 MMCV／MMSegmentation。重現時需要處理相依套件相容性，或把網路結構移植到較新的 PyTorch 生態。

---

## 一句話總結

**MonoViT 用 CNN 看局部細節、用 factorized Transformer 看全域關係，再用相鄰影片影格能否重建目標影像來自我監督；推論時只留下 DepthNet，單張影像就能輸出稠密相對深度。**

---

## 參考資料

- 原始論文：[MonoViT: Self-Supervised Monocular Depth Estimation with a Vision Transformer](https://arxiv.org/abs/2208.03543)
- 官方實作：[zxcqlf/MonoViT](https://github.com/zxcqlf/MonoViT)
- 編碼器來源：[MPViT: Multi-Path Vision Transformer for Dense Prediction](https://arxiv.org/abs/2112.11010)
- 自監督訓練基礎：[Digging Into Self-Supervised Monocular Depth Estimation（Monodepth2）](https://arxiv.org/abs/1806.01260)
- Monodepth2 官方實作：[nianticlabs/monodepth2](https://github.com/nianticlabs/monodepth2)
