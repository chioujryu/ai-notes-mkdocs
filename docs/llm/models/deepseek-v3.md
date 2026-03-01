# Deepseek-v3

> 先備知識
>
> * Transformer 的注意力機制與「KV cache」為什麼會吃顯存
> * MoE（Mixture-of-Experts）裡「總參數」與「每個 token 啟用參數」的差別
> * 路由（routing）、負載平衡（load balancing）、routing collapse 的基本概念
> * 低精度訓練（BF16、FP8）與數值穩定性的取捨
> * 解碼（decoding）速度與 Tokens Per Second（TPS）的意義

## 1. 故事背景

### 1-1 想做出接近閉源巨頭水準的模型，但訓練成本高到不敢想

把情境放在一間想做「可用、可部署、可商用」的開源團隊：你想把模型能力拉到能跟 GPT-4o、Claude 3.5 這類閉源模型同台比較，但只要沿用傳統大模型訓練方式，算力與成本很快就成為天花板。DeepSeek-V3 的報告開宗明義把目標鎖在「強性能」與「經濟成本」同時成立，並給出完整訓練只需 $2.788\text{M}$ H800 GPU hours 的量級，甚至用租用價假設換算到總成本。 

### 1-2 規模要更大，MoE 是捷徑，但「負載不均」會讓捷徑變陷阱

MoE 的誘惑很大：總參數可以衝很高，但每個 token 只啟用一小部分專家，計算量更像跟「啟用參數」走。然而 MoE 常見問題是專家負載不均，嚴重時會發生 routing collapse，導致訓練與推論效率都下降；傳統做法用 auxiliary loss 強推平衡，卻可能傷害模型表現。DeepSeek-V3 特別把「auxiliary-loss-free」當成核心改動之一，就是在這個背景下出現的。 ([arXiv][1])

### 1-3 長上下文與高吞吐是產品需求，但注意力顯存與解碼速度卡住落地

如果你的產品是「讀超長需求文件的工程助理」或「把公司知識庫整包吞進去的客服助理」，上下文長度拉高後，KV cache 會快速膨脹；另一方面，就算模型很準，解碼 TPS 太慢也會讓使用者覺得「像在等人打字」。DeepSeek-V3 的設計同時瞄準了兩個瓶頸：用 MLA 降 KV cache 壓力、用 Multi-Token Prediction（MTP）讓解碼加速。 ([arXiv][1])

## 2. 解決的痛點

### 2-1 痛點：KV cache 太大導致長上下文推論吃不消；解法：MLA 做低秩壓縮

在一般多頭注意力（MHA）下，KV cache 的量級可以用下面的直覺來記：

$$
\text{KVCache}*{\text{MHA}} \propto L \cdot N*{\text{layer}} \cdot n_h \cdot d_h
$$

其中 $L$ 是上下文長度、$N_{\text{layer}}$ 是層數、$n_h$ 是 head 數、$d_h$ 是每個 head 維度。

DeepSeek-V3 採用 Multi-head Latent Attention（MLA），把 key/value 用低秩的「latent 向量」表示，讓生成時需要快取的向量更少，從而「顯著降低 KV cache」但仍維持與 MHA 相近的性能。 ([arXiv][1])

用產品語言講：同樣要讀很長的文件，MLA 讓你不必為了上下文長度把 GPU 顯存堆到不可思議，長上下文才更像「能部署」而不是「只能展示」。

### 2-2 痛點：MoE 負載平衡要靠輔助損失，卻常犧牲品質；解法：auxiliary-loss-free + 不丟 token

DeepSeek-V3 的做法是：在路由時對每個 expert 加一個可調的 bias，讓 top-$K$ 選擇看的是「親和度 + bias」，bias 依照每個 batch 的負載情況動態調整，達到平衡但不必用大力的 auxiliary loss 硬推。可用下面的簡化式子理解：

$$
s'*{i,e} = s*{i,e} + b_e,\quad \text{Top-}K\ \text{routing based on } s'*{i,e}
$$

而 FFN 輸出仍由原本的 gating（來自 $s*{i,e}$）決定，bias 主要用在「分流」上。 ([arXiv][1])

更關鍵的是，因為負載平衡做得夠好，DeepSeek-V3 明確指出訓練與推論都「不需要 token-dropping」。這很像物流系統：以前為了不塞車只好「丟包裹」，現在用更好的分流策略讓包裹不必被丟掉，品質與穩定性就會更好。 ([arXiv][1])

### 2-3 痛點：解碼太慢；解法：MTP 一次多預測一個 token，配合 speculative decoding 加速

DeepSeek-V3 設定 MTP 深度 $D=1$，也就是每個位置除了「下一個 token」之外，還會額外預測「再下一個 token」。在推論端，這可以和 speculative decoding 類框架結合，讓模型有機會在一次步驟裡「吃下兩個 token」。 

報告提供了實測：額外預測的第二個 token 接受率約落在 $85%$ 到 $90%$，並帶來約 $1.8\times$ 的 TPS 提升。 

用一個直覺公式抓重點：如果第二個 token 被接受的機率是 $r$，那平均每步吃到的 token 數大致是

$$
\mathbb{E}[\text{tokens/step}] \approx 1 + r
$$

當 $r \in [0.85, 0.90]$ 時，就會很接近「接近兩倍、但有折損」的加速感，這就是使用者體感變快的來源。

### 2-4 痛點：超大模型訓練成本與穩定性；解法：FP8 混合精度 + 工程化訓練效率

DeepSeek-V3 在訓練端主打 FP8 mixed precision：把大量 GEMM 這種高算力密度操作放到 FP8，同時把某些敏感模組（例如 embedding、output head、gating、norm、attention）保留在較高精度以維持穩定。報告也提到相對於 BF16 baseline，FP8 訓練的相對 loss 誤差可以維持在 $0.25%$ 以內。 ([arXiv][1])

在成本面，報告把完整訓練拆成預訓練、長上下文延伸、後訓練，合計約 $2.788\text{M}$ H800 GPU hours；並強調整體訓練過程「沒有不可恢復的 loss spike、也不需要 rollback」。 

### 2-5 痛點：推理能力要強但又想控制輸出風格；解法：從 R1 系列蒸餾推理並保留可控性

DeepSeek-V3 的後訓練階段提到，會把 DeepSeek-R1 系列（長 Chain-of-Thought 模型）的推理能力蒸餾進 DeepSeek-V3，並把「驗證與反思」這類模式融入，同時維持輸出風格與長度的控制。這解決了實務上的矛盾：你想要更強的推理，但又不希望模型在所有情境都輸出冗長推理。 ([GitHub][2])

[1]: https://arxiv.org/html/2412.19437v2 "DeepSeek-V3 Technical Report"
[2]: https://github.com/deepseek-ai/DeepSeek-V3 "GitHub - deepseek-ai/DeepSeek-V3"
