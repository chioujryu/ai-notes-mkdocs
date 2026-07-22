# Activation Functions Overview

Activation function 讓神經網路在多層線性轉換之間加入非線性。沒有 activation，多層 MLP 或 Transformer feed-forward network 最後仍等價於一個線性模型，無法表達複雜決策邊界。

## 核心用途

- 引入非線性，使模型能學到高階特徵。
- 控制梯度傳遞，影響訓練穩定性與收斂速度。
- 改變輸出範圍，例如 sigmoid 會壓到 0 到 1，tanh 會壓到 -1 到 1。
- 影響稀疏性與計算成本，例如 ReLU 會產生大量 0。

## 常見選擇

| 類型 | 常見函數 | 典型用途 |
| --- | --- | --- |
| 壓縮型 | Sigmoid, Tanh | 二元輸出、門控、舊式 RNN |
| 分段線性 | ReLU, Leaky ReLU, PReLU | CNN、MLP、一般深度網路 |
| 平滑型 | GELU, SiLU | Transformer、現代 LLM |
| 門控型 | GLU, GEGLU, SwiGLU | Transformer FFN 改良 |
| 機率型 | Softmax | 多類別分類、attention weights |

## 實務原則

- 隱藏層預設可從 ReLU、GELU 或 SiLU 開始。
- Transformer 類模型常用 GELU、SwiGLU 或 GEGLU。
- 多類別分類輸出層通常使用 softmax。
- 二元分類輸出層通常輸出 logit，搭配 BCEWithLogits 類 loss，而不是先手動 sigmoid 再計算 loss。
