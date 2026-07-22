# Softmax

Softmax 會把一組 logit 轉成機率分布，所有輸出非負且總和為 1。它常用於多類別分類與 attention 權重計算。

## 定義

對向量 \(z\)：

$$
\text{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

較大的 logit 會得到較高機率，但 softmax 仍會保留其他類別的非零機率。

## 多類別分類

分類模型通常輸出 logits，訓練時直接交給 cross entropy loss。多數框架的 cross entropy 已包含 log-softmax，因此不需要在模型內先手動 softmax。

推薦：

- 模型輸出 raw logits。
- loss 使用 CrossEntropyLoss 類型。
- 推論時需要機率再套 softmax。

## Attention

Transformer attention 會先計算 query-key 分數，再使用 softmax 轉成權重：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

softmax 讓每個 token 對其他 token 的注意力權重形成分布。

## 數值穩定性

直接計算 \(e^z\) 可能 overflow。實作上通常會先減掉最大值：

$$
\text{softmax}(z_i)=\frac{e^{z_i-\max(z)}}{\sum_j e^{z_j-\max(z)}}
$$

## 實務建議

- 訓練分類模型時輸出 logits，不要先 softmax 再丟進 cross entropy。
- attention mask 通常透過把不允許的位置加上很大的負數，使 softmax 後接近 0。
- temperature 會改變分布尖銳度，常見於 sampling 與 distillation。
