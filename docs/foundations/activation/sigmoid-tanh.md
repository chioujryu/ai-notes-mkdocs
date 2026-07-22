# Sigmoid / Tanh

Sigmoid 和 tanh 都是早期神經網路常見的平滑 activation。它們會把輸入壓縮到固定範圍，因此適合做輸出轉換或門控，但在深層網路中容易遇到梯度飽和。

## Sigmoid

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

輸出範圍是 \(0, 1\)。當輸入非常大或非常小時，輸出會接近 1 或 0，梯度接近 0。

常見用途：

- 二元機率解讀。
- LSTM / GRU 等 recurrent model 的 gate。
- 多標籤分類中，每個 label 各自做 independent probability。

## Tanh

$$
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

輸出範圍是 \(-1, 1\)，且以 0 為中心。相較 sigmoid，tanh 的 hidden activation 通常更容易優化，但仍會在大幅度輸入時飽和。

## 梯度飽和問題

當 activation 進入飽和區時，反向傳播的梯度會變小。深層網路堆疊很多 sigmoid 或 tanh 時，前面幾層可能幾乎收不到有效梯度，導致訓練很慢。

## 實務建議

- 隱藏層通常優先使用 ReLU、GELU 或 SiLU。
- 若 loss 函數已包含 sigmoid，例如 BCEWithLogitsLoss，不要在模型輸出端再手動套 sigmoid。
- 在 gate 或需要有界輸出的地方，sigmoid / tanh 仍然合理。
