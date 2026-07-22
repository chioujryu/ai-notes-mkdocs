# ReLU Family

ReLU family 是深度學習中最常用的 activation 類型之一。它的優點是計算簡單、梯度不容易在正半軸飽和，缺點是負半軸可能讓神經元長期輸出 0。

## ReLU

$$
\text{ReLU}(x)=\max(0,x)
$$

特性：

- 正半軸梯度為 1，訓練通常比 sigmoid / tanh 穩定。
- 負半軸輸出 0，會帶來稀疏 activation。
- 若某些神經元長期落在負半軸，可能出現 dying ReLU。

## Leaky ReLU

$$
\text{LeakyReLU}(x)=\max(\alpha x,x)
$$

Leaky ReLU 在負半軸保留小斜率，降低 dying ReLU 風險。常見 \(\alpha\) 是 0.01。

## PReLU

PReLU 和 Leaky ReLU 類似，但負半軸斜率 \(\alpha\) 由模型學習。它增加少量參數，適合在大型模型或視覺模型中嘗試。

## ELU / SELU

ELU 在負半軸使用指數型平滑曲線，能讓 activation mean 更接近 0。SELU 則設計給 self-normalizing network，但需要搭配特定初始化、AlphaDropout 等條件。

## 實務建議

- 一般 MLP 或 CNN 可以先用 ReLU 作 baseline。
- 若觀察到大量 dead neuron，可嘗試 Leaky ReLU 或 GELU / SiLU。
- Transformer 現代架構通常不以 ReLU 為首選，較常使用 GELU、SiLU 或 gated FFN。
