# Modern / Other Activation Functions

現代模型除了 ReLU、GELU、SiLU 之外，也會依架構需求使用其他 activation。選擇 activation 時，重點不是單一函數是否最新，而是它是否符合模型結構、硬體成本與訓練穩定性。

## Mish

$$
\text{Mish}(x)=x\tanh(\ln(1+e^x))
$$

Mish 是平滑、非單調 activation，曾在部分視覺模型中使用。它保留負半軸訊號，但計算成本高於 ReLU。

## Hard Sigmoid / Hard Swish

Hard 版本使用分段線性函數近似 sigmoid 或 swish，常用在行動端模型中降低計算成本。

用途：

- MobileNet 類模型。
- 需要兼顧精度與推論速度的 edge deployment。

## Maxout

Maxout 從多個線性分支中取最大值，表達能力強，但參數與計算量較高。它在現代大型架構中不如 GELU / SiLU 常見。

## Identity

有些位置會刻意不使用 activation，例如 residual projection、最後一層 logits head，或 normalization 前後的線性投影。這不是遺漏，而是避免不必要地限制輸出範圍。

## 選擇建議

- 追求簡單與速度：ReLU。
- Transformer / LLM：GELU、GEGLU、SwiGLU。
- 視覺模型：ReLU、SiLU、Hard Swish 視架構而定。
- 輸出層：依任務使用 logits、sigmoid 或 softmax，不要和 loss 重複套用。
