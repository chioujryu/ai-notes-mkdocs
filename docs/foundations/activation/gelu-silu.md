# GELU / SiLU (Swish)

GELU 和 SiLU 是現代 Transformer 與 LLM 中常見的平滑 activation。它們不像 ReLU 那樣硬切斷負半軸，而是以平滑方式保留部分負值訊號。

## GELU

$$
\text{GELU}(x)=x\Phi(x)
$$

\(\Phi(x)\) 是標準常態分布的 CDF。直覺上，GELU 會根據輸入大小以機率方式保留訊號。

常見用途：

- BERT、GPT 類 Transformer 的 feed-forward network。
- 需要平滑 activation 的語言模型與多模態模型。

## SiLU / Swish

$$
\text{SiLU}(x)=x\sigma(x)
$$

SiLU 也稱為 Swish。它在負半軸保留小幅訊號，在正半軸近似線性。

常見用途：

- EfficientNet、YOLO 等視覺模型。
- LLaMA 系列常見的 SwiGLU 變體中會用到 SiLU。

## 與 ReLU 的差異

- GELU / SiLU 是平滑函數，梯度變化更連續。
- 負半軸不會全部變成 0，資訊保留較多。
- 計算成本略高於 ReLU，但在大型模型中通常可接受。

## 實務建議

- Transformer 類模型可優先使用 GELU 或 gated activation。
- CNN 或 detection 模型若已採用 SiLU，通常維持原架構選擇。
- 若追求極低延遲，ReLU 仍可能是更便宜的 baseline。
