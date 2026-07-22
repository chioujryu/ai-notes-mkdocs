# GLU Family (GEGLU / SwiGLU)

GLU family 透過 gate 控制資訊流，是 Transformer feed-forward network 常見改良。相較單純的 activation，GLU 會把輸入分成 value 與 gate 兩路，再做逐元素相乘。

## GLU

$$
\text{GLU}(a,b)=a\otimes\sigma(b)
$$

其中 \(a\) 是 value，\(b\) 是 gate，\(\otimes\) 是 element-wise multiplication。

## GEGLU

$$
\text{GEGLU}(a,b)=a\otimes\text{GELU}(b)
$$

GEGLU 將 sigmoid gate 換成 GELU gate，常見於 T5 變體與部分 Transformer FFN。

## SwiGLU

$$
\text{SwiGLU}(a,b)=a\otimes\text{SiLU}(b)
$$

SwiGLU 使用 SiLU gate，是許多現代 LLM 的常見選擇。

## 為什麼有效

- Gate 可以根據上下文選擇哪些 hidden feature 要通過。
- 乘法交互比單一路徑 activation 有更強表達能力。
- 在 Transformer FFN 中通常能提升品質，但會改變 hidden dimension 與參數量配置。

## 實務建議

- 若復現既有 LLM 架構，依原 paper 或 checkpoint 使用 GEGLU / SwiGLU。
- 從 ReLU/GELU FFN 改為 SwiGLU 時，要同步調整 intermediate size，避免參數量暴增。
- 小模型或簡單任務不一定需要 gate，GELU baseline 通常已足夠。
