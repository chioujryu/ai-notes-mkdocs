# RMSNorm

> **Prerequisite knowledge 先備知識**
>
> * 了解 Transformer 的殘差連接（residual connection）與「很深的堆疊」會讓訓練變得不穩定
> * 知道常見的正規化（normalization）概念：把向量「縮放到比較穩定的尺度」
> * 會看懂基本符號：平均值 $\mu$、方差 $\sigma^2$、維度 $d$、小常數 $\epsilon$

## 1. 故事背景

### 1-1 深度模型的老問題：訊號尺度在路上越走越失控

想像你在玩傳話遊戲：每一層都會把訊息「加工」一次，再用殘差把原訊息加回來。層數一多，訊號的整體大小（向量長度、能量）很容易忽大忽小，導致：

* 梯度不穩，訓練忽然爆掉或收斂很慢
* 不同 batch、不同 token 的激活尺度差異很大，優化器很難用同一套步伐學習

於是大家常在每個子模組前後加 **LayerNorm**，讓每一步的尺度更可控，Transformer 訓練才穩得住。

### 1-2 LayerNorm 很好用，但「每一步都要算平均與方差」很貴

LayerNorm 的核心是：先把向量減掉平均值（re-centering），再用標準差做縮放（re-scaling）。

$$
\mu=\frac{1}{d}\sum_{i=1}^{d}x_i,\quad
\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$

$$
\mathrm{LayerNorm}(x)=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\odot\gamma+\beta
$$

在 2019 年，Zhang 與 Sennrich 提出一個關鍵觀察：**LayerNorm 的「減平均（re-centering）不一定是必要的」**，而且這一步會帶來額外計算與實作成本；他們因此提出 **RMSNorm**（Root Mean Square Layer Normalization）。 ([arXiv][1])

### 1-3 RMSNorm 的想法：只做「縮放」，不做「搬到零均值」

RMSNorm 用 RMS（均方根）衡量向量大小，做純粹的縮放：

$$
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}
$$

$$
\mathrm{RMSNorm}(x)=\frac{x}{\mathrm{RMS}(x)}\odot g
$$

作者主張：很多網路結構只需要「對尺度不敏感」（re-scaling invariance）就夠了，不一定需要「對平移不敏感」（re-centering invariance）；RMSNorm 因此更簡單也更快，並在多種模型上達到和 LayerNorm 相近的效果，還報告了不同模型上可觀的加速幅度。 ([arXiv][1])

## 2. 解決的痛點

### 2-1 痛點一：LayerNorm 的額外成本，會拖慢大型模型的吞吐

把「減平均、算方差、再開根號」想成每一層都要多做一套統計工作；模型越大、層越深、序列越長，這個成本越顯眼。

RMSNorm 省掉「減平均」與與其相關的一些計算與反傳複雜度，因此更容易做出高效 kernel；也因此在現代 LLM 堆疊裡很常見（例如很多模型採用 pre-norm + RMSNorm 的配置）。 ([arXiv][1])

### 2-2 痛點二：你不一定想把「平均值」洗掉，尤其在殘差訊號裡

用一個直覺例子來看差異。假設某層輸出向量 $x=[3,4]$，$d=2$：

* RMSNorm 只看能量大小
  $$
  \mathrm{RMS}(x)=\sqrt{\frac{3^2+4^2}{2}}=\sqrt{12.5}\approx 3.535
  $$
  $$
  \mathrm{RMSNorm}(x)\approx[0.849,\ 1.131]\odot g
  $$

* LayerNorm 會先減平均變成零均值
  $$
  \mu=\frac{3+4}{2}=3.5,\quad x-\mu=[-0.5,0.5]
  $$
  $$
  \sigma=\sqrt{\frac{(-0.5)^2+(0.5)^2}{2}}=0.5,\quad
  \mathrm{LayerNorm}(x)=[-1,1]\odot\gamma+\beta
  $$

你可以把「向量的平均值」想成殘差訊號中的某種共同偏移量：LayerNorm 會把它直接扣掉；RMSNorm 則保留方向，只把整體大小調到穩定範圍。近年的幾何觀點也強調：LayerNorm 的定義和「全 1 方向」有內在關聯，而 RMSNorm 的行為更像純粹控制向量長度。 ([arXiv][2])

### 2-3 痛點三：訓練時尺度亂飄，RMSNorm 像「自動音量控制」

把每層輸出想成麥克風音量：

* 音量太大會爆音（梯度爆炸、數值不穩）
* 音量太小會聽不到（梯度消失、學不動）

RMSNorm 透過 $\mathrm{RMS}(x)$ 把輸出自動縮放回「差不多的音量」，再用可學參數 $g$ 決定每個維度應該放大或縮小多少；原論文也提到這種機制能帶來類似「隱式學習率調節」的效果，幫助穩定優化。 ([arXiv][1])

### 2-4 小結：RMSNorm 到底解了什麼

* **更便宜的正規化**：少一步 re-centering，實作與計算更簡單 ([arXiv][1])
* **保留殘差訊號的某些特性**：不強制零均值，主要控制「尺度」 ([arXiv][2])
* **符合現代 LLM 的工程趨勢**：大量模型採用 RMSNorm，甚至出現針對 RMSNorm 的更快實作研究 ([arXiv][3])

[1]: https://arxiv.org/abs/1910.07467?utm_source=chatgpt.com "Root Mean Square Layer Normalization"
[2]: https://arxiv.org/html/2409.12951v2?utm_source=chatgpt.com "Geometric Interpretation of Layer Normalization and a ..."
[3]: https://arxiv.org/html/2407.09577v1?utm_source=chatgpt.com "Flash normalization: fast RMSNorm for LLMs"


## 3（實際案例運算）

### 3-1 真實場景設定：聊天機器人在 Transformer block 進入子模組前先做 RMSNorm

#### 3-1-1 輸入 hidden states（2 個 token，hidden size=4）

我們把一句話切成 2 個 token（例如 token1=「今天」、token2=「下雨」），每個 token 都有 4 維的 hidden state。以 **row 表示 token**、以 **column 表示特徵維度**。

$$
X\ (shape=2\times 4)=
\begin{bmatrix}
2.0 & -1.0 & 0.0 & 3.0\\
0.5 & 1.5 & -2.0 & 1.0
\end{bmatrix}
$$

痛點：把多個 token 的表示整理成一個可批次處理的矩陣，讓後續層能用相同流程穩定處理序列資料。

#### 3-1-2 設定 RMSNorm 的可學縮放參數 $g$（elementwise affine 的 scale）

RMSNorm 會用一個可學向量 $g$（每個 column 一個縮放係數）來調整每個維度的重要性；這裡先給一組「已訓練到一半」的數值。

$$
g\ (shape=1\times 4)=
\begin{bmatrix}
1.2 & 0.8 & 1.0 & 0.5
\end{bmatrix}
$$

痛點：保留「可學的幅度調整」，避免正規化把所有維度都強迫變得一樣重要，維持表徵能力。 ([docs.pytorch.org][1])

---

### 3-2 計算每個 token 的 RMS（只對最後一個維度做）

#### 3-2-1 RMSNorm 的核心公式（前向）

對每一個 row（每個 token 向量）計算：

$$
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}
$$

並做（先除 RMS，再乘上 $g$）：

$$
\mathrm{RMSNorm}(x)=\frac{x}{\mathrm{RMS}(x)}\odot g
$$

這裡 $d=4$，取 $\epsilon=10^{-5}$。 ([docs.pytorch.org][1])

痛點：只用 RMS 控制「尺度」就能讓每層輸入幅度更穩定，同時比 LayerNorm 少做「減平均」等操作，降低正規化的計算負擔。 ([arXiv][2])

#### 3-2-2 把每個 token 的 RMS 算成一個 column 向量

token1（row1）平方和平均後再開根號：

* row1：$[2.0,-1.0,0.0,3.0]$
* $\frac{1}{4}(2.0^2+(-1.0)^2+0.0^2+3.0^2)=\frac{1}{4}(4+1+0+9)=3.5$
* $\mathrm{RMS}(row1)=\sqrt{3.5+10^{-5}}\approx 1.870831$

token2（row2）：

* row2：$[0.5,1.5,-2.0,1.0]$
* $\frac{1}{4}(0.5^2+1.5^2+(-2.0)^2+1.0^2)=\frac{1}{4}(0.25+2.25+4+1)=1.875$
* $\mathrm{RMS}(row2)=\sqrt{1.875+10^{-5}}\approx 1.369310$

$$
\mathrm{RMS}(X)\ (shape=2\times 1)=
\begin{bmatrix}
1.870831\\
1.369310
\end{bmatrix}
$$

痛點：把每個 token 的向量長度量化成單一尺度因子，後面就能用同一套規則把不同 token 的幅度拉回可控範圍，避免激活值忽大忽小造成訓練不穩。

---

### 3-3 做 RMS 正規化（逐 row 除以 RMS）

#### 3-3-1 逐 row 做 $\frac{X}{\mathrm{RMS}(X)}$（elementwise 除法）

$$
\hat{X}\ (shape=2\times 4)=
\begin{bmatrix}
2.0/1.870831 & -1.0/1.870831 & 0.0/1.870831 & 3.0/1.870831\\
0.5/1.369310 & 1.5/1.369310 & -2.0/1.369310 & 1.0/1.369310
\end{bmatrix}
$$

計算後（四捨五入到小數點後 6 位）：

$$
\hat{X}\ (shape=2\times 4)=
\begin{bmatrix}
1.069043 & -0.534522 & 0.000000 & 1.603565\\
0.365147 & 1.095442 & -1.460590 & 0.730295
\end{bmatrix}
$$

痛點：讓每個 token 的向量整體幅度被「自動音量控制」，降低梯度爆炸或消失的風險，使深層 Transformer 更容易穩定訓練。 ([arXiv][2])

#### 3-3-2 乘上可學縮放 $g$（elementwise 乘法）

$$
Y\ (shape=2\times 4)=\hat{X}\odot g
$$

$$
Y\ (shape=2\times 4)=
\begin{bmatrix}
1.069043\times 1.2 & -0.534522\times 0.8 & 0.000000\times 1.0 & 1.603565\times 0.5\\
0.365147\times 1.2 & 1.095442\times 0.8 & -1.460590\times 1.0 & 0.730295\times 0.5
\end{bmatrix}
$$

計算後：

$$
Y\ (shape=2\times 4)=
\begin{bmatrix}
1.282852 & -0.427617 & 0.000000 & 0.801783\\
0.438177 & 0.876354 & -1.460590 & 0.365147
\end{bmatrix}
$$

痛點：在「穩定尺度」的同時，允許模型把某些 column 放大或縮小，避免過度正規化導致資訊被壓扁、表徵能力下降。 ([docs.pytorch.org][1])

---

### 3-4 接到真實子模組：線性投影（例如注意力的 Q/K/V 或 FFN 的第一層）

#### 3-4-1 設定線性層權重矩陣 $W$

這裡用一個把 hidden size=4 投影到 3 維的線性層（真實 Transformer 裡可能是 attention 的投影或 FFN 的投影）。

$$
W\ (shape=4\times 3)=
\begin{bmatrix}
0.1 & -0.2 & 0.0\\
0.0 & 0.3 & -0.1\\
0.2 & 0.0 & 0.1\\
-0.1 & 0.1 & 0.2
\end{bmatrix}
$$

痛點：把已經尺度穩定的 token 表徵映射到任務需要的子空間，讓後續模組能用更合適的維度組合做運算（例如注意力打分或非線性變換）。

#### 3-4-2 做矩陣乘法：$Y\cdot W=Z$

$$
Y\ (shape=2\times 4)=
\begin{bmatrix}
1.282852 & -0.427617 & 0.000000 & 0.801783\\
0.438177 & 0.876354 & -1.460590 & 0.365147
\end{bmatrix}
$$

$$
W\ (shape=4\times 3)=
\begin{bmatrix}
0.1 & -0.2 & 0.0\\
0.0 & 0.3 & -0.1\\
0.2 & 0.0 & 0.1\\
-0.1 & 0.1 & 0.2
\end{bmatrix}
$$

$$
Y\cdot W=Z
$$

$$
Z\ (shape=2\times 3)=
\begin{bmatrix}
0.048107 & -0.304677 & 0.203118\\
-0.284815 & 0.211785 & -0.160665
\end{bmatrix}
$$

痛點：因為前面 RMSNorm 已經把每個 token 的幅度穩住，線性層比較不會遇到「輸入尺度飄動」導致輸出突然過大或過小，讓整個 block 的前向訊號更可控、更穩定。 ([arXiv][2])

[1]: https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html?utm_source=chatgpt.com "RMSNorm — PyTorch 2.10 documentation"
[2]: https://arxiv.org/abs/1910.07467?utm_source=chatgpt.com "Root Mean Square Layer Normalization"
