# BCELoss

## 1、真實場景與模型輸入

想像你在做「信用卡詐欺偵測」：每筆交易要判斷是不是詐欺（1=詐欺，0=正常）。模型會輸出「這筆交易是詐欺的機率」。

我們一次看 3 筆交易（batch size = 3），每筆交易都有一個真實標籤 $y$，模型輸出一個機率 $\hat{y}$。

$$
y\ (shape=3\times 1)=
\begin{bmatrix}
1\\
0\\
1
\end{bmatrix}
$$

$$
\hat{y}\ (shape=3\times 1)=
\begin{bmatrix}
0.90\\
0.20\\
0.40
\end{bmatrix}
$$

痛點：把真實世界的「是/否」決策變成可訓練的數值目標，讓模型能被監督學習推動改進。

---

## 2、從 logits 走到機率（前向傳播）

在真實模型裡，最後一層常先輸出 logits $z$（未經壓縮的分數），再用 sigmoid 變成機率：

$$
\hat{y}=\sigma(z)=\frac{1}{1+e^{-z}}
$$

我們這次示範一組 logits，經 sigmoid 後會對應到上面的機率（近似即可）：

$$
z\ (shape=3\times 1)=
\begin{bmatrix}
2.20\\
-1.39\\
-0.41
\end{bmatrix}
$$

$$
\hat{y}=\sigma(z)\ (shape=3\times 1)=
\begin{bmatrix}
0.90\\
0.20\\
0.40
\end{bmatrix}
$$

痛點：把任意實數分數壓到 $[0,1]$，才能對應「機率」並與標籤做可比對的損失計算。

---

## 3、BCE 的逐筆損失公式（前向計算）

Binary Cross Entropy（BCE）對每一筆樣本的損失定義為：

$$
\ell_i=-\Big(y_i\log(\hat{y}_i)+(1-y_i)\log(1-\hat{y}_i)\Big)
$$

我們把每筆交易代入（只做前向數值計算）：

### 3-1、第 1 筆（$y_1=1,\ \hat{y}_1=0.90$）

$$
\ell_1=-\log(0.90)=0.1053605
$$

痛點：當真實是正例時，強烈懲罰把機率壓低的模型，逼它「正例要更像正例」。

### 3-2、第 2 筆（$y_2=0,\ \hat{y}_2=0.20$）

$$
\ell_2=-\log(1-0.20)=-\log(0.80)=0.2231436
$$

痛點：當真實是負例時，強烈懲罰把機率抬高的模型，減少誤報造成的人工審核成本。

### 3-3、第 3 筆（$y_3=1,\ \hat{y}_3=0.40$）

$$
\ell_3=-\log(0.40)=0.9162907
$$

痛點：把「信心不足」的正例（機率偏低）放大懲罰，推動模型把關鍵特徵學得更清楚。

---

## 4、把逐筆損失組成向量

我們把每筆損失排成一個向量（方便之後做 batch 平均）：

$$
\ell\ (shape=3\times 1)=
\begin{bmatrix}
0.1053605\\
0.2231436\\
0.9162907
\end{bmatrix}
$$

痛點：把單筆誤差整理成批次形式，方便穩定地統計與監控訓練狀態。

---

## 5、Batch 平均 BCE Loss（最常用）

整個 batch 的 BCE loss 常用平均：

$$
L=\frac{1}{N}\sum_{i=1}^{N}\ell_i
$$

這裡 $N=3$：

$$
L\ (shape=1\times 1)=
\begin{bmatrix}
\frac{0.1053605+0.2231436+0.9162907}{3}
\end{bmatrix}
=
\begin{bmatrix}
0.4149316
\end{bmatrix}
$$

痛點：用平均讓 loss 不會因 batch 大小改變而失真，訓練更穩定、可比較不同設定。

---

## 6、用「矩陣視角」看 BCE（打包成同一個式子）

我們也可以把 $y$ 與 $\hat{y}$ 用元素運算打包成矩陣形式（仍是前向）：

先算 $\log(\hat{y})$ 與 $\log(1-\hat{y})$：

$$
\log(\hat{y})\ (shape=3\times 1)=
\begin{bmatrix}
\log(0.90)\\
\log(0.20)\\
\log(0.40)
\end{bmatrix}
=
\begin{bmatrix}
-0.1053605\\
-1.6094379\\
-0.9162907
\end{bmatrix}
$$

$$
\log(1-\hat{y})\ (shape=3\times 1)=
\begin{bmatrix}
\log(0.10)\\
\log(0.80)\\
\log(0.60)
\end{bmatrix}
=
\begin{bmatrix}
-2.3025851\\
-0.2231436\\
-0.5108256
\end{bmatrix}
$$

接著用 BCE 的結構把它們組合（元素乘法與加法，仍是逐元素的前向組合）：

$$
\ell\ (shape=3\times 1)=
-\Big(
y\odot \log(\hat{y})+(1-y)\odot \log(1-\hat{y})
\Big)
$$

對應到我們前面算出的：

$$
\ell\ (shape=3\times 1)=
\begin{bmatrix}
0.1053605\\
0.2231436\\
0.9162907
\end{bmatrix}
$$

痛點：把正負例兩種情況用同一個公式統一處理，實作簡潔且避免分支錯誤，訓練流程更可靠。