# 1 cross-entropy

## 1-1 真實場景：產線瑕疵三分類（OK / 刮傷 / 髒污）

### 1-1-1 把真實世界的「結果」變成模型能吃的輸入與標籤

想像你在產線上拍到一張 iPhone 外觀圖，前面的大模型（例如 CNN/VLM）先把影像濃縮成 4 個「可解釋特徵」：
$x_1$：刮痕線條強度、$x_2$：高頻紋理變化、$x_3$：反光不均、$x_4$：顆粒狀噪點

我們只看單筆樣本（batch size = 1），特徵向量如下：

$$
x\ (shape=1\times 4)=
\begin{bmatrix}
0.6 & 1.0 & -0.4 & 0.2
\end{bmatrix}
$$

這張圖的人工判定是「刮傷」，用 one-hot 標籤表示（類別順序：OK、刮傷、髒污）：

$$
y\ (shape=1\times 3)=
\begin{bmatrix}
0 & 1 & 0
\end{bmatrix}
$$

痛點：把真實世界的類別結論轉成「可計算的監督訊號」，才能訓練模型對齊人類判定。 ([維基百科][1])

### 1-1-2 線性分類器：把特徵轉成各類別分數（logits）

我們用最簡單的最後一層：線性層 $z = xW + b$（這裡 $z$ 叫 logits/score）。([cs231n.stanford.edu][2])

權重矩陣與偏置如下：

$$
W\ (shape=4\times 3)=
\begin{bmatrix}
0.5 & -0.3 & 0.1\\
0.2 & 0.4 & -0.5\\
-0.6 & 0.2 & 0.3\\
0.1 & -0.2 & 0.7
\end{bmatrix}
$$

$$
b\ (shape=1\times 3)=
\begin{bmatrix}
0.44 & 0.20 & -0.38
\end{bmatrix}
$$

先做矩陣乘法：

$$
x\ (shape=1\times 4)\cdot W\ (shape=4\times 3)=s\ (shape=1\times 3)
$$

$$
s\ (shape=1\times 3)=
\begin{bmatrix}
0.76 & 0.10 & -0.42
\end{bmatrix}
$$

再加上偏置得到 logits：

$$
s\ (shape=1\times 3)+b\ (shape=1\times 3)=z\ (shape=1\times 3)
$$

$$
z\ (shape=1\times 3)=
\begin{bmatrix}
1.20 & 0.30 & -0.80
\end{bmatrix}
$$

痛點：把高維特徵壓成「每個類別一個分數」，讓模型能在多類別之間做可比較的決策。 ([cs231n.stanford.edu][2])

## 1-2 softmax：把 logits 變成機率分佈

### 1-2-1 數值穩定：先減掉最大 logit（避免 exp 爆掉）

softmax 會用到 $e^{z_i}$，若 $z$ 很大容易數值溢位，所以常做「減最大值」的等價變形來穩定計算。([Cross Validated][3])

先取最大值（這裡用矩陣表示成 $1\times 1$）：

$$
m\ (shape=1\times 1)=
\begin{bmatrix}
1.20
\end{bmatrix}
$$

做位移（broadcast 概念：同一個 $m$ 減到每個欄位）：

$$
z\ (shape=1\times 3)-m\ (shape=1\times 1)=z'\ (shape=1\times 3)
$$

$$
z'\ (shape=1\times 3)=
\begin{bmatrix}
0.00 & -0.90 & -2.00
\end{bmatrix}
$$

痛點：避免指數運算溢位，讓推論與訓練在大分數時也能穩定不炸掉。 ([Cross Validated][3])

### 1-2-2 指數與歸一化：得到每一類的預測機率

先做逐元素指數：

$$
e\ (shape=1\times 3)=\exp(z')\ (shape=1\times 3)
$$

$$
e\ (shape=1\times 3)=
\begin{bmatrix}
1.0000 & 0.4066 & 0.1353
\end{bmatrix}
$$

再把它們加總成分母：

$$
t\ (shape=1\times 1)=
\begin{bmatrix}
1.5419
\end{bmatrix}
$$

做歸一化得到機率（總和為 1）：

$$
e\ (shape=1\times 3)/t\ (shape=1\times 1)=p\ (shape=1\times 3)
$$

$$
p\ (shape=1\times 3)=
\begin{bmatrix}
0.6487 & 0.2637 & 0.0876
\end{bmatrix}
$$

痛點：把「分數」轉成可解讀、可比較、總和為 1 的機率分佈，方便做風險判斷與決策。 ([cs231n.github.io][4])

## 1-3 cross-entropy：用「正確類別的機率」計算損失

### 1-3-1 把機率轉成損失：$-\log(p_{\text{true}})$（one-hot 會自動挑中正確類）

cross-entropy（分類常用的形式）可以寫成：

$$
L=-\sum_{c=1}^{3} y_c\log(p_c)
$$

在 one-hot 標籤下，等價於 $L=-\log(p_{\text{刮傷}})$。([維基百科][1])

先算 $\log(p)$：

$$
\log(p)\ (shape=1\times 3)=
\begin{bmatrix}
-0.4329 & -1.3330 & -2.4340
\end{bmatrix}
$$

轉置成欄向量：

$$
\log(p)^T\ (shape=3\times 1)=
\begin{bmatrix}
-0.4329\\
-1.3330\\
-2.4340
\end{bmatrix}
$$

用矩陣乘法「挑出」正確類別那一項（因為 $y=[0,1,0]$）：

$$
y\ (shape=1\times 3)\cdot \log(p)^T\ (shape=3\times 1)=u\ (shape=1\times 1)
$$

$$
u\ (shape=1\times 1)=
\begin{bmatrix}
-1.3330
\end{bmatrix}
$$

最後取負號得到 cross-entropy loss：

$$
L\ (shape=1\times 1)= -u\ (shape=1\times 1)
$$

$$
L\ (shape=1\times 1)=
\begin{bmatrix}
1.3330
\end{bmatrix}
$$

痛點：當模型把「正確類別」機率壓得很低時，會被用對數強力懲罰，逼模型把信心拉回正確答案。 ([維基百科][1])

### 1-3-2 直覺檢查：同樣是「刮傷」，機率越高 loss 越小

如果模型更有把握（$p_{\text{刮傷}}$ 更大），$-\log(\cdot)$ 就更小：

$$
p_{\text{刮傷}}=0.90\ (shape=1\times 1)=
\begin{bmatrix}
0.90
\end{bmatrix}
$$

$$
L=-\log(0.90)\ (shape=1\times 1)=
\begin{bmatrix}
0.1053
\end{bmatrix}
$$

反過來如果幾乎不信（$p_{\text{刮傷}}$ 很小），loss 會變大：

$$
p_{\text{刮傷}}=0.10\ (shape=1\times 1)=
\begin{bmatrix}
0.10
\end{bmatrix}
$$

$$
L=-\log(0.10)\ (shape=1\times 1)=
\begin{bmatrix}
2.3026
\end{bmatrix}
$$

痛點：把「信心程度」壓縮成單一可優化指標，讓訓練能穩定朝提升正確類別機率前進。 ([維基百科][1])

[1]: https://en.wikipedia.org/wiki/Cross-entropy?utm_source=chatgpt.com "Cross-entropy"
[2]: https://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture03.pdf?utm_source=chatgpt.com "Lecture 3: Loss Functions and Optimization"
[3]: https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning?utm_source=chatgpt.com "How does the subtraction of the logit maximum improve ..."
[4]: https://cs231n.github.io/linear-classify/?utm_source=chatgpt.com "Linear Classification"
