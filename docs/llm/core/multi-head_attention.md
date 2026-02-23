# Attention

![1768612680096](image/attention/1768612680096.png)


# 1 multi head attention

## 1-1 真實場景與輸入

### 1-1-1 真實場景：外送客服同一句話要同時抓「原因」與「賠償政策」
客人說：「外送 晚到 補償」。系統要一邊理解「晚到」是事件原因，一邊也要連到「外送」對應的賠償規則；如果只有單一注意力，容易只抓到其中一個面向而漏掉另一個。  
痛點：同一句短訊裡有多個重點時，避免只關注單一面向而漏資訊。

### 1-1-2 輸入向量矩陣 $X$
我們用 3 個 token（外送、晚到、補償），每個 token 用 4 維向量表示（教學用小維度）。

$$
X\ (shape=3\times 4)=
\begin{bmatrix}
1 & 0 & 0 & 1\\
0 & 1 & 1 & 0\\
1 & 1 & 0 & 0
\end{bmatrix}
$$
痛點：把文字轉成可運算的向量，才能用矩陣運算做關聯與聚合。

## 1-2 多頭注意力的總覽（只做前向傳播）

### 1-2-1 Multi-Head Attention 的前向形式（本例做 self-attention）
我們做 2 個 head（$h=2$），每個 head 都做一次 attention，最後把兩個 head 的輸出串接（concat）再做一次線性投影。

$$
\mathrm{head}^{(i)}=\mathrm{Attention}\left(Q^{(i)},K^{(i)},V^{(i)}\right)
$$

$$
O_{\mathrm{concat}}=\mathrm{Concat}\left(\mathrm{head}^{(1)},\mathrm{head}^{(2)}\right)
$$

$$
Y=O_{\mathrm{concat}}W_O+b_O
$$

痛點：讓模型能用不同「角度」同時看同一句話，分工捕捉不同關係。

### 1-2-2 本例維度設定
- 序列長度 $n=3$
- 模型維度 $d_{\mathrm{model}}=4$
- head 數 $h=2$
- 每個 head：$d_k=d_v=2$
- 縮放常數：$\sqrt{d_k}=\sqrt{2}=1.414214$

痛點：把高維表示切成多個小子空間，降低單一注意力過度擁擠、表達力不足的問題。

## 1-3 Head 1：投影得到 $Q^{(1)},K^{(1)},V^{(1)}$

### 1-3-1 Head 1 的 $W_Q^{(1)},b_Q^{(1)}$
$$
W_Q^{(1)}\ (shape=4\times 2)=
\begin{bmatrix}
1 & 0\\
0 & 1\\
0 & 0\\
0 & 0
\end{bmatrix}
$$

$$
b_Q^{(1)}\ (shape=1\times 2)=
\begin{bmatrix}
0 & 0
\end{bmatrix}
$$
痛點：讓每個 head 用自己的投影方式，形成不同的「提問方式」（Query）。

### 1-3-2 計算 $Q^{(1)}=XW_Q^{(1)}+b_Q^{(1)}$
$$
X\ (shape=3\times 4)\cdot W_Q^{(1)}\ (shape=4\times 2)=Q_{\mathrm{raw}}^{(1)}\ (shape=3\times 2)
$$

$$
Q_{\mathrm{raw}}^{(1)}\ (shape=3\times 2)=
\begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{bmatrix}
$$

$$
B_Q^{(1)}\ (shape=3\times 2)=
\begin{bmatrix}
0 & 0\\
0 & 0\\
0 & 0
\end{bmatrix}
$$

$$
Q_{\mathrm{raw}}^{(1)}\ (shape=3\times 2)+B_Q^{(1)}\ (shape=3\times 2)=Q^{(1)}\ (shape=3\times 2)
$$

$$
Q^{(1)}\ (shape=3\times 2)=
\begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{bmatrix}
$$
痛點：讓句子中每個位置都能生成自己的 Query，後面才能「各自問各自要的資訊」。

### 1-3-3 Head 1 的 $W_K^{(1)},b_K^{(1)}$
（設計成讓「補償」更容易對到「晚到」的索引）

$$
W_K^{(1)}\ (shape=4\times 2)=
\begin{bmatrix}
-0.707107 & -0.707107\\
0.707107 & 0.707107\\
0 & 0\\
0 & 0
\end{bmatrix}
$$

$$
b_K^{(1)}\ (shape=1\times 2)=
\begin{bmatrix}
0.707107 & 0.707107
\end{bmatrix}
$$
痛點：把「可被比對的索引」獨立建出來，讓關鍵關係更容易在相似度裡浮現。

### 1-3-4 計算 $K^{(1)}=XW_K^{(1)}+b_K^{(1)}$
$$
X\ (shape=3\times 4)\cdot W_K^{(1)}\ (shape=4\times 2)=K_{\mathrm{raw}}^{(1)}\ (shape=3\times 2)
$$

$$
K_{\mathrm{raw}}^{(1)}\ (shape=3\times 2)=
\begin{bmatrix}
-0.707107 & -0.707107\\
0.707107 & 0.707107\\
0 & 0
\end{bmatrix}
$$

$$
B_K^{(1)}\ (shape=3\times 2)=
\begin{bmatrix}
0.707107 & 0.707107\\
0.707107 & 0.707107\\
0.707107 & 0.707107
\end{bmatrix}
$$

$$
K_{\mathrm{raw}}^{(1)}\ (shape=3\times 2)+B_K^{(1)}\ (shape=3\times 2)=K^{(1)}\ (shape=3\times 2)
$$

$$
K^{(1)}\ (shape=3\times 2)=
\begin{bmatrix}
0 & 0\\
1.414214 & 1.414214\\
0.707107 & 0.707107
\end{bmatrix}
$$
痛點：建立一組「可快速搜尋/比對」的 key 表徵，提升關聯對齊的可控性。

### 1-3-5 Head 1 的 $W_V^{(1)},b_V^{(1)}$
$$
W_V^{(1)}\ (shape=4\times 2)=
\begin{bmatrix}
1 & 0\\
0 & 1\\
0 & 0\\
0 & 0
\end{bmatrix}
$$

$$
b_V^{(1)}\ (shape=1\times 2)=
\begin{bmatrix}
0 & 0
\end{bmatrix}
$$
痛點：把「要被加權帶走的內容」與 Key 分離，避免索引與內容互相污染。

### 1-3-6 計算 $V^{(1)}=XW_V^{(1)}+b_V^{(1)}$
$$
X\ (shape=3\times 4)\cdot W_V^{(1)}\ (shape=4\times 2)=V_{\mathrm{raw}}^{(1)}\ (shape=3\times 2)
$$

$$
V_{\mathrm{raw}}^{(1)}\ (shape=3\times 2)=
\begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{bmatrix}
$$

$$
B_V^{(1)}\ (shape=3\times 2)=
\begin{bmatrix}
0 & 0\\
0 & 0\\
0 & 0
\end{bmatrix}
$$

$$
V_{\mathrm{raw}}^{(1)}\ (shape=3\times 2)+B_V^{(1)}\ (shape=3\times 2)=V^{(1)}\ (shape=3\times 2)
$$

$$
V^{(1)}\ (shape=3\times 2)=
\begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{bmatrix}
$$
痛點：準備好可被加權聚合的內容向量，讓注意力真正「搬運資訊」而不是只算分數。

## 1-4 Head 1：Scaled Dot-Product Attention

### 1-4-1 轉置 $K^{(1)T}$
$$
K^{(1)T}\ (shape=2\times 3)=
\begin{bmatrix}
0 & 1.414214 & 0.707107\\
0 & 1.414214 & 0.707107
\end{bmatrix}
$$
痛點：把 Key 排成可一次對全句做相似度計算的形狀，提升計算效率。

### 1-4-2 分數矩陣 $S_{\mathrm{raw}}^{(1)}=Q^{(1)}K^{(1)T}$
$$
Q^{(1)}\ (shape=3\times 2)\cdot K^{(1)T}\ (shape=2\times 3)=S_{\mathrm{raw}}^{(1)}\ (shape=3\times 3)
$$

$$
S_{\mathrm{raw}}^{(1)}\ (shape=3\times 3)=
\begin{bmatrix}
0 & 1.414214 & 0.707107\\
0 & 1.414214 & 0.707107\\
0 & 2.828428 & 1.414214
\end{bmatrix}
$$
痛點：一次矩陣乘法就得到「每個詞對所有詞」的關聯度，避免逐詞比對太慢。

### 1-4-3 縮放 $S^{(1)}=\dfrac{S_{\mathrm{raw}}^{(1)}}{1.414214}$
$$
S^{(1)}\ (shape=3\times 3)=\frac{1}{1.414214}\,S_{\mathrm{raw}}^{(1)}\ (shape=3\times 3)
$$

$$
S^{(1)}\ (shape=3\times 3)=
\begin{bmatrix}
0 & 1 & 0.5\\
0 & 1 & 0.5\\
0 & 2 & 1
\end{bmatrix}
$$
痛點：避免分數因維度變大而過度極端，讓 softmax 不容易飽和、推論更穩定。

### 1-4-4 權重矩陣 $A^{(1)}=\mathrm{softmax}(S^{(1)})$（逐列 softmax）
$$
A^{(1)}\ (shape=3\times 3)=\mathrm{softmax}\left(S^{(1)}\ (shape=3\times 3)\right)
$$

$$
A^{(1)}\ (shape=3\times 3)=
\begin{bmatrix}
0.186324 & 0.506480 & 0.307196\\
0.186324 & 0.506480 & 0.307196\\
0.090031 & 0.665241 & 0.244728
\end{bmatrix}
$$
痛點：把關聯度轉成「比例分配」，自動壓低不重要的詞以避免噪聲干擾。

### 1-4-5 Head 1 輸出 $O^{(1)}=A^{(1)}V^{(1)}$
$$
A^{(1)}\ (shape=3\times 3)\cdot V^{(1)}\ (shape=3\times 2)=O^{(1)}\ (shape=3\times 2)
$$

$$
O^{(1)}\ (shape=3\times 2)=
\begin{bmatrix}
0.493520 & 0.813676\\
0.493520 & 0.813676\\
0.334759 & 0.909969
\end{bmatrix}
$$
痛點：把全句資訊依權重加權匯總回每個位置，形成「已整理過的上下文表示」。

## 1-5 Head 2：投影得到 $Q^{(2)},K^{(2)},V^{(2)}$

### 1-5-1 Head 2 的 $W_Q^{(2)},b_Q^{(2)}$
（本例讓 Query 投影先與 Head 1 相同，差異主要放在 $K^{(2)},V^{(2)}$，用來示範不同 head 的分工）

$$
W_Q^{(2)}\ (shape=4\times 2)=
\begin{bmatrix}
1 & 0\\
0 & 1\\
0 & 0\\
0 & 0
\end{bmatrix}
$$

$$
b_Q^{(2)}\ (shape=1\times 2)=
\begin{bmatrix}
0 & 0
\end{bmatrix}
$$
痛點：即使 Query 類似，不同 head 仍可透過不同 Key/Value 走向不同的資訊子空間。

### 1-5-2 計算 $Q^{(2)}=XW_Q^{(2)}+b_Q^{(2)}$
$$
X\ (shape=3\times 4)\cdot W_Q^{(2)}\ (shape=4\times 2)=Q_{\mathrm{raw}}^{(2)}\ (shape=3\times 2)
$$

$$
Q_{\mathrm{raw}}^{(2)}\ (shape=3\times 2)=
\begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{bmatrix}
$$

$$
B_Q^{(2)}\ (shape=3\times 2)=
\begin{bmatrix}
0 & 0\\
0 & 0\\
0 & 0
\end{bmatrix}
$$

$$
Q_{\mathrm{raw}}^{(2)}\ (shape=3\times 2)+B_Q^{(2)}\ (shape=3\times 2)=Q^{(2)}\ (shape=3\times 2)
$$

$$
Q^{(2)}\ (shape=3\times 2)=
\begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{bmatrix}
$$
痛點：每個位置都具備「發問能力」，後面才能對不同 Key 子空間做查詢。

### 1-5-3 Head 2 的 $W_K^{(2)},b_K^{(2)}$
（設計成讓「補償」更容易對到「外送」的索引，模擬另一種關注面向：政策/主題）

$$
W_K^{(2)}\ (shape=4\times 2)=
\begin{bmatrix}
-0.707107 & -0.707107\\
-1.414214 & -1.414214\\
0 & 0\\
0 & 0
\end{bmatrix}
$$

$$
b_K^{(2)}\ (shape=1\times 2)=
\begin{bmatrix}
2.121321 & 2.121321
\end{bmatrix}
$$
痛點：用不同 Key 投影，讓另一個 head 能「偏好」另一種關聯（例如主題/政策而不是事件原因）。

### 1-5-4 計算 $K^{(2)}=XW_K^{(2)}+b_K^{(2)}$
$$
X\ (shape=3\times 4)\cdot W_K^{(2)}\ (shape=4\times 2)=K_{\mathrm{raw}}^{(2)}\ (shape=3\times 2)
$$

$$
K_{\mathrm{raw}}^{(2)}\ (shape=3\times 2)=
\begin{bmatrix}
-0.707107 & -0.707107\\
-1.414214 & -1.414214\\
-2.121321 & -2.121321
\end{bmatrix}
$$

$$
B_K^{(2)}\ (shape=3\times 2)=
\begin{bmatrix}
2.121321 & 2.121321\\
2.121321 & 2.121321\\
2.121321 & 2.121321
\end{bmatrix}
$$

$$
K_{\mathrm{raw}}^{(2)}\ (shape=3\times 2)+B_K^{(2)}\ (shape=3\times 2)=K^{(2)}\ (shape=3\times 2)
$$

$$
K^{(2)}\ (shape=3\times 2)=
\begin{bmatrix}
1.414214 & 1.414214\\
0.707107 & 0.707107\\
0 & 0
\end{bmatrix}
$$
痛點：在另一個子空間建立不同的索引排序，讓模型能同時學到多種「對齊規則」。

### 1-5-5 Head 2 的 $W_V^{(2)},b_V^{(2)}$
（讓 Value 取不同的輸入維度，模擬「帶走的內容」也不同）

$$
W_V^{(2)}\ (shape=4\times 2)=
\begin{bmatrix}
0 & 0\\
0 & 0\\
1 & 0\\
0 & 1
\end{bmatrix}
$$

$$
b_V^{(2)}\ (shape=1\times 2)=
\begin{bmatrix}
0 & 0
\end{bmatrix}
$$
痛點：不同 head 不只用不同方式比對，也能搬運不同種類的內容特徵。

### 1-5-6 計算 $V^{(2)}=XW_V^{(2)}+b_V^{(2)}$
$$
X\ (shape=3\times 4)\cdot W_V^{(2)}\ (shape=4\times 2)=V_{\mathrm{raw}}^{(2)}\ (shape=3\times 2)
$$

$$
V_{\mathrm{raw}}^{(2)}\ (shape=3\times 2)=
\begin{bmatrix}
0 & 1\\
1 & 0\\
0 & 0
\end{bmatrix}
$$

$$
B_V^{(2)}\ (shape=3\times 2)=
\begin{bmatrix}
0 & 0\\
0 & 0\\
0 & 0
\end{bmatrix}
$$

$$
V_{\mathrm{raw}}^{(2)}\ (shape=3\times 2)+B_V^{(2)}\ (shape=3\times 2)=V^{(2)}\ (shape=3\times 2)
$$

$$
V^{(2)}\ (shape=3\times 2)=
\begin{bmatrix}
0 & 1\\
1 & 0\\
0 & 0
\end{bmatrix}
$$
痛點：同一段輸入能被不同 head 以不同方式摘要，提升資訊聚合的多樣性。

## 1-6 Head 2：Scaled Dot-Product Attention

### 1-6-1 轉置 $K^{(2)T}$
$$
K^{(2)T}\ (shape=2\times 3)=
\begin{bmatrix}
1.414214 & 0.707107 & 0\\
1.414214 & 0.707107 & 0
\end{bmatrix}
$$
痛點：把第二個 head 的 Key 也整理成可批次比對的形狀，維持高效計算。

### 1-6-2 分數矩陣 $S_{\mathrm{raw}}^{(2)}=Q^{(2)}K^{(2)T}$
$$
Q^{(2)}\ (shape=3\times 2)\cdot K^{(2)T}\ (shape=2\times 3)=S_{\mathrm{raw}}^{(2)}\ (shape=3\times 3)
$$

$$
S_{\mathrm{raw}}^{(2)}\ (shape=3\times 3)=
\begin{bmatrix}
1.414214 & 0.707107 & 0\\
1.414214 & 0.707107 & 0\\
2.828428 & 1.414214 & 0
\end{bmatrix}
$$
痛點：在另一個 head 得到另一套「關聯度地圖」，補足單一地圖的盲區。

### 1-6-3 縮放 $S^{(2)}=\dfrac{S_{\mathrm{raw}}^{(2)}}{1.414214}$
$$
S^{(2)}\ (shape=3\times 3)=\frac{1}{1.414214}\,S_{\mathrm{raw}}^{(2)}\ (shape=3\times 3)
$$

$$
S^{(2)}\ (shape=3\times 3)=
\begin{bmatrix}
1 & 0.5 & 0\\
1 & 0.5 & 0\\
2 & 1 & 0
\end{bmatrix}
$$
痛點：維持數值尺度一致，避免某個 head 因分數尺度不同而主導所有學習。

### 1-6-4 權重矩陣 $A^{(2)}=\mathrm{softmax}(S^{(2)})$（逐列 softmax）
$$
A^{(2)}\ (shape=3\times 3)=\mathrm{softmax}\left(S^{(2)}\ (shape=3\times 3)\right)
$$

$$
A^{(2)}\ (shape=3\times 3)=
\begin{bmatrix}
0.506480 & 0.307196 & 0.186324\\
0.506480 & 0.307196 & 0.186324\\
0.665241 & 0.244728 & 0.090031
\end{bmatrix}
$$
痛點：把第二個 head 的注意力也變成穩定的比例權重，方便做加權聚合。

### 1-6-5 Head 2 輸出 $O^{(2)}=A^{(2)}V^{(2)}$
$$
A^{(2)}\ (shape=3\times 3)\cdot V^{(2)}\ (shape=3\times 2)=O^{(2)}\ (shape=3\times 2)
$$

$$
O^{(2)}\ (shape=3\times 2)=
\begin{bmatrix}
0.307196 & 0.506480\\
0.307196 & 0.506480\\
0.244728 & 0.665241
\end{bmatrix}
$$
痛點：把第二種角度的重點摘要回每個位置，形成互補的上下文資訊。

## 1-7 合併兩個 Head 並輸出

### 1-7-1 串接（Concat）兩個 head 的輸出
$$
O_{\mathrm{concat}}\ (shape=3\times 4)=
\mathrm{Concat}\left(O^{(1)}\ (shape=3\times 2),\ O^{(2)}\ (shape=3\times 2)\right)
$$

$$
O_{\mathrm{concat}}\ (shape=3\times 4)=
\begin{bmatrix}
0.493520 & 0.813676 & 0.307196 & 0.506480\\
0.493520 & 0.813676 & 0.307196 & 0.506480\\
0.334759 & 0.909969 & 0.244728 & 0.665241
\end{bmatrix}
$$
痛點：把不同 head 的互補資訊保留下來，不要在單一 head 裡互相擠掉。

### 1-7-2 輸出投影參數 $W_O,b_O$
（教學用：先用最簡單的 $W_O=I$ 來看清楚多頭本身的效果）

$$
W_O\ (shape=4\times 4)=
\begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{bmatrix}
$$

$$
b_O\ (shape=1\times 4)=
\begin{bmatrix}
0 & 0 & 0 & 0
\end{bmatrix}
$$
痛點：把 concat 後的資訊重新投影回模型需要的表徵空間，方便接到後續層。

### 1-7-3 最終輸出 $Y=O_{\mathrm{concat}}W_O+b_O$
$$
O_{\mathrm{concat}}\ (shape=3\times 4)\cdot W_O\ (shape=4\times 4)=Y_{\mathrm{raw}}\ (shape=3\times 4)
$$

$$
Y_{\mathrm{raw}}\ (shape=3\times 4)=
\begin{bmatrix}
0.493520 & 0.813676 & 0.307196 & 0.506480\\
0.493520 & 0.813676 & 0.307196 & 0.506480\\
0.334759 & 0.909969 & 0.244728 & 0.665241
\end{bmatrix}
$$

$$
B_O\ (shape=3\times 4)=
\begin{bmatrix}
0 & 0 & 0 & 0\\
0 & 0 & 0 & 0\\
0 & 0 & 0 & 0
\end{bmatrix}
$$

$$
Y_{\mathrm{raw}}\ (shape=3\times 4)+B_O\ (shape=3\times 4)=Y\ (shape=3\times 4)
$$

$$
Y\ (shape=3\times 4)=
\begin{bmatrix}
0.493520 & 0.813676 & 0.307196 & 0.506480\\
0.493520 & 0.813676 & 0.307196 & 0.506480\\
0.334759 & 0.909969 & 0.244728 & 0.665241
\end{bmatrix}
$$
痛點：把多頭結果變成後續網路可直接消化的統一輸出，利於串接與部署。

## 1-8 回到真實場景：兩個 head 如何分工

### 1-8-1 Head 1：當你處理「補償」時更關注「晚到」
取 Head 1 的注意力矩陣第 3 列（token3=補償 對全部 tokens 的權重）：

$$
A^{(1)}_{\text{補償}}\ (shape=1\times 3)=
\begin{bmatrix}
0.090031 & 0.665241 & 0.244728
\end{bmatrix}
$$

它對 token2=晚到 的權重最大（0.665241），表示 Head 1 更像在做「事件原因對齊」，讓模型生成回覆時自然把「晚到」當作補償判斷依據。  
痛點：把「原因」與「處理動作」對齊，降低客服回覆漏講原因或誤判補償條件。

### 1-8-2 Head 2：同樣處理「補償」時更關注「外送」這個主題/政策上下文
取 Head 2 的注意力矩陣第 3 列：

$$
A^{(2)}_{\text{補償}}\ (shape=1\times 3)=
\begin{bmatrix}
0.665241 & 0.244728 & 0.090031
\end{bmatrix}
$$

它對 token1=外送 的權重最大（0.665241），表示 Head 2 更像在做「主題/政策對齊」，把補償連到外送規則（而不是例如餐點品質或其他情境）。  
痛點：把「主題」與「規則」對齊，避免只看事件而忽略適用政策，導致答覆不一致。
