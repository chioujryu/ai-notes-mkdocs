# KV Cache

## 1、咖啡店的訂單記憶術

想像你是一家生意興隆的咖啡店的老闆。你的咖啡店有個特別的服務：顧客可以一次點很多杯咖啡，而且可以不斷修改訂單。

有一天，來了一位叫小明的顧客。他點了第一杯咖啡：「一杯熱美式。」

你轉身對身後的咖啡師大喊：「一杯熱美式！」

咖啡師開始製作。

過了一分鐘，小明說：「我還要一杯冰拿鐵。」

這時候，你必須讓咖啡師知道，總共有兩杯咖啡要做。於是你又對咖啡師大喊：「前面有一杯熱美式，再加一杯冰拿鐵！」

咖啡師心裡想：「我知道有熱美式了，你不用每次都重複前面的。」

又過了一分鐘，小明說：「再一杯卡布奇諾。」

你又得扯開嗓子：「前面有一杯熱美式、一杯冰拿鐵，再加一杯卡布奇諾！」

就這樣，每次小明加點，你都要把前面所有的訂單重新複誦一遍。咖啡師不僅要記住新的飲料，還要從頭聽你把所有訂單念完，才能確認總共有哪些工作要做。

到後來，小明總共點了 100 杯咖啡，而你每次加點時，都要從第一杯開始念：「一杯熱美式、一杯冰拿鐵、一杯卡布奇諾……」念完前面 99 杯，才能說出第 100 杯是什麼。

咖啡師受不了了：「老闆，你可不可以只告訴我『加點』的那一杯是什麼？前面的我都記住了啊！」

你恍然大悟：「對耶！我為什麼每次都要重複所有的東西？」

## 2、AI 的生成困境：重複計算的浪費

這個咖啡店的故事，完美對應了大型語言模型在生成文字時遇到的效率問題。

當模型在生成每一個新的詞彙（Token）時，它就像那個疲憊的咖啡師，需要理解整個對話的上下文。但在沒有 KV Cache 的技術之前，模型的運作方式就像那個笨拙的老闆：

每次要生成一個新字，模型都要把「從頭到尾所有的對話歷史」和「已經生成的所有文字」重新看一遍、算一遍。

假設我們要模型生成「我喜歡吃蘋果」這句話：
1. 生成「我」：模型要計算「開頭符號」。
2. 生成「喜歡」：模型要重新計算「開頭符號」和「我」。
3. 生成「吃」：模型要重新計算「開頭符號」、「我」、「喜歡」。
4. 生成「蘋果」：模型要重新計算「開頭符號」、「我」、「喜歡」、「吃」。

可以發現，隨著生成的文字越來越多，每次需要計算的量也越來越大。這就像老闆每次都要把訂單從頭念一遍，非常沒有效率。

## 3、解決的痛點：KV Cache 的記憶魔法

KV Cache 的誕生，正是為了解決這個「重複計算」的巨大痛點，讓老闆學會只說「加點的那一杯」。

在 Transformer 模型的注意力機制中，每一個詞彙都會被轉換成兩組重要的向量：鍵（Key）和值（Value）。你可以把它們想像成咖啡訂單上的「飲料名稱」和「製作細節」。

- **K (Key)**：飲料的名稱，用來被搜尋。
- **V (Value)**：製作這杯飲料的詳細步驟。

當模型在生成一個新詞時，它會用這個新詞的查詢（Query，可以想像成「現在要做的飲料」），去跟前面所有詞的 Key 進行比對（找出相關的歷史訂單），然後用比對的結果去加權前面的 Value（綜合過去的製作經驗），來決定如何生成這個新詞。

### 3-1、減少重複計算，加速生成

有了 KV Cache 之後，過程就變了：
1. 生成第一個字「我」時，模型計算並產生「我」的 Key 和 Value。
2. 模型把「我」的 K 和 V 暫時存放在快取（Cache）裡，就像咖啡師把第一杯訂單記在心裡。
3. 要生成第二個字「喜歡」時，模型只需要：
   - 從快取中讀取第一個字「我」的 K 和 V。
   - 計算當前「喜歡」這個字的 Q，並與快取中的 K 進行注意力計算。
   - 生成「喜歡」後，再把「喜歡」的 K 和 V 也加入快取。
4. 要生成第三個字「吃」時，模型直接從快取中讀取「我」和「喜歡」的 K、V，然後只計算「吃」的部分。

如此一來，每次生成新詞時，需要計算的量就恆定不變，而不是隨著文字長度而線性增長。這就像咖啡師只需要處理「最新加點的那一杯」，大幅提升了效率，讓生成速度變得飛快。

### 3-2、節省運算資源，降低成本

重複計算的減少，直接帶來了運算資源的節省。對於長篇文章的生成，KV Cache 的效益尤其顯著。

假設生成一個 1000 個字的文章，沒有 KV Cache 可能需要進行 1000 + 999 + 998 + ... 次的計算，時間複雜度約為 $O(n^2)$。而有 KV Cache 後，計算量只需要約 $O(n)$。

這個效率的提升，讓使用者能更快得到回應，也讓營運模型的成本大幅降低。它使得即時的長篇對話、複雜的文件生成變得可行，是現代大型語言模型能夠如此實用和普及的關鍵技術之一。

就像那位終於學會只報加點項目的老闆，KV Cache 讓 AI 模型從一個吃力不討好的重複勞動者，變成一個真正高效、懂得「記住」的聰明夥伴。

---

## 4、從零開始：一個簡單的生成任務

我是你的 AI 數學老師，今天我們要親手算一次 KV Cache 的運算過程。讓我們用一個超級簡單的例子，來看看 KV Cache 到底省掉了哪些計算。

假設我們有一個極小的模型，正在生成一句話：「我 愛 你」。目前已經有兩個詞：「我」和「愛」，現在要生成第三個詞「你」。

模型的設定如下：
- 詞嵌入維度 $d_{model} = 4$（每個詞用 4 個數字表示）
- 注意力頭數 $h = 2$（每個頭處理部分資訊）
- 每個頭的維度 $d_k = d_{model} / h = 2$

輸入序列是兩個詞：
- 詞1「我」：$x_1$
- 詞2「愛」：$x_2$

假設經過詞嵌入後，這兩個詞的向量表示為：

$$
X\ (shape=2\times 4)=
\begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1
\end{bmatrix}
$$

第一行是「我」，第二行是「愛」。我們要生成第三個詞「你」。

## 5、沒有 KV Cache：重複計算的痛苦

首先，我們來看沒有 KV Cache 時會發生什麼事。

### 5-1、第一步：準備查詢、鍵、值矩陣

對於每個注意力頭，我們都需要將輸入 $X$ 乘上對應的權重矩陣，得到查詢 $Q$、鍵 $K$、值 $V$。

假設頭 1 的權重矩陣為：

$$
W^Q_1\ (shape=4\times 2)=
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 0 \\
0 & 1
\end{bmatrix}
$$

$$
W^K_1\ (shape=4\times 2)=
\begin{bmatrix}
1 & 0 \\
1 & 0 \\
0 & 1 \\
0 & 1
\end{bmatrix}
$$

$$
W^V_1\ (shape=4\times 2)=
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 0 \\
0 & 1
\end{bmatrix}
$$

頭 2 的權重矩陣為：

$$
W^Q_2\ (shape=4\times 2)=
\begin{bmatrix}
0 & 1 \\
1 & 0 \\
0 & 1 \\
1 & 0
\end{bmatrix}
$$

$$
W^K_2\ (shape=4\times 2)=
\begin{bmatrix}
0 & 1 \\
0 & 1 \\
1 & 0 \\
1 & 0
\end{bmatrix}
$$

$$
W^V_2\ (shape=4\times 2)=
\begin{bmatrix}
0 & 1 \\
1 & 0 \\
0 & 1 \\
1 & 0
\end{bmatrix}
$$

### 5-2、第二步：計算頭 1 的 Q、K、V

計算頭 1 的查詢矩陣：

$$
Q_1 = X \cdot W^Q_1
$$

$$
X\ (2\times 4) \cdot W^Q_1\ (4\times 2) = Q_1\ (2\times 2)
$$

$$
Q_1=
\begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1
\end{bmatrix}
\cdot
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 0 \\
0 & 1
\end{bmatrix}
=
\begin{bmatrix}
(1\cdot1 + 0\cdot0 + 1\cdot1 + 0\cdot0) & (1\cdot0 + 0\cdot1 + 1\cdot0 + 0\cdot1) \\
(0\cdot1 + 1\cdot0 + 0\cdot1 + 1\cdot0) & (0\cdot0 + 1\cdot1 + 0\cdot0 + 1\cdot1)
\end{bmatrix}
$$

$$
Q_1\ (shape=2\times 2)=
\begin{bmatrix}
2 & 0 \\
0 & 2
\end{bmatrix}
$$

計算頭 1 的鍵矩陣：

$$
K_1 = X \cdot W^K_1
$$

$$
X\ (2\times 4) \cdot W^K_1\ (4\times 2) = K_1\ (2\times 2)
$$

$$
K_1=
\begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1
\end{bmatrix}
\cdot
\begin{bmatrix}
1 & 0 \\
1 & 0 \\
0 & 1 \\
0 & 1
\end{bmatrix}
=
\begin{bmatrix}
(1\cdot1 + 0\cdot1 + 1\cdot0 + 0\cdot0) & (1\cdot0 + 0\cdot0 + 1\cdot1 + 0\cdot1) \\
(0\cdot1 + 1\cdot1 + 0\cdot0 + 1\cdot0) & (0\cdot0 + 1\cdot0 + 0\cdot1 + 1\cdot1)
\end{bmatrix}
$$

$$
K_1\ (shape=2\times 2)=
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
$$

計算頭 1 的值矩陣：

$$
V_1 = X \cdot W^V_1
$$

$$
X\ (2\times 4) \cdot W^V_1\ (4\times 2) = V_1\ (2\times 2)
$$

$$
V_1=
\begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1
\end{bmatrix}
\cdot
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 0 \\
0 & 1
\end{bmatrix}
=
\begin{bmatrix}
2 & 0 \\
0 & 2
\end{bmatrix}
$$

### 5-3、第三步：計算頭 2 的 Q、K、V

計算頭 2 的查詢矩陣：

$$
Q_2 = X \cdot W^Q_2
$$

$$
X\ (2\times 4) \cdot W^Q_2\ (4\times 2) = Q_2\ (2\times 2)
$$

$$
Q_2=
\begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1
\end{bmatrix}
\cdot
\begin{bmatrix}
0 & 1 \\
1 & 0 \\
0 & 1 \\
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
(1\cdot0 + 0\cdot1 + 1\cdot0 + 0\cdot1) & (1\cdot1 + 0\cdot0 + 1\cdot1 + 0\cdot0) \\
(0\cdot0 + 1\cdot1 + 0\cdot0 + 1\cdot1) & (0\cdot1 + 1\cdot0 + 0\cdot1 + 1\cdot0)
\end{bmatrix}
$$

$$
Q_2\ (shape=2\times 2)=
\begin{bmatrix}
0 & 2 \\
2 & 0
\end{bmatrix}
$$

計算頭 2 的鍵矩陣：

$$
K_2 = X \cdot W^K_2
$$

$$
X\ (2\times 4) \cdot W^K_2\ (4\times 2) = K_2\ (2\times 2)
$$

$$
K_2=
\begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1
\end{bmatrix}
\cdot
\begin{bmatrix}
0 & 1 \\
0 & 1 \\
1 & 0 \\
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
(1\cdot0 + 0\cdot0 + 1\cdot1 + 0\cdot1) & (1\cdot1 + 0\cdot1 + 1\cdot0 + 0\cdot0) \\
(0\cdot0 + 1\cdot0 + 0\cdot1 + 1\cdot1) & (0\cdot1 + 1\cdot1 + 0\cdot0 + 1\cdot0)
\end{bmatrix}
$$

$$
K_2\ (shape=2\times 2)=
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
$$

計算頭 2 的值矩陣：

$$
V_2 = X \cdot W^V_2
$$

$$
X\ (2\times 4) \cdot W^V_2\ (4\times 2) = V_2\ (2\times 2)
$$

$$
V_2=
\begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1
\end{bmatrix}
\cdot
\begin{bmatrix}
0 & 1 \\
1 & 0 \\
0 & 1 \\
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
(1\cdot0 + 0\cdot1 + 1\cdot0 + 0\cdot1) & (1\cdot1 + 0\cdot0 + 1\cdot1 + 0\cdot0) \\
(0\cdot0 + 1\cdot1 + 0\cdot0 + 1\cdot1) & (0\cdot1 + 1\cdot0 + 0\cdot1 + 1\cdot0)
\end{bmatrix}
$$

$$
V_2\ (shape=2\times 2)=
\begin{bmatrix}
0 & 2 \\
2 & 0
\end{bmatrix}
$$

### 5-4、第四步：計算注意力分數（頭 1）

現在要為頭 1 計算注意力分數。用 $Q_1$ 乘上 $K_1$ 的轉置：

$$
Q_1 \cdot K_1^T
$$

首先計算 $K_1^T$：

$$
K_1^T\ (shape=2\times 2)=
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
$$

然後計算乘積：

$$
\text{Score}_1 = Q_1 \cdot K_1^T
$$

$$
Q_1\ (2\times 2) \cdot K_1^T\ (2\times 2) = \text{Score}_1\ (2\times 2)
$$

$$
\text{Score}_1=
\begin{bmatrix}
2 & 0 \\
0 & 2
\end{bmatrix}
\cdot
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
=
\begin{bmatrix}
(2\cdot1 + 0\cdot1) & (2\cdot1 + 0\cdot1) \\
(0\cdot1 + 2\cdot1) & (0\cdot1 + 2\cdot1)
\end{bmatrix}
$$

$$
\text{Score}_1\ (shape=2\times 2)=
\begin{bmatrix}
2 & 2 \\
2 & 2
\end{bmatrix}
$$

### 5-5、第五步：計算注意力分數（頭 2）

為頭 2 計算注意力分數。用 $Q_2$ 乘上 $K_2$ 的轉置：

首先計算 $K_2^T$：

$$
K_2^T\ (shape=2\times 2)=
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
$$

然後計算乘積：

$$
\text{Score}_2 = Q_2 \cdot K_2^T
$$

$$
Q_2\ (2\times 2) \cdot K_2^T\ (2\times 2) = \text{Score}_2\ (2\times 2)
$$

$$
\text{Score}_2=
\begin{bmatrix}
0 & 2 \\
2 & 0
\end{bmatrix}
\cdot
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
=
\begin{bmatrix}
(0\cdot1 + 2\cdot1) & (0\cdot1 + 2\cdot1) \\
(2\cdot1 + 0\cdot1) & (2\cdot1 + 0\cdot1)
\end{bmatrix}
$$

$$
\text{Score}_2\ (shape=2\times 2)=
\begin{bmatrix}
2 & 2 \\
2 & 2
\end{bmatrix}
$$

### 5-6、第六步：softmax 和加權和（頭 1）

對注意力分數做 softmax。為簡化計算，假設 softmax 後數值不變（實際上會變，但這裡為了簡化）：

$$
\text{Attention}_1 = \text{softmax}(\text{Score}_1) \cdot V_1
$$

假設 softmax 後矩陣不變：

$$
\text{softmax}(\text{Score}_1) \approx
\begin{bmatrix}
2 & 2 \\
2 & 2
\end{bmatrix}
$$

乘以 $V_1$：

$$
\text{Attention}_1=
\begin{bmatrix}
2 & 2 \\
2 & 2
\end{bmatrix}
\cdot
\begin{bmatrix}
2 & 0 \\
0 & 2
\end{bmatrix}
=
\begin{bmatrix}
(2\cdot2 + 2\cdot0) & (2\cdot0 + 2\cdot2) \\
(2\cdot2 + 2\cdot0) & (2\cdot0 + 2\cdot2)
\end{bmatrix}
$$

$$
\text{Attention}_1\ (shape=2\times 2)=
\begin{bmatrix}
4 & 4 \\
4 & 4
\end{bmatrix}
$$

### 5-7、第七步：softmax 和加權和（頭 2）

同樣的步驟，假設 softmax 後矩陣不變：

$$
\text{softmax}(\text{Score}_2) \approx
\begin{bmatrix}
2 & 2 \\
2 & 2
\end{bmatrix}
$$

乘以 $V_2$：

$$
\text{Attention}_2=
\begin{bmatrix}
2 & 2 \\
2 & 2
\end{bmatrix}
\cdot
\begin{bmatrix}
0 & 2 \\
2 & 0
\end{bmatrix}
=
\begin{bmatrix}
(2\cdot0 + 2\cdot2) & (2\cdot2 + 2\cdot0) \\
(2\cdot0 + 2\cdot2) & (2\cdot2 + 2\cdot0)
\end{bmatrix}
$$

$$
\text{Attention}_2\ (shape=2\times 2)=
\begin{bmatrix}
4 & 4 \\
4 & 4
\end{bmatrix}
$$

### 5-8、第八步：合併多頭輸出

將兩個頭的輸出拼接起來：

$$
\text{Attention} = [\text{Attention}_1 | \text{Attention}_2]
$$

$$
\text{Attention}\ (shape=2\times 4)=
\begin{bmatrix}
4 & 4 & 4 & 4 \\
4 & 4 & 4 & 4
\end{bmatrix}
$$

這個矩陣就是我們對「我」和「愛」這兩個詞做完注意力計算的結果。接下來我們會用這個結果來生成第三個詞「你」。

### 5-9、第九步：生成第三個詞

現在我們要生成第三個詞，模型會用這個 Attention 結果做下一步預測。但關鍵問題是：**我們已經把前面兩個詞的計算全部做完了嗎？**

沒有！我們剛剛做的計算，只是為了生成第三個詞「你」而做的。但注意看，我們從頭到尾，把「我」和「愛」這兩個詞的 Q、K、V 全部重新算了一遍。

這就像前面咖啡店的故事：每次要加點一杯新的咖啡，老闆都要把前面所有訂單重新念一遍。這裡的 Q、K、V 計算，就是那個「重新念一遍」的過程。

痛點：每次生成新詞都需要重新計算所有歷史詞的查詢、鍵、值矩陣，導致計算量隨序列長度呈平方級增長，生成速度越來越慢。

## 6、有 KV Cache：只計算新增的部分

現在我們來看看，如果有 KV Cache 會怎麼做。

### 6-1、第一步：快取中已有歷史詞的 K 和 V

假設我們在第一輪生成「我」和「愛」的時候，就已經把它們的 K 和 V 存起來了。

從前面的計算，我們已經有：

頭 1 的 K 和 V 快取：

$$
K_{1\_cache}\ (shape=2\times 2)=
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
$$

$$
V_{1\_cache}\ (shape=2\times 2)=
\begin{bmatrix}
2 & 0 \\
0 & 2
\end{bmatrix}
$$

頭 2 的 K 和 V 快取：

$$
K_{2\_cache}\ (shape=2\times 2)=
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
$$

$$
V_{2\_cache}\ (shape=2\times 2)=
\begin{bmatrix}
0 & 2 \\
2 & 0
\end{bmatrix}
$$

現在我們要生成第三個詞「你」。首先，我們需要第三個詞的輸入向量。假設「你」的詞嵌入向量為：

$$
x_3\ (shape=1\times 4)=
\begin{bmatrix}
1 & 1 & 0 & 0
\end{bmatrix}
$$

### 6-2、第二步：只計算第三個詞的 Q、K、V

有了 KV Cache，我們不需要重新計算前面兩個詞的 K 和 V，只需要計算第三個詞的 Q、K、V。

先計算頭 1 的部分：

第三個詞的查詢：

$$
q_{3,1} = x_3 \cdot W^Q_1
$$

$$
x_3\ (1\times 4) \cdot W^Q_1\ (4\times 2) = q_{3,1}\ (1\times 2)
$$

$$
q_{3,1}=
\begin{bmatrix}
1 & 1 & 0 & 0
\end{bmatrix}
\cdot
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 0 \\
0 & 1
\end{bmatrix}
=
\begin{bmatrix}
(1\cdot1 + 1\cdot0 + 0\cdot1 + 0\cdot0) & (1\cdot0 + 1\cdot1 + 0\cdot0 + 0\cdot1)
\end{bmatrix}
$$

$$
q_{3,1}\ (shape=1\times 2)=
\begin{bmatrix}
1 & 1
\end{bmatrix}
$$

第三個詞的鍵：

$$
k_{3,1} = x_3 \cdot W^K_1
$$

$$
x_3\ (1\times 4) \cdot W^K_1\ (4\times 2) = k_{3,1}\ (1\times 2)
$$

$$
k_{3,1}=
\begin{bmatrix}
1 & 1 & 0 & 0
\end{bmatrix}
\cdot
\begin{bmatrix}
1 & 0 \\
1 & 0 \\
0 & 1 \\
0 & 1
\end{bmatrix}
=
\begin{bmatrix}
(1\cdot1 + 1\cdot1 + 0\cdot0 + 0\cdot0) & (1\cdot0 + 1\cdot0 + 0\cdot1 + 0\cdot1)
\end{bmatrix}
$$

$$
k_{3,1}\ (shape=1\times 2)=
\begin{bmatrix}
2 & 0
\end{bmatrix}
$$

第三個詞的值：

$$
v_{3,1} = x_3 \cdot W^V_1
$$

$$
x_3\ (1\times 4) \cdot W^V_1\ (4\times 2) = v_{3,1}\ (1\times 2)
$$

$$
v_{3,1}=
\begin{bmatrix}
1 & 1 & 0 & 0
\end{bmatrix}
\cdot
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 0 \\
0 & 1
\end{bmatrix}
=
\begin{bmatrix}
(1\cdot1 + 1\cdot0 + 0\cdot1 + 0\cdot0) & (1\cdot0 + 1\cdot1 + 0\cdot0 + 0\cdot1)
\end{bmatrix}
$$

$$
v_{3,1}\ (shape=1\times 2)=
\begin{bmatrix}
1 & 1
\end{bmatrix}
$$

### 6-3、第三步：計算頭 2 的部分

計算頭 2 的查詢：

$$
q_{3,2} = x_3 \cdot W^Q_2
$$

$$
x_3\ (1\times 4) \cdot W^Q_2\ (4\times 2) = q_{3,2}\ (1\times 2)
$$

$$
q_{3,2}=
\begin{bmatrix}
1 & 1 & 0 & 0
\end{bmatrix}
\cdot
\begin{bmatrix}
0 & 1 \\
1 & 0 \\
0 & 1 \\
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
(1\cdot0 + 1\cdot1 + 0\cdot0 + 0\cdot1) & (1\cdot1 + 1\cdot0 + 0\cdot1 + 0\cdot0)
\end{bmatrix}
$$

$$
q_{3,2}\ (shape=1\times 2)=
\begin{bmatrix}
1 & 1
\end{bmatrix}
$$

頭 2 的鍵：

$$
k_{3,2} = x_3 \cdot W^K_2
$$

$$
x_3\ (1\times 4) \cdot W^K_2\ (4\times 2) = k_{3,2}\ (1\times 2)
$$

$$
k_{3,2}=
\begin{bmatrix}
1 & 1 & 0 & 0
\end{bmatrix}
\cdot
\begin{bmatrix}
0 & 1 \\
0 & 1 \\
1 & 0 \\
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
(1\cdot0 + 1\cdot0 + 0\cdot1 + 0\cdot1) & (1\cdot1 + 1\cdot1 + 0\cdot0 + 0\cdot0)
\end{bmatrix}
$$

$$
k_{3,2}\ (shape=1\times 2)=
\begin{bmatrix}
0 & 2
\end{bmatrix}
$$

頭 2 的值：

$$
v_{3,2} = x_3 \cdot W^V_2
$$

$$
x_3\ (1\times 4) \cdot W^V_2\ (4\times 2) = v_{3,2}\ (1\times 2)
$$

$$
v_{3,2}=
\begin{bmatrix}
1 & 1 & 0 & 0
\end{bmatrix}
\cdot
\begin{bmatrix}
0 & 1 \\
1 & 0 \\
0 & 1 \\
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
(1\cdot0 + 1\cdot1 + 0\cdot0 + 0\cdot1) & (1\cdot1 + 1\cdot0 + 0\cdot1 + 0\cdot0)
\end{bmatrix}
$$

$$
v_{3,2}\ (shape=1\times 2)=
\begin{bmatrix}
1 & 1
\end{bmatrix}
$$

### 6-4、第四步：組合完整的 K 和 V 矩陣

現在我們有：
- 從快取來的：所有歷史詞的 K 和 V（2 個詞）
- 新計算的：當前詞的 K 和 V（1 個詞）

把它們組合起來：

頭 1 的完整 K 矩陣：

$$
K_{1\_total} = [K_{1\_cache}; k_{3,1}]
$$

$$
K_{1\_cache}\ (2\times 2)=
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
$$

$$
k_{3,1}\ (1\times 2)=
\begin{bmatrix}
2 & 0
\end{bmatrix}
$$

組合後：

$$
K_{1\_total}\ (shape=3\times 2)=
\begin{bmatrix}
1 & 1 \\
1 & 1 \\
2 & 0
\end{bmatrix}
$$

頭 1 的完整 V 矩陣：

$$
V_{1\_total} = [V_{1\_cache}; v_{3,1}]
$$

$$
V_{1\_cache}\ (2\times 2)=
\begin{bmatrix}
2 & 0 \\
0 & 2
\end{bmatrix}
$$

$$
v_{3,1}\ (1\times 2)=
\begin{bmatrix}
1 & 1
\end{bmatrix}
$$

組合後：

$$
V_{1\_total}\ (shape=3\times 2)=
\begin{bmatrix}
2 & 0 \\
0 & 2 \\
1 & 1
\end{bmatrix}
$$

頭 2 的完整 K 矩陣：

$$
K_{2\_total} = [K_{2\_cache}; k_{3,2}]
$$

$$
K_{2\_cache}\ (2\times 2)=
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
$$

$$
k_{3,2}\ (1\times 2)=
\begin{bmatrix}
0 & 2
\end{bmatrix}
$$

組合後：

$$
K_{2\_total}\ (shape=3\times 2)=
\begin{bmatrix}
1 & 1 \\
1 & 1 \\
0 & 2
\end{bmatrix}
$$

頭 2 的完整 V 矩陣：

$$
V_{2\_total} = [V_{2\_cache}; v_{3,2}]
$$

$$
V_{2\_cache}\ (2\times 2)=
\begin{bmatrix}
0 & 2 \\
2 & 0
\end{bmatrix}
$$

$$
v_{3,2}\ (1\times 2)=
\begin{bmatrix}
1 & 1
\end{bmatrix}
$$

組合後：

$$
V_{2\_total}\ (shape=3\times 2)=
\begin{bmatrix}
0 & 2 \\
2 & 0 \\
1 & 1
\end{bmatrix}
$$

### 6-5、第五步：計算注意力分數（只針對新詞）

注意！這裡是關鍵。我們只需要計算新詞「你」和其他詞的注意力分數，不需要重新計算歷史詞之間的注意力分數。

對於頭 1，我們用 $q_{3,1}$ 和完整的 $K_{1\_total}$ 計算注意力分數：

$$
\text{score}_{3,1} = q_{3,1} \cdot K_{1\_total}^T
$$

先計算 $K_{1\_total}^T$：

$$
K_{1\_total}^T\ (shape=2\times 3)=
\begin{bmatrix}
1 & 1 & 2 \\
1 & 1 & 0
\end{bmatrix}
$$

然後計算：

$$
q_{3,1}\ (1\times 2) \cdot K_{1\_total}^T\ (2\times 3) = \text{score}_{3,1}\ (1\times 3)
$$

$$
\text{score}_{3,1}=
\begin{bmatrix}
1 & 1
\end{bmatrix}
\cdot
\begin{bmatrix}
1 & 1 & 2 \\
1 & 1 & 0
\end{bmatrix}
$$

$$
= \begin{bmatrix}
(1\cdot1 + 1\cdot1) & (1\cdot1 + 1\cdot1) & (1\cdot2 + 1\cdot0)
\end{bmatrix}
$$

$$
\text{score}_{3,1}\ (shape=1\times 3)=
\begin{bmatrix}
2 & 2 & 2
\end{bmatrix}
$$

對於頭 2，同樣計算：

$$
\text{score}_{3,2} = q_{3,2} \cdot K_{2\_total}^T
$$

先計算 $K_{2\_total}^T$：

$$
K_{2\_total}^T\ (shape=2\times 3)=
\begin{bmatrix}
1 & 1 & 0 \\
1 & 1 & 2
\end{bmatrix}
$$

然後計算：

$$
q_{3,2}\ (1\times 2) \cdot K_{2\_total}^T\ (2\times 3) = \text{score}_{3,2}\ (1\times 3)
$$

$$
\text{score}_{3,2}=
\begin{bmatrix}
1 & 1
\end{bmatrix}
\cdot
\begin{bmatrix}
1 & 1 & 0 \\
1 & 1 & 2
\end{bmatrix}
$$

$$
= \begin{bmatrix}
(1\cdot1 + 1\cdot1) & (1\cdot1 + 1\cdot1) & (1\cdot0 + 1\cdot2)
\end{bmatrix}
$$

$$
\text{score}_{3,2}\ (shape=1\times 3)=
\begin{bmatrix}
2 & 2 & 2
\end{bmatrix}
$$

### 6-6、第六步：softmax 和加權和

假設 softmax 後數值不變，對於頭 1：

$$
\text{attention}_{3,1} = \text{score}_{3,1} \cdot V_{1\_total}
$$

$$
\text{score}_{3,1}\ (1\times 3) \cdot V_{1\_total}\ (3\times 2) = \text{attention}_{3,1}\ (1\times 2)
$$

$$
\text{attention}_{3,1}=
\begin{bmatrix}
2 & 2 & 2
\end{bmatrix}
\cdot
\begin{bmatrix}
2 & 0 \\
0 & 2 \\
1 & 1
\end{bmatrix}
$$

$$
= \begin{bmatrix}
(2\cdot2 + 2\cdot0 + 2\cdot1) & (2\cdot0 + 2\cdot2 + 2\cdot1)
\end{bmatrix}
$$

$$
\text{attention}_{3,1}\ (shape=1\times 2)=
\begin{bmatrix}
6 & 6
\end{bmatrix}
$$

對於頭 2：

$$
\text{attention}_{3,2} = \text{score}_{3,2} \cdot V_{2\_total}
$$

$$
\text{score}_{3,2}\ (1\times 3) \cdot V_{2\_total}\ (3\times 2) = \text{attention}_{3,2}\ (1\times 2)
$$

$$
\text{attention}_{3,2}=
\begin{bmatrix}
2 & 2 & 2
\end{bmatrix}
\cdot
\begin{bmatrix}
0 & 2 \\
2 & 0 \\
1 & 1
\end{bmatrix}
$$

$$
= \begin{bmatrix}
(2\cdot0 + 2\cdot2 + 2\cdot1) & (2\cdot2 + 2\cdot0 + 2\cdot1)
\end{bmatrix}
$$

$$
\text{attention}_{3,2}\ (shape=1\times 2)=
\begin{bmatrix}
6 & 6
\end{bmatrix}
$$

### 6-7、第七步：合併輸出

將兩個頭的輸出拼接：

$$
\text{attention}_3 = [\text{attention}_{3,1} | \text{attention}_{3,2}]
$$

$$
\text{attention}_3\ (shape=1\times 4)=
\begin{bmatrix}
6 & 6 & 6 & 6
\end{bmatrix}
$$

這就是第三個詞「你」的注意力輸出結果。

痛點：KV Cache 透過儲存歷史詞的鍵和值矩陣，使每次生成新詞時只需計算新詞的查詢、鍵、值，以及新詞與所有歷史詞的注意力，將計算複雜度從 $O(n^2)$ 降為 $O(n)$，大幅提升生成速度。

## 7、計算量對比：有 KV Cache 省了多少？

讓我們用這個簡單的例子，來比較兩種方式的計算量。

### 7-1、沒有 KV Cache 的計算量

生成第三個詞時：
- 計算 Q1、K1、V1：3 次矩陣乘法（$2\times4$ 乘 $4\times2$）
- 計算 Q2、K2、V2：3 次矩陣乘法
- 計算注意力分數（兩個頭）：2 次矩陣乘法（$2\times2$ 乘 $2\times2$）
- 計算加權和（兩個頭）：2 次矩陣乘法（$2\times2$ 乘 $2\times2$）

總計：約 10 次矩陣乘法

### 7-2、有 KV Cache 的計算量

生成第三個詞時：
- 計算 q3,1、k3,1、v3,1：3 次矩陣乘法（$1\times4$ 乘 $4\times2$）
- 計算 q3,2、k3,2、v3,2：3 次矩陣乘法
- 計算注意力分數（兩個頭）：2 次矩陣乘法（$1\times2$ 乘 $2\times3$）
- 計算加權和（兩個頭）：2 次矩陣乘法（$1\times3$ 乘 $3\times2$）

總計：約 10 次矩陣乘法，但注意：
- 沒有 KV Cache 的矩陣乘法大多是 $2\times4$ 乘 $4\times2$（較大）
- 有 KV Cache 的矩陣乘法大多是 $1\times4$ 乘 $4\times2$ 或 $1\times2$ 乘 $2\times3$（較小）

當序列長度 $n$ 很大時，差距會更明顯：
- 沒有 KV Cache：每次生成都要做 $O(n^2)$ 的計算
- 有 KV Cache：每次生成只做 $O(n)$ 的計算

### 7-3、記憶體佔用代價

但是，有 KV Cache 也要付出代價：

對於頭 1 的 KV Cache：
- K_cache：儲存了 2 個詞 × 2 維度 = 4 個數字
- V_cache：儲存了 2 個詞 × 2 維度 = 4 個數字

對於頭 2 的 KV Cache：
- K_cache：4 個數字
- V_cache：4 個數字

總 KV Cache 大小：16 個數字

當序列長度 $n$ 很大時：
- KV Cache 大小 = $2 \times n \times d_k \times h$（兩個頭各存 K 和 V）
- 這是典型的「用空間換時間」

痛點：KV Cache 透過犧牲 GPU 記憶體來換取計算速度，讓模型在生成長文本時能夠即時回應，但需要權衡記憶體容量與生成效率。