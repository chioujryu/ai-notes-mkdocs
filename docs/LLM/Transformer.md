# Transformer

![alt text](<assets/你一定要懂得 AI 筆記_v2/image.png>)
## **Encoder**
好的，我看到你給的圖是 Transformer 的 Encoder（BERT 左邊）和 Decoder（GPT 右邊）結構圖。你說只需要在一個 **Encoder Block** 裡一步一步做前向傳播，而且只包含 **Input Embedding → Positional Encoding → Multi-Head Attention** 這三個部分，那我們就用一個真實的數值例子來算。

---

### **1.假設輸入**
假設我們的輸入是一個簡單的句子：

```
"He likes AI"
```

假設我們詞彙表（vocab）很小，對應的 token ID 是：

```
He     → 1
likes  → 5
AI     → 8
```

所以輸入的 token IDs 是：
```
[1, 5, 8]
```

---

### **2.Input Embedding**
假設 embedding 維度是 \( d_\text{model} = 4 \)，詞嵌入矩陣 \( E \) 是：


$$
E =
\begin{bmatrix}
0.1 & 0.3 & 0.5 & 0.7 \\   % token 0
0.2 & 0.4 & 0.6 & 0.8 \\   % token 1 (He)
0.9 & 0.1 & 0.1 & 0.3 \\
0.1 & 0.2 & 0.1 & 0.9 \\
0.5 & 0.5 & 0.5 & 0.5 \\
0.6 & 0.6 & 0.4 & 0.4 \\   % token 5 (likes)
0.3 & 0.7 & 0.2 & 0.8 \\
0.9 & 0.5 & 0.3 & 0.1 \\
0.8 & 0.2 & 0.4 & 0.6     % token 8 (AI)
\end{bmatrix}
$$


查表得到：

$$
\text{Embedding} =
\begin{bmatrix}
0.2 & 0.4 & 0.6 & 0.8 \\  % He
0.6 & 0.6 & 0.4 & 0.4 \\  % likes
0.8 & 0.2 & 0.4 & 0.6     % AI
\end{bmatrix}
$$


---

### **3.Positional Encoding**
假設我們用最簡化的（sinusoidal）位置編碼，結果如下（假設已計算好）：


$$
PE =
\begin{bmatrix}
0.00 & 1.00 & 0.00 & 1.00 \\ % pos 0
0.84 & 0.54 & 0.54 & 0.84 \\ % pos 1
0.91 & 0.42 & 0.91 & 0.42    % pos 2
\end{bmatrix}
$$


加入位置編碼：

$$
X = \text{Embedding} + PE =
\begin{bmatrix}
0.2 & 1.4 & 0.6 & 1.8 \\
1.44 & 1.14 & 0.94 & 1.24 \\
1.71 & 0.62 & 1.31 & 1.02
\end{bmatrix}
$$


---

### **4.Multi-Head Attention**
（假設 2 個頭，每頭維度 = 2）
![alt text](<assets/你一定要懂得 AI 筆記_v2/image-1.png>)
#### Step 4.1: 建立 Q, K, V
對於每一個頭，我們有不同的參數矩陣：

假設：

$$
W_Q^{(1)} =
\begin{bmatrix}
0.1 & 0.3 \\
0.2 & 0.4 \\
0.3 & 0.5 \\
0.4 & 0.6
\end{bmatrix}, \quad
W_K^{(1)} =
\begin{bmatrix}
0.2 & 0.1 \\
0.4 & 0.3 \\
0.6 & 0.5 \\
0.8 & 0.7
\end{bmatrix}, \quad
W_V^{(1)} =
\begin{bmatrix}
0.5 & 0.9 \\
0.1 & 0.3 \\
0.7 & 0.5 \\
0.2 & 0.4
\end{bmatrix}
$$


我們先計算第一個頭的 Q:

$$
Q^{(1)} = X \cdot W_Q^{(1)}
$$


以第一行為例：


$$
[0.2, 1.4, 0.6, 1.8] \cdot
\begin{bmatrix}
0.1 & 0.3 \\
0.2 & 0.4 \\
0.3 & 0.5 \\
0.4 & 0.6
\end{bmatrix}=
[ (0.2)(0.1)+(1.4)(0.2)+(0.6)(0.3)+(1.8)(0.4),\;
  (0.2)(0.3)+(1.4)(0.4)+(0.6)(0.5)+(1.8)(0.6) ]
$$


$$
= [0.02+0.28+0.18+0.72,\; 0.06+0.56+0.30+1.08]
$$


$$
= [1.20, 2.00]
$$


同理計算三行，假設結果：

$$
Q^{(1)} =
\begin{bmatrix}
1.20 & 2.00 \\
1.00 & 1.80 \\
1.10 & 1.90
\end{bmatrix}
$$


同樣計算 \(K^{(1)}\)、\(V^{(1)}\)（這裡省略細算，但原理相同）。

---

#### Step 4.2: Attention Score

$$
\text{Attention}(Q,K,V) = \text{Softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

假設 \(d_k = 2\)，則 \(\sqrt{d_k} = \sqrt{2} \approx 1.414\)。

假設計完 \(QK^T\) 後得到：

$$
QK^T =
\begin{bmatrix}
5.0 & 4.6 & 4.8 \\
4.6 & 4.4 & 4.5 \\
4.8 & 4.5 & 4.7
\end{bmatrix}
$$


除以 \(\sqrt{2}\)：

$$
\begin{bmatrix}
3.54 & 3.25 & 3.40 \\
3.25 & 3.11 & 3.18 \\
3.40 & 3.18 & 3.32
\end{bmatrix}
$$


---

#### Step 4.3 Softmax（每一行分別做）

我們現在有：

$$
\frac{QK^T}{\sqrt{d_k}} =
\begin{bmatrix}
3.54 & 3.25 & 3.40 \\
3.25 & 3.11 & 3.18 \\
3.40 & 3.18 & 3.32
\end{bmatrix}
$$


---

##### **第一行 Softmax**
先取 exp：

$$
e^{3.54} \approx 34.48,\quad e^{3.25} \approx 25.79,\quad e^{3.40} \approx 29.96
$$

總和：

$$
\text{sum} \approx 34.48 + 25.79 + 29.96 = 90.23
$$

Softmax：

$$
[\,0.382, \; 0.286, \; 0.332\,]
$$


---

##### **第二行 Softmax**

$$
e^{3.25} \approx 25.79,\quad e^{3.11} \approx 22.41,\quad e^{3.18} \approx 24.03
$$

總和：

$$
72.23
$$

Softmax：

$$
[\,0.357, \; 0.310, \; 0.333\,]
$$


---

##### **第三行 Softmax**

$$
e^{3.40} \approx 29.96,\quad e^{3.18} \approx 24.03,\quad e^{3.32} \approx 27.63
$$

總和：

$$
81.62
$$

Softmax：

$$
[\,0.367, \; 0.294, \; 0.339\,]
$$


---

因此注意力權重矩陣 \(A^{(1)}\) 為：

$$
A^{(1)} \approx
\begin{bmatrix}
0.382 & 0.286 & 0.332 \\
0.357 & 0.310 & 0.333 \\
0.367 & 0.294 & 0.339
\end{bmatrix}
$$


---

#### Step 4.4 乘 V

假設第一個頭的 \(V^{(1)}\) 已計算為：

$$
V^{(1)} =
\begin{bmatrix}
0.50 & 0.90 \\
0.10 & 0.30 \\
0.70 & 0.50
\end{bmatrix}
$$


計算：

$$
O^{(1)} = A^{(1)} \cdot V^{(1)}
$$


---

##### **第一行輸出**
第一列：

$$
0.382(0.50) + 0.286(0.10) + 0.332(0.70) = 0.191 + 0.0286 + 0.2324 \approx 0.452
$$

第二列：

$$
0.382(0.90) + 0.286(0.30) + 0.332(0.50) = 0.3438 + 0.0858 + 0.166 \approx 0.596
$$

→ \([0.452, 0.596]\)

---

##### **第二行輸出**

$$
[0.357, 0.310, 0.333] \cdot V^{(1)}
$$

第一列：

$$
0.1785 + 0.031 + 0.2331 \approx 0.443
$$

第二列：

$$
0.3213 + 0.093 + 0.1665 \approx 0.581
$$

→ \([0.443, 0.581]\)

---

#### **第三行輸出**

$$
[0.367, 0.294, 0.339] \cdot V^{(1)}
$$

第一列：

$$
0.1835 + 0.0294 + 0.2373 \approx 0.450
$$

第二列：

$$
0.3303 + 0.0882 + 0.1695 \approx 0.588
$$

→ \([0.450, 0.588]\)

---

因此第一個頭的輸出：

$$
O^{(1)} \approx
\begin{bmatrix}
0.452 & 0.596 \\
0.443 & 0.581 \\
0.450 & 0.588
\end{bmatrix}
$$


---

#### **第二個頭的計算（假設數據）**

為了節省篇幅，假設我們已經得到第二個頭的輸出：

$$
O^{(2)} \approx
\begin{bmatrix}
0.30 & 0.70 \\
0.60 & 0.90 \\
0.80 & 0.60
\end{bmatrix}
$$


---

#### **拼接所有頭**

$$
O_{\text{concat}} =
\begin{bmatrix}
0.452 & 0.596 & 0.300 & 0.700 \\
0.443 & 0.581 & 0.600 & 0.900 \\
0.450 & 0.588 & 0.800 & 0.600
\end{bmatrix}
$$


---

#### **輸出線性層 \(W_O\)**

假設：

$$
W_O =
\begin{bmatrix}
0.2 & 0.1 & 0.0 & 0.3 \\
0.0 & 0.2 & 0.4 & 0.1 \\
0.5 & 0.1 & 0.3 & 0.2 \\
0.1 & 0.0 & 0.2 & 0.4
\end{bmatrix}
$$


---

#### **Token 1**

$$
[0.452, 0.596, 0.300, 0.700] \cdot W_O
$$

1️⃣：

$$
0.452(0.2) + 0.596(0.0) + 0.300(0.5) + 0.700(0.1) = 0.0904+0+0.150+0.070=0.3104
$$

2️⃣：

$$
0.0452+0.1192+0.030+0=0.1944
$$

3️⃣：

$$
0+0.2384+0.090+0.140=0.4684
$$

4️⃣：

$$
0.1356+0.0596+0.060+0.280=0.5352
$$

→ \([0.310, 0.194, 0.468, 0.535]\)

---

同理計算 Token 2、Token 3（略），假設最後結果：

$$
\text{MHA_output} \approx
\begin{bmatrix}
0.310 & 0.194 & 0.468 & 0.535 \\
0.414 & 0.244 & 0.508 & 0.605 \\
0.470 & 0.270 & 0.520 & 0.590
\end{bmatrix}
$$



### **5. Add & Norm（第一次）**
（第一次殘差連接與層正規化）

#### 殘差相加
殘差輸入 \(X\)（Positional Encoding 後）：

$$
X =
\begin{bmatrix}
0.20 & 1.40 & 0.60 & 1.80 \\
1.44 & 1.14 & 0.94 & 1.24 \\
1.71 & 0.62 & 1.31 & 1.02
\end{bmatrix}
$$

多頭注意力輸出：

$$
\text{MHA_output} =
\begin{bmatrix}
0.310 & 0.194 & 0.468 & 0.535 \\
0.414 & 0.244 & 0.508 & 0.605 \\
0.470 & 0.270 & 0.520 & 0.590
\end{bmatrix}
$$


相加：

$$
S = X + \text{MHA_output} =
\begin{bmatrix}
0.510 & 1.594 & 1.068 & 2.335 \\
1.854 & 1.384 & 1.448 & 1.845 \\
2.180 & 0.890 & 1.830 & 1.610
\end{bmatrix}
$$


---

#### 層正規化（LayerNorm，γ=1, β=0）
對每一行做：

$$
\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$


**Token 1**：
- 均值：
\(\mu = (0.510 + 1.594 + 1.068 + 2.335)/4 \approx 1.37675\)  
- 方差：
\(\sigma^2 \approx 0.565\)，標準差 \(\sigma \approx 0.751\)  
- 標準化：

$$
[-1.153, 0.290, -0.410, 1.273]
$$


**Token 2**：
- \(\mu \approx 1.63275\)，\(\sigma \approx 0.183\)  
- 標準化：

$$
[1.210, -1.359, -1.006, 1.155]
$$


**Token 3**：
- \(\mu \approx 1.6275\)，\(\sigma \approx 0.547\)  
- 標準化：

$$
[1.010, -1.347, 0.371, -0.034]
$$


---

LayerNorm 後：

$$
Z_1 =
\begin{bmatrix}
-1.153 & 0.290 & -0.410 & 1.273 \\
1.210 & -1.359 & -1.006 & 1.155 \\
1.010 & -1.347 & 0.371 & -0.034
\end{bmatrix}
$$


---

### **6. Feed Forward Network（FFN）**

兩層全連接，中間 ReLU：

$$
\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2
$$


假設：

$$
W_1 =
\begin{bmatrix}
0.2 & 0.5 & 0.1 & 0.4 & 0.3 & 0.6 \\
0.7 & 0.2 & 0.3 & 0.0 & 0.5 & 0.1 \\
0.6 & 0.1 & 0.4 & 0.3 & 0.2 & 0.2 \\
0.0 & 0.3 & 0.5 & 0.2 & 0.4 & 0.1
\end{bmatrix}
,\quad
W_2 =
\begin{bmatrix}
0.1 & 0.3 & 0.2 & 0.5 \\
0.4 & 0.1 & 0.3 & 0.2 \\
0.2 & 0.4 & 0.5 & 0.3 \\
0.3 & 0.2 & 0.4 & 0.1 \\
0.5 & 0.3 & 0.1 & 0.2 \\
0.2 & 0.1 & 0.3 & 0.4
\end{bmatrix}
$$


---

#### Token 1：
第一層：

$$
[-1.153, 0.290, -0.410, 1.273] \cdot W_1 \approx [-0.326, 0.104, 0.344, 0.301, 0.305, 0.195]
$$

ReLU：
\([0, 0.104, 0.344, 0.301, 0.305, 0.195]\)

第二層：

$$
[0.316, 0.210, 0.265, 0.163]
$$


---

#### Token 2：
第一層：

$$
[1.210, -1.359, -1.006, 1.155] \cdot W_1 \approx [-0.372, -0.355, -0.454, 0.153, 0.040, -0.278]
$$

ReLU：
\([0, 0, 0, 0.153, 0.040, 0]\)

第二層：

$$
[0.058, 0.031, 0.066, 0.015]
$$


---

#### Token 3：
第一層：

$$
[1.010, -1.347, 0.371, -0.034] \cdot W_1 \approx [-0.423, -0.392, -0.354, -0.271, -0.290, -0.256]
$$

ReLU：
全為 0

第二層：
\([0, 0, 0, 0]\)

---

FFN 輸出：

$$
\text{FFN_output} =
\begin{bmatrix}
0.316 & 0.210 & 0.265 & 0.163 \\
0.058 & 0.031 & 0.066 & 0.015 \\
0     & 0     & 0     & 0
\end{bmatrix}
$$


---

### **7. Add & Norm（第二次）**

殘差相加：

$$
U = Z_1 + \text{FFN_output} =
\begin{bmatrix}
-0.837 & 0.500 & -0.145 & 1.436 \\
1.268 & -1.328 & -0.940 & 1.170 \\
1.010 & -1.347 & 0.371 & -0.034
\end{bmatrix}
$$


對每一行做 LayerNorm（略去重複計算步驟）：

**最終 Encoder Block 輸出**：

$$
\text{Output} \approx
\begin{bmatrix}
-1.046 & 0.049 & 0.531 & 0.466 \\
1.394 & -1.303 & -0.586 & 0.495 \\
1.010 & -1.347 & 0.371 & -0.034
\end{bmatrix}
$$










## **Decoder**
![alt text](<assets/你一定要懂得 AI 筆記_v2/image.png>)
### **1. Output Embedding**
假設我們的解碼器要生成句子 `"likes AI"`（對應 token ID: `[5, 8]`），詞嵌入矩陣 \( E \) 與之前相同：


$$
E =
\begin{bmatrix}
0.1 & 0.3 & 0.5 & 0.7 \\   % token 0
0.2 & 0.4 & 0.6 & 0.8 \\   % token 1
0.9 & 0.1 & 0.1 & 0.3 \\
0.1 & 0.2 & 0.1 & 0.9 \\
0.5 & 0.5 & 0.5 & 0.5 \\
0.6 & 0.6 & 0.4 & 0.4 \\   % token 5 (likes)
0.3 & 0.7 & 0.2 & 0.8 \\
0.9 & 0.5 & 0.3 & 0.1 \\
0.8 & 0.2 & 0.4 & 0.6     % token 8 (AI)
\end{bmatrix}
$$


查表：

$$
\text{Embedding} =
\begin{bmatrix}
0.6 & 0.6 & 0.4 & 0.4 \\  % likes
0.8 & 0.2 & 0.4 & 0.6     % AI
\end{bmatrix}
$$


---

### **2. Positional Encoding**
假設我們用相同的簡化正弦位置編碼（d_model=4）：


$$
PE =
\begin{bmatrix}
0.00 & 1.00 & 0.00 & 1.00 \\ % pos 0
0.84 & 0.54 & 0.54 & 0.84   % pos 1
\end{bmatrix}
$$


相加後：

$$
X =
\begin{bmatrix}
0.6 & 1.6 & 0.4 & 1.4 \\
1.64 & 0.74 & 0.94 & 1.44
\end{bmatrix}
$$

這就是進入 **Masked Multi-Head Attention** 的輸入。

---

### **3. Masked Multi-Head Attention**
假設我們有 **2 個頭**，每個頭維度 \(d_k = d_v = 2\)。

**頭1** 權重：

$$
W_Q^{(1)} =
\begin{bmatrix}
0.1 & 0.3 \\
0.2 & 0.4 \\
0.3 & 0.5 \\
0.4 & 0.6
\end{bmatrix},\quad
W_K^{(1)} =
\begin{bmatrix}
0.2 & 0.1 \\
0.4 & 0.3 \\
0.6 & 0.5 \\
0.8 & 0.7
\end{bmatrix},\quad
W_V^{(1)} =
\begin{bmatrix}
0.1 & 0.2 \\
0.3 & 0.4 \\
0.5 & 0.6 \\
0.7 & 0.8
\end{bmatrix}
$$


**頭2** 權重：

$$
W_Q^{(2)} =
\begin{bmatrix}
0.2 & 0.4 \\
0.3 & 0.5 \\
0.4 & 0.6 \\
0.5 & 0.7
\end{bmatrix},\quad
W_K^{(2)} =
\begin{bmatrix}
0.1 & 0.3 \\
0.2 & 0.4 \\
0.3 & 0.5 \\
0.4 & 0.6
\end{bmatrix},\quad
W_V^{(2)} =
\begin{bmatrix}
0.2 & 0.1 \\
0.4 & 0.3 \\
0.6 & 0.5 \\
0.8 & 0.7
\end{bmatrix}
$$


---

#### 3.1. 計算 Q, K, V

##### 頭1：


$$
Q^{(1)} = X \cdot W_Q^{(1)} =
\begin{bmatrix}
0.6 & 1.6 & 0.4 & 1.4 \\
1.64 & 0.74 & 0.94 & 1.44
\end{bmatrix}
\cdot
\begin{bmatrix}
0.1 & 0.3 \\
0.2 & 0.4 \\
0.3 & 0.5 \\
0.4 & 0.6
\end{bmatrix}=
\begin{bmatrix}
1.06 & 1.86 \\
1.17 & 2.122
\end{bmatrix}
$$



$$
K^{(1)} = X \cdot W_K^{(1)} =
\begin{bmatrix}
0.6 & 1.6 & 0.4 & 1.4 \\
1.64 & 0.74 & 0.94 & 1.44
\end{bmatrix}
\cdot
\begin{bmatrix}
0.2 & 0.1 \\
0.4 & 0.3 \\
0.6 & 0.5 \\
0.8 & 0.7
\end{bmatrix}=
\begin{bmatrix}
2.12 & 1.72 \\
2.34 & 1.864
\end{bmatrix}
$$



$$
V^{(1)} = X \cdot W_V^{(1)} =
\begin{bmatrix}
0.6 & 1.6 & 0.4 & 1.4 \\
1.64 & 0.74 & 0.94 & 1.44
\end{bmatrix}
\cdot
\begin{bmatrix}
0.1 & 0.2 \\
0.3 & 0.4 \\
0.5 & 0.6 \\
0.7 & 0.8
\end{bmatrix}=
\begin{bmatrix}
1.72 & 2.12 \\
1.864 & 2.34
\end{bmatrix}
$$


---

##### 頭2：

$$
Q^{(2)} = X \cdot W_Q^{(2)} =
\begin{bmatrix}
0.6 & 1.6 & 0.4 & 1.4 \\
1.64 & 0.74 & 0.94 & 1.44
\end{bmatrix}
\cdot
\begin{bmatrix}
0.2 & 0.4 \\
0.3 & 0.5 \\
0.4 & 0.6 \\
0.5 & 0.7
\end{bmatrix}=
\begin{bmatrix}
1.20 & 2.00 \\
1.312 & 2.192
\end{bmatrix}
$$



$$
K^{(2)} = X \cdot W_K^{(2)} =
\begin{bmatrix}
0.6 & 1.6 & 0.4 & 1.4 \\
1.64 & 0.74 & 0.94 & 1.44
\end{bmatrix}
\cdot
\begin{bmatrix}
0.1 & 0.3 \\
0.2 & 0.4 \\
0.3 & 0.5 \\
0.4 & 0.6
\end{bmatrix}=
\begin{bmatrix}
1.20 & 2.00 \\
1.312 & 2.192
\end{bmatrix}
$$



$$
V^{(2)} = X \cdot W_V^{(2)} =
\begin{bmatrix}
0.6 & 1.6 & 0.4 & 1.4 \\
1.64 & 0.74 & 0.94 & 1.44
\end{bmatrix}
\cdot
\begin{bmatrix}
0.2 & 0.1 \\
0.4 & 0.3 \\
0.6 & 0.5 \\
0.8 & 0.7
\end{bmatrix}=
\begin{bmatrix}
1.72 & 1.40 \\
1.864 & 1.544
\end{bmatrix}
$$


---

#### 3.2. 計算 Attention Scores（QK^T / sqrt(d_k)）

##### 頭1：

$$
\text{scores}^{(1)} = \frac{Q^{(1)} \cdot (K^{(1)})^T}{\sqrt{2}} =
\frac{
\begin{bmatrix}
1.06 & 1.86 \\
1.17 & 2.122
\end{bmatrix}
\cdot
\begin{bmatrix}
2.12 & 2.34 \\
1.72 & 1.864
\end{bmatrix}
}{\sqrt{2}}=
\begin{bmatrix}
3.853 & 4.208 \\
4.335 & 4.732
\end{bmatrix}
$$


---

##### 頭2：

$$
\text{scores}^{(2)} = \frac{Q^{(2)} \cdot (K^{(2)})^T}{\sqrt{2}} =
\frac{
\begin{bmatrix}
1.20 & 2.00 \\
1.312 & 2.192
\end{bmatrix}
\cdot
\begin{bmatrix}
1.20 & 1.312 \\
2.00 & 2.192
\end{bmatrix}
}{\sqrt{2}}=
\begin{bmatrix}
2.262 & 2.471 \\
2.701 & 2.949
\end{bmatrix}
$$


---

#### 3.3. Mask 未來位置（Decoder Mask）

對於 \(T=2\)，mask 後：


$$
\text{scores}^{(1)}_{\text{masked}} =
\begin{bmatrix}
3.853 & -\infty \\
4.335 & 4.732
\end{bmatrix},
\quad
\text{scores}^{(2)}_{\text{masked}} =
\begin{bmatrix}
2.262 & -\infty \\
2.701 & 2.949
\end{bmatrix}
$$


---

#### 3.4. Softmax


$$
\text{attn}^{(1)} =
\begin{bmatrix}
1 & 0 \\
0.401 & 0.599
\end{bmatrix},
\quad
\text{attn}^{(2)} =
\begin{bmatrix}
1 & 0 \\
0.438 & 0.562
\end{bmatrix}
$$


---

#### 3.5. 輸出每個頭（Attn ⋅ V）

##### 頭1：

$$
\text{head}_1 = \text{attn}^{(1)} \cdot V^{(1)} =
\begin{bmatrix}
1 & 0 \\
0.401 & 0.599
\end{bmatrix}
\cdot
\begin{bmatrix}
1.72 & 2.12 \\
1.864 & 2.34
\end{bmatrix}=
\begin{bmatrix}
1.72 & 2.12 \\
1.805 & 2.251
\end{bmatrix}
$$


##### 頭2：

$$
\text{head}_2 = \text{attn}^{(2)} \cdot V^{(2)} =
\begin{bmatrix}
1 & 0 \\
0.438 & 0.562
\end{bmatrix}
\cdot
\begin{bmatrix}
1.72 & 1.40 \\
1.864 & 1.544
\end{bmatrix}=
\begin{bmatrix}
1.72 & 1.40 \\
1.799 & 1.463
\end{bmatrix}
$$


---

#### 3.6. Concat 並投影

將兩個頭輸出 concat（沿特徵維度拼接）：

$$
\text{Concat} =
\begin{bmatrix}
1.72 & 2.12 & 1.72 & 1.40 \\
1.805 & 2.251 & 1.799 & 1.463
\end{bmatrix}
$$


假設輸出權重：

$$
W_O =
\begin{bmatrix}
0.2 & 0.4 & 0.6 & 0.8 \\
0.1 & 0.3 & 0.5 & 0.7 \\
0.9 & 0.7 & 0.5 & 0.3 \\
0.8 & 0.6 & 0.4 & 0.2
\end{bmatrix}
$$


最終：

$$
\text{MaskedMHA_out} = \text{Concat} \cdot W_O
$$


$$
\text{MaskedMHA_out} \approx
\begin{bmatrix}
3.224 & 3.368 & 3.512 & 3.656 \\
3.376 & 3.534 & 3.693 & 3.852
\end{bmatrix}
$$


### **4.Multi-Head Attention**

假設有 **2 個頭**，每個頭的維度 \(d_k = d_v = 2\)

---

**頭1權重**

$$
W_Q^{(1)} =
\begin{bmatrix}
0.1 & 0.3 \\
0.2 & 0.4 \\
0.3 & 0.5 \\
0.4 & 0.6
\end{bmatrix},\quad
W_K^{(1)} =
\begin{bmatrix}
0.2 & 0.1 \\
0.4 & 0.3 \\
0.6 & 0.5 \\
0.8 & 0.7
\end{bmatrix},\quad
W_V^{(1)} =
\begin{bmatrix}
0.1 & 0.2 \\
0.3 & 0.4 \\
0.5 & 0.6 \\
0.7 & 0.8
\end{bmatrix}
$$


---

**頭2權重**

$$
W_Q^{(2)} =
\begin{bmatrix}
0.2 & 0.4 \\
0.3 & 0.5 \\
0.4 & 0.6 \\
0.5 & 0.7
\end{bmatrix},\quad
W_K^{(2)} =
\begin{bmatrix}
0.1 & 0.3 \\
0.2 & 0.4 \\
0.3 & 0.5 \\
0.4 & 0.6
\end{bmatrix},\quad
W_V^{(2)} =
\begin{bmatrix}
0.2 & 0.1 \\
0.4 & 0.3 \\
0.6 & 0.5 \\
0.8 & 0.7
\end{bmatrix}
$$


---

#### **4.1 計算 Q, K, V**

**頭1：**

$$
Q^{(1)} = \text{MaskedMHA_out} \cdot W_Q^{(1)} =
\begin{bmatrix}
3.224 & 3.368 & 3.512 & 3.656 \\
3.376 & 3.534 & 3.693 & 3.852
\end{bmatrix}
\cdot
\begin{bmatrix}
0.1 & 0.3 \\
0.2 & 0.4 \\
0.3 & 0.5 \\
0.4 & 0.6
\end{bmatrix}=
\begin{bmatrix}
2.816 & 4.816 \\
2.948 & 5.038
\end{bmatrix}
$$



$$
K^{(1)} = \text{Enc_out} \cdot W_K^{(1)} =
\begin{bmatrix}
-1.046 & 0.049 & 0.531 & 0.466 \\
1.394 & -1.303 & -0.586 & 0.495 \\
1.010 & -1.347 & 0.371 & -0.034
\end{bmatrix}
\cdot
\begin{bmatrix}
0.2 & 0.1 \\
0.4 & 0.3 \\
0.6 & 0.5 \\
0.8 & 0.7
\end{bmatrix}=
\begin{bmatrix}
0.524 & 0.374 \\
-0.502 & -0.518 \\
-0.324 & -0.604
\end{bmatrix}
$$



$$
V^{(1)} = \text{Enc_out} \cdot W_V^{(1)} =
\begin{bmatrix}
-1.046 & 0.049 & 0.531 & 0.466 \\
1.394 & -1.303 & -0.586 & 0.495 \\
1.010 & -1.347 & 0.371 & -0.034
\end{bmatrix}
\cdot
\begin{bmatrix}
0.1 & 0.2 \\
0.3 & 0.4 \\
0.5 & 0.6 \\
0.7 & 0.8
\end{bmatrix}=
\begin{bmatrix}
0.44 & 0.54 \\
-0.388 & -0.394 \\
-0.292 & -0.444
\end{bmatrix}
$$


---

**頭2：**

$$
Q^{(2)} = \text{MaskedMHA_out} \cdot W_Q^{(2)} =
\begin{bmatrix}
3.224 & 3.368 & 3.512 & 3.656 \\
3.376 & 3.534 & 3.693 & 3.852
\end{bmatrix}
\cdot
\begin{bmatrix}
0.2 & 0.4 \\
0.3 & 0.5 \\
0.4 & 0.6 \\
0.5 & 0.7
\end{bmatrix}=
\begin{bmatrix}
3.648 & 5.648 \\
3.818 & 5.890
\end{bmatrix}
$$



$$
K^{(2)} = \text{Enc_out} \cdot W_K^{(2)} =
\begin{bmatrix}
-1.046 & 0.049 & 0.531 & 0.466 \\
1.394 & -1.303 & -0.586 & 0.495 \\
1.010 & -1.347 & 0.371 & -0.034
\end{bmatrix}
\cdot
\begin{bmatrix}
0.1 & 0.3 \\
0.2 & 0.4 \\
0.3 & 0.5 \\
0.4 & 0.6
\end{bmatrix}=
\begin{bmatrix}
0.324 & 0.674 \\
-0.258 & -0.478 \\
-0.194 & -0.514
\end{bmatrix}
$$



$$
V^{(2)} = \text{Enc_out} \cdot W_V^{(2)} =
\begin{bmatrix}
-1.046 & 0.049 & 0.531 & 0.466 \\
1.394 & -1.303 & -0.586 & 0.495 \\
1.010 & -1.347 & 0.371 & -0.034
\end{bmatrix}
\cdot
\begin{bmatrix}
0.2 & 0.1 \\
0.4 & 0.3 \\
0.6 & 0.5 \\
0.8 & 0.7
\end{bmatrix}=
\begin{bmatrix}
0.524 & 0.374 \\
-0.502 & -0.518 \\
-0.324 & -0.604
\end{bmatrix}
$$


#### **4.2 計算 scores**

**頭1**

$$
\text{scores}^{(1)} = \frac{Q^{(1)} \cdot (K^{(1)})^T}{\sqrt{2}} =
\frac{
\begin{bmatrix}
2.816 & 4.816 \\
2.948 & 5.038
\end{bmatrix}
\cdot
\begin{bmatrix}
0.524 & -0.502 & -0.324 \\
0.374 & -0.518 & -0.604
\end{bmatrix}
}{1.414}=
\begin{bmatrix}
2.156 & -3.392 & -3.888 \\
2.257 & -3.549 & -4.068
\end{bmatrix}
$$


**頭2**

$$
\text{scores}^{(2)} = \frac{Q^{(2)} \cdot (K^{(2)})^T}{\sqrt{2}} =
\frac{
\begin{bmatrix}
3.648 & 5.648 \\
3.818 & 5.890
\end{bmatrix}
\cdot
\begin{bmatrix}
0.324 & -0.258 & -0.194 \\
0.674 & -0.478 & -0.514
\end{bmatrix}
}{1.414}=
\begin{bmatrix}
3.999 & -4.743 & -4.870 \\
4.180 & -4.963 & -5.090
\end{bmatrix}
$$


---

#### **4.3 softmax 得到 attn**

**頭1**

$$
\text{attn}^{(1)} =
\begin{bmatrix}
0.999 & 0.0005 & 0.0005 \\
0.999 & 0.0005 & 0.0005
\end{bmatrix}
$$


**頭2**

$$
\text{attn}^{(2)} =
\begin{bmatrix}
0.999 & 0.0005 & 0.0005 \\
0.999 & 0.0005 & 0.0005
\end{bmatrix}
$$


（由於 scores 差異很大，softmax 幾乎全在第一列）

---

#### **4.4 計算 head 輸出**

**頭1**

$$
\text{head}_1 = \text{attn}^{(1)} \cdot V^{(1)} =
\begin{bmatrix}
0.999 & 0.0005 & 0.0005 \\
0.999 & 0.0005 & 0.0005
\end{bmatrix}
\cdot
\begin{bmatrix}
0.44 & 0.54 \\
-0.388 & -0.394 \\
-0.292 & -0.444
\end{bmatrix}=
\begin{bmatrix}
0.439 & 0.539 \\
0.439 & 0.539
\end{bmatrix}
$$


**頭2**

$$
\text{head}_2 = \text{attn}^{(2)} \cdot V^{(2)} =
\begin{bmatrix}
0.999 & 0.0005 & 0.0005 \\
0.999 & 0.0005 & 0.0005
\end{bmatrix}
\cdot
\begin{bmatrix}
0.524 & 0.374 \\
-0.502 & -0.518 \\
-0.324 & -0.604
\end{bmatrix}=
\begin{bmatrix}
0.523 & 0.373 \\
0.523 & 0.373
\end{bmatrix}
$$


---

#### **4.5 concat 兩個頭**


$$
\text{Concat} =
\begin{bmatrix}
0.439 & 0.539 & 0.523 & 0.373 \\
0.439 & 0.539 & 0.523 & 0.373
\end{bmatrix}
$$


---

#### **4.6 輸出投影**

假設輸出矩陣：

$$
W_O =
\begin{bmatrix}
0.2 & 0.4 & 0.6 & 0.8 \\
0.1 & 0.3 & 0.5 & 0.7 \\
0.9 & 0.7 & 0.5 & 0.3 \\
0.8 & 0.6 & 0.4 & 0.2
\end{bmatrix}
$$


計算：

$$
\text{MHA_out} = \text{Concat} \cdot W_O =
\begin{bmatrix}
0.439 & 0.539 & 0.523 & 0.373 \\
0.439 & 0.539 & 0.523 & 0.373
\end{bmatrix}
\cdot
\begin{bmatrix}
0.2 & 0.4 & 0.6 & 0.8 \\
0.1 & 0.3 & 0.5 & 0.7 \\
0.9 & 0.7 & 0.5 & 0.3 \\
0.8 & 0.6 & 0.4 & 0.2
\end{bmatrix}=
\begin{bmatrix}
1.285 & 1.199 & 1.113 & 1.027 \\
1.285 & 1.199 & 1.113 & 1.027
\end{bmatrix}
$$



### **5. 已知 MHA_out 與殘差輸入**

我們剛剛計算得到的 Encoder–Decoder Multi-Head Attention 輸出：

$$
\text{MHA_out} =
\begin{bmatrix}
1.285 & 1.199 & 1.113 & 1.027 \\
1.285 & 1.199 & 1.113 & 1.027
\end{bmatrix}
$$


殘差輸入是 **MaskedMHA_out**：

$$
\text{MaskedMHA_out} =
\begin{bmatrix}
3.224 & 3.368 & 3.512 & 3.656 \\
3.376 & 3.534 & 3.693 & 3.852
\end{bmatrix}
$$


---

### **6. Add & Norm（第一個 Add & Norm）**

殘差相加：

$$
\text{Add}_1 = \text{MaskedMHA_out} + \text{MHA_out} =
\begin{bmatrix}
3.224+1.285 & 3.368+1.199 & 3.512+1.113 & 3.656+1.027 \\
3.376+1.285 & 3.534+1.199 & 3.693+1.113 & 3.852+1.027
\end{bmatrix}=
\begin{bmatrix}
4.509 & 4.567 & 4.625 & 4.683 \\
4.661 & 4.733 & 4.806 & 4.879
\end{bmatrix}
$$


LayerNorm（假設此處簡化為均值歸一化，方便示範）：
均值 μ、標準差 σ 按行計算，得到：

$$
\text{Norm}_1 =
\begin{bmatrix}
-1.341 & -0.447 & 0.447 & 1.341 \\
-1.341 & -0.447 & 0.447 & 1.341
\end{bmatrix}
$$


---

### **7. Feed Forward**

假設 FFN 第一層權重：

$$
W_1 =
\begin{bmatrix}
0.2 & 0.4 & 0.6 & 0.8 \\
0.1 & 0.3 & 0.5 & 0.7 \\
0.9 & 0.7 & 0.5 & 0.3 \\
0.8 & 0.6 & 0.4 & 0.2
\end{bmatrix}
,\quad b_1 =
\begin{bmatrix}
0.1 & 0.2 & 0.3 & 0.4
\end{bmatrix}
$$


第一層：

$$
\text{FFN}_1 = \text{Norm}_1 \cdot W_1 =
\begin{bmatrix}
-1.341 & -0.447 & 0.447 & 1.341 \\
-1.341 & -0.447 & 0.447 & 1.341
\end{bmatrix}
\cdot
\begin{bmatrix}
0.2 & 0.4 & 0.6 & 0.8 \\
0.1 & 0.3 & 0.5 & 0.7 \\
0.9 & 0.7 & 0.5 & 0.3 \\
0.8 & 0.6 & 0.4 & 0.2
\end{bmatrix}=
\begin{bmatrix}
1.072 & 0.896 & 0.720 & 0.544 \\
1.072 & 0.896 & 0.720 & 0.544
\end{bmatrix}
$$

加偏置：

$$
\text{FFN}_1 + b_1 =
\begin{bmatrix}
1.172 & 1.096 & 1.020 & 0.944 \\
1.172 & 1.096 & 1.020 & 0.944
\end{bmatrix}
$$


激活（ReLU 不影響正數）：

$$
\text{Act} =
\begin{bmatrix}
1.172 & 1.096 & 1.020 & 0.944 \\
1.172 & 1.096 & 1.020 & 0.944
\end{bmatrix}
$$


---

第二層權重：

$$
W_2 =
\begin{bmatrix}
0.5 & 0.6 & 0.7 & 0.8 \\
0.4 & 0.5 & 0.6 & 0.7 \\
0.3 & 0.4 & 0.5 & 0.6 \\
0.2 & 0.3 & 0.4 & 0.5
\end{bmatrix}
,\quad b_2 =
\begin{bmatrix}
0.05 & 0.06 & 0.07 & 0.08
\end{bmatrix}
$$


第二層：

$$
\text{FFN}_2 = \text{Act} \cdot W_2 =
\begin{bmatrix}
1.172 & 1.096 & 1.020 & 0.944 \\
1.172 & 1.096 & 1.020 & 0.944
\end{bmatrix}
\cdot
\begin{bmatrix}
0.5 & 0.6 & 0.7 & 0.8 \\
0.4 & 0.5 & 0.6 & 0.7 \\
0.3 & 0.4 & 0.5 & 0.6 \\
0.2 & 0.3 & 0.4 & 0.5
\end{bmatrix}=
\begin{bmatrix}
1.270 & 1.680 & 2.090 & 2.500 \\
1.270 & 1.680 & 2.090 & 2.500
\end{bmatrix}
$$

加偏置：

$$
\text{FFN_out} =
\begin{bmatrix}
1.320 & 1.740 & 2.160 & 2.580 \\
1.320 & 1.740 & 2.160 & 2.580
\end{bmatrix}
$$


---

### **8. Add & Norm（第二次）**

殘差相加：

$$
\text{Add}_2 = \text{Norm}_1 + \text{FFN_out} =
\begin{bmatrix}
-1.341+1.320 & -0.447+1.740 & 0.447+2.160 & 1.341+2.580 \\
-1.341+1.320 & -0.447+1.740 & 0.447+2.160 & 1.341+2.580
\end{bmatrix}=
\begin{bmatrix}
-0.021 & 1.293 & 2.607 & 3.921 \\
-0.021 & 1.293 & 2.607 & 3.921
\end{bmatrix}
$$


LayerNorm：
均值 μ=1.95，σ≈1.604，得到：

$$
\text{Norm}_2 =
\begin{bmatrix}
-1.227 & -0.410 & 0.410 & 1.227 \\
-1.227 & -0.410 & 0.410 & 1.227
\end{bmatrix}
$$


---

### **9. Linear（輸出層）**

假設詞彙表大小 \(V=6\)，輸出層權重：

$$
W_{\text{out}} =
\begin{bmatrix}
0.2 & 0.1 & 0.4 & 0.3 & 0.5 & 0.6 \\
0.3 & 0.2 & 0.5 & 0.4 & 0.6 & 0.7 \\
0.4 & 0.3 & 0.6 & 0.5 & 0.7 & 0.8 \\
0.5 & 0.4 & 0.7 & 0.6 & 0.8 & 0.9
\end{bmatrix}
$$


計算：

$$
\text{Logits} = \text{Norm}_2 \cdot W_{\text{out}} =
\begin{bmatrix}
-1.227 & -0.410 & 0.410 & 1.227 \\
-1.227 & -0.410 & 0.410 & 1.227
\end{bmatrix}
\cdot
\begin{bmatrix}
0.2 & 0.1 & 0.4 & 0.3 & 0.5 & 0.6 \\
0.3 & 0.2 & 0.5 & 0.4 & 0.6 & 0.7 \\
0.4 & 0.3 & 0.6 & 0.5 & 0.7 & 0.8 \\
0.5 & 0.4 & 0.7 & 0.6 & 0.8 & 0.9
\end{bmatrix}=
\begin{bmatrix}
0.410 & 0.320 & 0.740 & 0.650 & 0.870 & 0.980 \\
0.410 & 0.320 & 0.740 & 0.650 & 0.870 & 0.980
\end{bmatrix}
$$


---

### **10. Softmax**

對每行做 softmax（沿詞彙表方向）：

$$
\text{Probs} =
\begin{bmatrix}
0.139 & 0.127 & 0.208 & 0.190 & 0.236 & 0.256 \\
0.139 & 0.127 & 0.208 & 0.190 & 0.236 & 0.256
\end{bmatrix}
$$


---

✅ **最終輸出**（每個位置的詞彙表概率分佈）：

$$
\boxed{
\begin{bmatrix}
0.139 & 0.127 & 0.208 & 0.190 & 0.236 & 0.256 \\
0.139 & 0.127 & 0.208 & 0.190 & 0.236 & 0.256
\end{bmatrix}
}
$$
