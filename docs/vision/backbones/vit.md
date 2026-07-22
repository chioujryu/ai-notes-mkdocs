# ViT
![alt text](../assets/vit/vit-01.png)
## **歷史背景**
- **2020 年**，谷歌研究團隊在論文《An Image is Worth 16x16 Words》中提出了 **Vision Transformer (ViT)**，將 **Transformer 模型**（原本在 NLP 領域的成功）引入到計算機視覺領域。
- **核心理念**: 
  - 將圖像切分為固定大小的補丁（patches），每個補丁類似於 NLP 中的「詞」（Token）。
  - 使用 Transformer 的注意力機制來建模圖像中不同補丁之間的關係。

---

## **解決的痛點**

1. **克服 CNN 的局限性**
   - **痛點**: 傳統卷積神經網路（CNN）依賴於局部感受野，難以捕捉全局關聯。
   - **解決方法**: ViT 使用 **Self-Attention 機制**，能夠直接捕捉圖像中補丁之間的全局依賴關係，無需通過多層堆疊的卷積層來逐漸擴展感受野。

2. **減少對手工設計的依賴**
   - **痛點**: CNN 的設計需要大量的專家知識（例如卷積核大小、池化策略等）。
   - **解決方法**: ViT 的設計簡單，只需要將圖像分割為補丁，並將其視為序列輸入，與 NLP 中的 Transformer 模型結構一致，減少了手工設計的需求。

3. **適應大規模數據訓練**
   - **痛點**: CNN 在小數據集上表現良好，但需要專門的設計（如數據增強）來避免過擬合。
   - **解決方法**: ViT 在大規模數據集（如 ImageNet-21k）上預訓練後，能夠很好地遷移到小數據集上進行微調，顯示出非常強的泛化性能。

4. **多模態統一**
   - **痛點**: CNN 與 NLP 的模型之間設計差異大，限制了多模態學習（如圖像-文本融合）。
   - **解決方法**: Transformer 在 NLP 和視覺領域的應用可以共享相同的架構，為多模態學習（例如 CLIP、DALL-E 等）奠定了基礎。

---

## Vision Transformer (ViT) 數學運算流程

### **假設的輸入**

假設輸入為一張 RGB 圖像：
- **圖像尺寸**: \(224 \times 224 \times 3\)
  - Height: \(224\)
  - Width: \(224\)
  - Channels: \(3\)

模型的超參數設置如下：
- **Patch Size**: \(16 \times 16\)（圖像被分割成 \(16 \times 16\) 的小塊）。
- **Hidden Dimension (\(d_{\text{model}}\))**: \(768\)（Transformer 的嵌入特徵維度）。
- **Number of Patches**:
  
  $$
  N = \frac{\text{Height}}{\text{Patch Size}} \times \frac{\text{Width}}{\text{Patch Size}} = \frac{224}{16} \times \frac{224}{16} = 14 \times 14 = 196
  \]
$$  圖像被分割成 \(196\) 個 \(16 \times 16\) 的小塊。
- **Number of Transformer Layers**: \(L = 12\)。
- **Number of Heads in Multi-Head Attention**: \(h = 12\)。
- **MLP Hidden Dimension**: \(3072\)（MLP 中間層的隱藏維度）。

---

### **Step 1: 圖像分割成 Patches**

1. **分割圖像為 Patches**：
   將圖像分割成 \(16 \times 16\) 的小塊。每個小塊的大小為：
   
   $$
   16 \times 16 \times 3
   \]
$$
2. **展平每個 Patch**：
   每個 \(16 \times 16 \times 3\) 的小塊被展平為一個向量：
   
   $$
   \text{Flattened Patch} = \text{Reshape}(16 \times 16 \times 3) = 768
   \]
$$   每個 Patch 的形狀為 \(\mathbb{R}^{768}\)。

3. **得到所有 Patches 的矩陣**：
   將 \(196\) 個展平的 Patch 組合成一個矩陣：
   
   $$
   P \in \mathbb{R}^{196 \times 768}
   \]
$$
---

### **Step 2: Linear Projection of Patches**

1. **線性投影**：
   使用一個線性投影矩陣 \(W_E\) 將每個 Patch 投影到 Transformer 的特徵空間：
   
   $$
   E = P \cdot W_E
   \]
$$   其中：
   - \(P \in \mathbb{R}^{196 \times 768}\) 是所有 Patches 的矩陣。
   - \(W_E \in \mathbb{R}^{768 \times 768}\) 是線性投影矩陣。
   - \(E \in \mathbb{R}^{196 \times 768}\) 是投影後的 Patch 嵌入。

2. **添加 Class Embedding**：
   添加一個可學習的 class embedding \(E_{\text{cls}} \in \mathbb{R}^{1 \times 768}\)，用於表示整張圖像的全局信息：
   
   $$
   E' = \text{Concat}(E_{\text{cls}}, E)
   \]
$$   現在，\(E'\) 的 shape 為：
   
   $$
   E' \in \mathbb{R}^{197 \times 768}
   \]
$$   （包含 \(196\) 個 Patch 和 \(1\) 個 class embedding）。

---

### **Step 3: 加入 Position Embedding**

1. **添加位置編碼**：
   為每個 Patch 和 class embedding 添加位置編碼（Position Embedding），詳細可以參考：
   
   $$
   E_{\text{pos}} = E' + P_{\text{pos}}
   \]
$$   其中：
   - \(P_{\text{pos}} \in \mathbb{R}^{197 \times 768}\) 是可學習的位置嵌入矩陣。

2. **結果的形狀**：
   添加位置編碼後的輸入矩陣：
   
   $$
   E_{\text{pos}} \in \mathbb{R}^{197 \times 768}
   \]
$$
---

### **Step 4: Transformer Encoder**

Transformer Encoder 包含 \(L = 12\) 層 Transformer Block，每層包含以下兩個主要部分：
1. **Multi-Head Attention**
2. **Feed-Forward Network (MLP)**

---

#### **Step 4.1: Multi-Head Attention**

1. **生成 Query (\(Q\))、Key (\(K\)) 和 Value (\(V\))**：
   使用三個線性投影矩陣 \(W_Q\)、\(W_K\)、\(W_V\) 將輸入 \(E_{\text{pos}}\) 投影到 Query、Key 和 Value 空間：
   
   $$
   Q = E_{\text{pos}} \cdot W_Q, \quad K = E_{\text{pos}} \cdot W_K, \quad V = E_{\text{pos}} \cdot W_V
   \]
$$   其中：
   - \(W_Q, W_K, W_V \in \mathbb{R}^{768 \times d_k}\)，且 \(d_k = \frac{d_{\text{model}}}{h} = \frac{768}{12} = 64\)。
   - \(Q, K, V \in \mathbb{R}^{197 \times 64}\)（每個頭的 Query、Key 和 Value）。

2. **計算 Scaled Dot-Product Attention**：
   
   $$
   \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V
   \]
$$   其中：
   - \(Q \cdot K^T \in \mathbb{R}^{197 \times 197}\)，表示每個 Token 對其他 Token 的注意力分數。
   - 通過 Softmax，得到注意力矩陣 \(A \in \mathbb{R}^{197 \times 197}\)。

3. **加權求和**：
   將注意力矩陣 \(A\) 應用到 \(V\) 上：
   
   $$
   Z = A \cdot V
   \]
$$   \(Z\) 的 shape 為：
   
   $$
   Z \in \mathbb{R}^{197 \times 64}
   \]
$$
4. **拼接多頭輸出**：
   將 \(12\) 個頭的輸出拼接起來：
   
   $$
   Z_{\text{concat}} = \text{Concat}(Z_1, Z_2, \dots, Z_{12})
   \]
$$   \(Z_{\text{concat}}\) 的 shape 為：
   
   $$
   Z_{\text{concat}} \in \mathbb{R}^{197 \times 768}
   \]
$$
5. **通過線性投影**：
   使用線性投影矩陣 \(W_O \in \mathbb{R}^{768 \times 768}\)：
   
   $$
   Z_{\text{final}} = Z_{\text{concat}} \cdot W_O
   \]
$$   \(Z_{\text{final}}\) 的 shape 為：
   
   $$
   Z_{\text{final}} \in \mathbb{R}^{197 \times 768}
   \]
$$
---

#### **Step 4.2: Feed-Forward Network (MLP)**

1. **MLP 結構**：
   
   $$
   \text{MLP}(x) = \sigma(x \cdot W_1 + b_1) \cdot W_2 + b_2
   \]
$$   其中：
   - \(W_1 \in \mathbb{R}^{768 \times 3072}\)
   - \(W_2 \in \mathbb{R}^{3072 \times 768}\)
   - \(b_1 \in \mathbb{R}^{3072}\)，\(b_2 \in \mathbb{R}^{768}\)。

2. **MLP 的輸入輸出**：
   - 輸入 \(x \in \mathbb{R}^{197 \times 768}\)。
   - 輸出 \(\text{MLP}(x) \in \mathbb{R}^{197 \times 768}\)。

---

#### **Step 4.3: Layer Normalization 和殘差連接**

1. **Layer Normalization**：
   對每層輸入進行標準化：
   
   $$
   \text{Norm}(x) = \frac{x - \mu}{\sigma}
   \]
$$
2. **殘差連接**：
   每層的輸出加上原始輸入：
   
   $$
   x' = \text{Norm}(x + \text{SubLayer}(x))
   \]
$$
---

### **Step 5: 最終分類輸出**

1. **提取 Class Token**：
   最終 Transformer Encoder 的輸出中，class embedding \(E_{\text{cls}}'\)：
   
   $$
   E_{\text{cls}}' \in \mathbb{R}^{1 \times 768}
   \]
$$
2. **MLP Head**：
   使用全連接層將 class embedding 映射到類別空間：
   
   $$
   \text{Class Scores} = E_{\text{cls}}' \cdot W_{\text{head}} + b_{\text{head}}
   \]
$$   其中：
   - \(W_{\text{head}} \in \mathbb{R}^{768 \times \text{num_classes}}\)
   - \(b_{\text{head}} \in \mathbb{R}^{\text{num_classes}}\)

3. **輸出分類分數**：
   
   $$
   \text{Class Scores} \in \mathbb{R}^{\text{num_classes}}
   \]
$$
---

### **總結流程**
1. **Patch 分割**：將圖像分割成 \(196\) 個 \(768\)-維的 Patches。
2. **Transformer Encoder**：通過 \(12\) 層 Transformer Block，提取全局特徵。
3. **分類輸出**：使用 class embedding 和 MLP Head，輸出分類結果。

ViT 的核心是利用 Transformer 的全局注意力機制，捕捉圖像中不同區域的關聯性，實現高效的圖像分類。
