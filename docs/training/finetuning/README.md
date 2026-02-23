# ✅ **LLM / VLM Fine-tuning 方法比較總表（2025 最新版）**

| 方法                      | 全名                                      | 主要用途                    | 需要資料格式                       | 是否需 Reward Model  | 訓練難度     | 訓練成本         | 行為控制力    | 優點                     | 缺點                       |
| ----------------------- | --------------------------------------- | ----------------------- | ---------------------------- | ----------------- | -------- | ------------ | -------- | ---------------------- | ------------------------ |
| **SFT**                 | Supervised Fine-Tuning                  | 讓模型模仿示範答案               | 指令 → 回答（單一答案）                | ❌ 不需要             | ⭐ 最簡單    | ⭐ 最低         | ⭐ 基本     | 簡單可行、資料容易製作            | 容易「跟著錯答案」、無偏好學習、易胡扯      |
| **DPO**                 | Direct Preference Optimization          | 依照偏好選擇回答                | prompt + chosen + rejected   | ❌ 不需要             | ⭐⭐ 中等    | ⭐⭐ 中低        | ⭐⭐ 好     | 無需 Reward Model，對齊效果很好 | 需要負樣本（rejected），不能做複雜 RL |
| **ORPO**                | Odds-Ratio Preference Optimization      | 改良版 DPO，更穩定，資料只需 chosen | prompt + chosen（不需 rejected） | ❌ 不需要             | ⭐ 中等     | ⭐⭐ 中低        | ⭐⭐       | 訓練更穩、資料準備更簡單           | 缺少明確反例，效果略弱於 DPO         |
| **PPO（RLHF）**           | Proximal Policy Optimization            | 最強控制模型行為、最強對齊           | prompt → 模型 → reward score   | ✔ 需要 Reward Model | ⭐⭐⭐⭐⭐ 最難 | ⭐⭐⭐⭐⭐ 最高     | ⭐⭐⭐⭐⭐ 最強 | 控制力極強，可精細調整模型行為        | 成本爆炸、極難訓練、容易不穩定          |
| **KTO**                 | Kahneman-Tversky Optimization           | Anti-DPO，使用「心理偏好」       | prompt + chosen / rejected   | ❌ 不需要             | ⭐⭐       | ⭐⭐           | ⭐⭐       | 更穩定、不需 reward、對人類風格更友善 | 新方法，工具較少                 |
| **RLAIF**               | Reinforcement Learning from AI Feedback | 用 AI 自動產生 reward        | prompt → AI judge            | ✔ 需要 AI Judge     | ⭐⭐⭐⭐     | ⭐⭐⭐⭐         | ⭐⭐⭐⭐     | 不需要人類標註、效果接近 RLHF      | 仍需 reward 產生器（大模型）       |
| **QLoRA SFT**           | QLoRA Supervised FT                     | 便宜做 SFT                 | instruction → output         | ❌                 | ⭐        | ⭐❗極低（單張 GPU） | ⭐        | 省錢，常見方法                | 是 SFT，無對齊能力              |
| **QLoRA DPO**           | QLoRA DPO                               | 超便宜做 DPO                | prompt + chosen + rejected   | ❌                 | ⭐⭐       | ⭐❗低          | ⭐⭐       | 便宜做偏好學習                | 效果略弱於 full fine-tune     |
| **Adapters (LoRA/IA3)** | Low-Rank Adapters                       | 模型快速注入新知識               | instruction → output         | ❌                 | ⭐⭐       | ⭐⭐           | ⭐⭐       | 可更新小模組，不改整個模型          | 過度依賴舊模型能力                |

---

# 🔍 **最常用的三大 Fine-tuning 方法**

## **1️⃣ SFT（Supervised Fine-Tuning）**

**適合：訓練「模型能回答什麼」**

✔ 資料容易取得
✔ 適合教模型新技能
❌ 沒辦法讓模型「更安全」、「避免亂回答」、「偏好更好答案」

👉 **適合初步訓練模型基本能力**

---

## **2️⃣ DPO（Direct Preference Optimization）**

**適合：訓練「模型應該比較喜歡哪個回答」**

✔ 不需要 Reward model
✔ 比 RLHF 更穩定
✔ 效果接近 RLHF
✔ 訓練成本中等
❌ 需要 `chosen` 和 `rejected`（負樣本）

👉 **目前業界最常用的 alignment 方法（成本低 / 效果強）**

---

## **3️⃣ PPO（RLHF）**

**適合：強化控制模型行為、避免亂回覆、安全調優**

✔ 最強
✔ 可細緻調整模型答案
✔ OpenAI、Anthropic、Google 等使用
❌ 成本最高
❌ 必需 Reward Model
❌ 訓練不穩定

👉 **只有大公司花 10M+ 美金才會用 RLHF**

---

# 🎯 **依照你的需求給建議（最重要）**

你目前要微調 InternVL（VLM）
如果你的需求是：

## 🚩 **我只是想讓模型看得懂我的圖片、產生我想要的回答**

→ **SFT** 就可以了
→ 不需要 DPO / RLHF

## 🚩 **我要模型避免亂回答、回答風格統一、偏好特定類型答案**

→ **DPO（最推薦）**

## 🚩 **我想把模型打造成類似 GPT-4V 的行為模式、安全且穩定**

→ 需要 **RLHF / PPO**（訓練成本巨大）

---

# 🧩 想要我幫你產生什麼？

我可以提供：

### 🔧 1. **SFT 範例（InternVL 單圖微調完整程式）**

### 🔧 2. **DPO 範例（TRL + InternVL）**

### 🔧 3. **你的資料轉換成 DPO 格式的工具**

### 🔧 4. **完整微調 pipeline（資料 → 訓練 → infer）**

### 🔧 5. **製作 InternVL 單圖 / 多圖資料集的方法**

你想要哪一個？
