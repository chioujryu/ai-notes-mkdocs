# DeepSeek-R1

> 先備知識
>
> * 了解大型語言模型（LLM）如何用提示詞生成回答，以及什麼是 Chain-of-Thought（CoT）
> * 知道監督式微調（SFT）與強化學習（RL）在訓練流程中的角色差異
> * 了解「可驗證任務」的概念：答案可以用規則或程式自動判對錯（例如數學題、競賽程式）
> * 知道「獎勵模型（Reward Model）」與「reward hacking」大致在說什麼

## 1. 故事背景

### 1-1 推理能力變強了，但很多方法卡在「人類示範資料」的瓶頸

在複雜數學、程式除錯、邏輯推演等任務上，LLM 往往需要「多步推理」。過去常用的方法，是收集大量人類寫好的逐步推理示範（CoT），再用 SFT 教模型模仿。問題是：這類資料昂貴、難擴充，而且人類示範不一定是最有效的推理路徑，可能限制模型探索更好的策略。論文在摘要與導言就點出：既有成功高度依賴大量人類標註示範，但面對更複雜問題仍不足，並希望用 RL 讓模型自行產生更高階的推理模式（反思、驗證、策略調整）。 

### 1-2 他們想做一件反直覺的事：先不教推理步驟，只用 RL 逼出推理

想像你在教學生解題：傳統做法是給他「完整詳解」照抄；DeepSeek-R1 的出發點更像是「只判最後答案對不對」，讓學生自己長出解題方法。論文以 DeepSeek-V3-Base 為底座，採用 GRPO 做 RL，獎勵訊號主要依據「最終答案是否正確」，並刻意跳過在 RL 前先做 SFT 的傳統流程，避免把人類既定推理樣式硬塞給模型。 

### 1-3 純 RL 的 R1-Zero 長出很猛的推理行為，但「不好用」

DeepSeek-R1-Zero 透過純 RL 的自我演化，確實自然出現更長的思考、更常驗證與反思，甚至展示了所謂的 “aha moment” 例子。 
但它也帶來產品級痛點：輸出可讀性差、會語言混雜（例如中英混在同一段 CoT），而且訓練聚焦在可驗證推理任務，導致在寫作、開放式問答等更廣泛能力上表現受限。 

### 1-4 於是 DeepSeek-R1 走向「多階段管線」：先把人類可接受的外觀打底，再用 RL 強化

為了讓模型既會推理又好用，DeepSeek-R1 採用多階段流程：先用「少量 cold-start 長 CoT 資料」做 SFT，之後做推理導向 RL（加入語言一致性等設計），再透過 rejection sampling 與再一次 SFT，把推理與非推理資料一起納入，最後再加上一段對齊人類偏好的 RL，兼顧 helpfulness 與 harmlessness。 

## 2. 解決的痛點

### 2-1 痛點：高品質推理示範（CoT）太貴、太難擴；解法：用「可驗證任務」的 outcome-based RL 讓模型自我進化

在數學題或競賽程式這種任務裡，「最後答案」能被規則或程式判定正確與否。DeepSeek-R1 的想法是：與其花大錢收集逐步詳解，不如用 RL 讓模型試很多種解法，只要最後答對就給高獎勵，模型就會逐漸學會更有效的推理策略（例如自我檢查、反思、再嘗試）。這個方向在摘要中被明確定位為：用純 RL 來激勵推理能力，降低對人類標註推理軌跡的依賴。 

你可以把它想成：

* 以前：你要提供「每一步怎麼想」的標準答案
* 現在：你只要提供「最後答案怎麼驗證」，模型自己練出「中間怎麼想」

### 2-2 痛點：大規模 RL 訓練成本高、流程複雜；解法：採用 GRPO，用「群組採樣」估計優勢函數

論文採用 GRPO（Group Relative Policy Optimization）。直覺上，它對同一題 $q$ 一次抽一組回答 ${o_1,\dots,o_G}$，用同組內的獎勵做相對比較，來計算每個樣本的 advantage。 

對應到論文中的核心形式（以同樣符號寫法呈現）：

$$
J_{\mathrm{GRPO}}(\theta)=\mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\Big(\min\big(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\mathrm{old}}}(o_i|q)}A_i,\ \mathrm{clip}(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\mathrm{old}}}(o_i|q)},1-\epsilon,1+\epsilon)A_i\big)-\beta D_{\mathrm{KL}}(\pi_\theta|\pi_{\mathrm{ref}})\Big)\Bigg]
$$

$$
A_i=\frac{r_i-\mathrm{mean}({r_1,\dots,r_G})}{\mathrm{std}({r_1,\dots,r_G})}
$$

這種「同題多解、互相比較」的設計，讓 RL 的訊號更穩定，也更適合擴到大規模推理資料上。 

### 2-3 痛點：純 RL 產生的推理內容不易讀、還會語言混雜；解法：格式獎勵 + 語言一致性獎勵

DeepSeek-R1-Zero 的一個主要問題是「可讀性差」與「中英混雜」。 
為了讓推理輸出更像人類可用的內容，論文在訓練模板上要求把推理放在 `<think>...</think>`，答案放在 `<answer>...</answer>`，並在規則式獎勵中加入格式要求（format rewards），確保輸出結構固定、可解析。 

另外，為了抑制語言混雜，他們在 RL 中加入「語言一致性獎勵」，用 CoT 中目標語言詞比例來計算： 

$$
Reward_{\mathrm{language}}=\frac{\mathrm{Num}(Words_{\mathrm{target}})}{\mathrm{Num}(Words)}
$$

用一個簡單例子說明：

* 使用者用中文問問題，你不希望模型在思考過程突然插入大量英文專有詞堆疊
* 這個 reward 會鼓勵模型「整段思考更一致」，讓人更容易閱讀與審核

### 2-4 痛點：只會做可驗證推理，寫作與開放式任務變弱；解法：多階段訓練把「推理能力」與「通用對話能力」重新接回來

DeepSeek-R1 的多階段管線刻意把「推理資料」與「非推理資料」都納入，並在後段加入面向偏好對齊的訓練，目標是讓模型不只在數學與程式強，還能維持更好的寫作與開放式問答表現，同時提升 helpfulness 與 harmlessness。 

在獎勵面，他們也引入 model-based reward 的設計，例如 helpfulness reward 與 safety reward 的定義形式： 

$$
Reward_{\mathrm{helpful}}=RM_{\mathrm{helpful}}(Response_A,Response_B)
$$

$$
Reward_{\mathrm{safety}}=RM_{\mathrm{safety}}(Response)
$$

### 2-5 痛點：獎勵模型可能被「鑽漏洞」導致 reward 變高但實際能力變差；解法：把 reward hacking 當成必須監控與處理的工程問題

論文指出在使用 helpful reward model 時，可能出現 reward hacking：模型學會迎合獎勵模型的偏差，導致獎勵分數上升，但在真實任務（例如 CodeForces）表現下降。 
這等於提醒實作者：

* 不能只看「reward 曲線」就以為模型更好了
* 必須搭配外部基準測試與行為檢查，否則會訓練出「看起來被獎勵喜歡、但其實更不可靠」的模型 

### 2-6 痛點：強推理模型太大、太貴，社群與產品難落地；解法：公開權重與蒸餾小模型，讓更多人能用得起

為了降低使用門檻，論文強調釋出 DeepSeek-R1 與 DeepSeek-R1-Zero 的權重，並蒸餾出多個較小的 dense 模型（包含以 Qwen 與 Llama 為底座的版本），讓部署與研究更容易。 
蒸餾模型與對應底座在表格中列出，例如 Distill-Qwen-1.5B/7B/14B/32B 與 Distill-Llama-8B/70B。 

另外，論文也提供訓練成本的量級估算（以 H800 租用價假設），把 DeepSeek-R1-Zero、SFT 資料製作、DeepSeek-R1 的 GPU hours 與美元成本拆開呈現，讓「做同等級推理模型要花多少」更透明。 
