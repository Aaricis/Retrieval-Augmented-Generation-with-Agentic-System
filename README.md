# Build a Q&A LLM Agentic System to Answer Questions

## Project Overview

本项目实现了一个 **检索增强生成（RAG）** 系统，利用多个AI agent，通过结合网络搜索能力与大型语言模型（LLM）推理，来回答问题。

## System Architecture

该系统采用三个专门的agent协同工作。

1. **Question Extraction Agent**
   - **Role**: Professional question analyst
   - **Function**: Extracts core questions from complex descriptions
   - **Output**: Simplified, clear question statements
   
2. **Keyword Extraction Agent**
   - **Role**: Professional keyword extraction expert
   - **Function**: Identifies 2-5 optimal search keywords from questions
   - **Output**: Comma-separated keyword list
   
3. **QA Agent**
   - **Role**: Knowledge integrator and answer generator
   - **Function**: Generates answers based on retrieved context
   - **Output**: Traditional Chinese responses based on provided data

## Q&A Questions

90 道手工设计的题目，模型在训练阶段从未见过，题目的答案是一个词或一个短语。

[Q&A Questions](https://github.com/Aaricis/Retrieval-Augmented-Generation-with-Agentic-System/blob/main/qa.csv)

## Evaluation

使用gemini-2.5-flash来检查模型回答与标准答案是否一致。

```python
prompt = f"""
你是一个严谨的评测专家，请判断以下问题的两个答案是否一致，并给出 1~5 分评分与简短评语：

【问题】
{question}

【标准答案】
{gt_answer}

【学生回答】
{pred_answer}

评分范围：
5 = 完全正确；
4 = 大致正确，略有差异；
3 = 一般正确，有部分缺漏；
2 = 有重大偏差；
1 = 完全错误。

請用如下格式回答：
分數: X
評語: Y
"""
```

评分4-5分都是与标准答案较一致的回答。

## Q&A Pipeline Implementation

### version1

```
User Question -> QA Agent -> Fianl answer
```

### version2

```
User Question -> Google Search -> Web Content -> QA Agent -> Fianl answer
```

### version3

```
User Question -> Question Extraction Agent -> Core Question -> Keyword Extraction Agent -> Search Keywords -> Google Search -> Web Content -> QA Agent -> Fianl answer
```

### version4

```
User Question -> Question Extraction Agent -> Core Question -> Google Search -> Web Content -> QA Agent -> Fianl answer
```

## Prompt Engineering

现今流行的商用模型GPT-4、Gemini 2.5、Kimi K2等，拥有强大的语言能力，prompt不需要特定的格式，把需要的任务描述清楚即可。对于Llama-3.1-8B之类的小模型，设计prompt时需要一些技巧，例如提供额外咨询、提供范例等。经过不断尝试，才能获得满意的结果。

[Prompting techniques](https://www.youtube.com/watch?v=A3Yx35KrSN0) 

### Question Extraction Agent

```python
question_extraction_agent = LLMAgent(
    role_description=(
        "你是一位專業的問題分析師，擅長從冗長、含混或背景繁複的敘述中，"
        "準確抽取出最核心、需要被解答的完整問題。你只使用繁體中文回答。"
    ),
    task_description=(
        "請從下列敘述中，**萃取出一句內容完整、具體明確、可獨立理解的「疑問句」**，這句話必須清楚地表達使用者真正想知道的事情。\n\n"
        "請嚴格遵守以下規則：\n"
        "- 只輸出一句**具備問句語氣的問題句**，不可回述陳述句或背景敘述\n"
        "- **問句中必須包含問詞**，如「是誰」、「什麼」、「哪裡」、「為何」、「怎麼做」等\n"
        "- **保留與問題密切相關的背景資訊（如主體、時間、地點、專有名詞等）**，使問句本身具備足夠上下文，能獨立理解\n"
        "- **不得過度簡化**導致失去關鍵資訊，也**不得只留下背景資訊而忽略提問本身**\n"
        "- 禁止輸出任何說明、標點或多句內容，只能輸出一個完整的疑問句\n\n"
        "【正確範例】：\n"
        "敘述：卑南族是位在臺東平原的一個原住民族，以驍勇善戰、擅長巫術聞名，曾經統治整個臺東平原。相傳卑南族的祖先發源自 ruvuwa'an，該地位於現今的哪個行政區劃？\n"
        "✅正確輸出：卑南族祖先發源地 ruvuwa'an 位於現今的哪個行政區劃？\n"
        "❌錯誤輸出：該地位於現今的哪個行政區劃？（缺乏主體與背景，意義不明）\n\n"
        "請開始處理下方敘述："
    )
)
```

### Keyword Extraction Agent

```python
keyword_extraction_agent = LLMAgent(
    role_description=(
        "你是一位專業的搜尋關鍵字萃取專家，擅長從問題中提取出最適合用來搜尋的具體關鍵詞或短語。"
        "你只會用繁體中文回答。"
    ),
    task_description=(
        "請從下列問題中，**萃取出最能提升搜尋精準度的名詞或短語**，"
        "應包含有辨識力的具體資訊，例如：人名、課程名稱、機構、主題、時間、內容編號等。\n\n"
        "請避免：\n"
        "- 過於籠統的詞（如：問題、資料、內容）\n"
        "- 非名詞（如：想知道、了解、請問）\n\n"
        "請直接輸出以「逗號」分隔的關鍵字，不要加任何說明或多餘標點。\n\n"
        "【範例 1】\n"
        "輸入：李宏毅在2023年春季的《機器學習》課程中，第15個作業是什麼？\n"
        "輸出：李宏毅,2023年春季,機器學習,第15個作業\n\n"
        "【範例 2】\n"
        "輸入：請推薦台大電機系學生在大三可以選的通識好課。\n"
        "輸出：台大,電機系,大三,通識課,推薦課程\n\n"
        "【範例 3】\n"
        "輸入：為什麼貓會在你打電腦的時候跳到鍵盤上？\n"
        "輸出：貓,鍵盤行為,人貓互動,電腦\n"
    )
)
```

### QA Agent

```python
qa_agent = LLMAgent(
    role_description="你是极简回答器，仅用繁体中文。",
    task_description=(
        "请用一句繁体中文回答，开头固定为「答：」。\n"
        "无答案时回复「答：资料中未提及。」"
    )
)
```

## Results


| 模型                            | 方法                                                         | 正确率 |
| ------------------------------- | ------------------------------------------------------------ | ------ |
| Meta-Llama-3.1-8B-Instruct-Q8_0 | 直接回答                                                     | 14/90  |
| Meta-Llama-3.1-8B-Instruct-Q8_0 | RAG                                                          | 30/90  |
| Meta-Llama-3.1-8B-Instruct-Q8_0 | 抽取关键问题 + 抽取问题关键词 + Google Search+Prompt Engineering + RAG | 32/90  |
| Meta-Llama-3.1-8B-Instruct-Q8_0 | 抽取关键问题 + Google Search + Prompt Engineering + RAG      | 38/90  |
| qwen1_5-14b-chat-q6_k           | 抽取关键问题 + 抽取问题关键词 + Google Search+Prompt Engineering + RAG | 30/90  |
| qwen1_5-14b-chat-q6_k           | 抽取关键问题 + Google Search + Prompt Engineering + RAG      | 32/90  |

### 分析

1. 抽取关键问题之后，原本的用户输入已经非常简短。如果此时再抽取关键字，会过分简化原本的用户输入，搜到很多无关信息；
2. qwen模型虽然参数量超过llama，但是效果不如llama。这是因为**中文大语言模型在量化状态下处理数字 token 时常见的 tokenization 错误或截断现象**，尤其是连续的阿拉伯数字。例如，将’2014‘解析为’214‘，中间的'0'被丢弃；
3. 小模型指令跟随的能力不是很强，prompt要求只输出简短答案，但是模型不受控制的输出一长串解释。
