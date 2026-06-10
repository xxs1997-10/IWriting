# IWriting · 国际中文写作智能反馈系统

针对来华/海外留学生的中文作文批改工具，基于 Gradio + DeepSeek + FAISS RAG。

## 文件结构

| 文件 | 作用 |
|---|---|
| `final_agent.py` | 主程序（约 490 行） |
| `start.bat` | 双击启动脚本（推荐使用） |
| `bcc_faiss.index` | BCC 语料库 FAISS 索引（本地资源，不入 git） |
| `bcc_metadata.pkl` | BCC 语料元数据 |
| `hsk_vocab.csv` | HSK 词汇表 |
| `hsk_char.csv` | HSK 汉字表 |
| `bcc_corpus_clean.csv` | BCC 语料原文 |
| `research_data.csv` | 科研数据自动存档（运行时生成） |
| `venv/` | Python 虚拟环境（不入 git） |
| `.env` | API 密钥（不入 git） |

## 启动方式

### 方式 1：双击桌面 IWriting 图标（推荐）
- 桌面有个 `IWriting.lnk`，双击即启动
- 控制台黑框保持开着，关闭即退出程序

### 方式 2：直接跑 start.bat
- 在项目根目录双击 `start.bat`
- 自动激活 venv、启动 Gradio、3 秒后浏览器自动打开

### 方式 3：VSCode 运行
- VSCode 打开 `final_agent.py`，点运行
- 浏览器访问 `http://localhost:7860`

## 功能

- 5 维评分（内容/结构/连贯性/词汇/语法），每维 1-5 分，总分 25
- HSK3/4/5/6 分级评分量表（rubric 1-5 分逐档详尽描述）
- BCC 语料库 RAG 检索：学生作文 → 相似语料 → 喂给 AI 当参考
- HSK 词汇超纲分析（jieba 分词 + 词表匹配，超纲词标 ⭐）
- 五维雷达图可视化
- 多维度反馈重点（老师可指定本次重点看哪几项）
- 科研数据自动存档（`research_data.csv`）

## 环境依赖

- Python 3.x
- venv
- 见 `requirements.txt`（如果存在）
- `.env` 中需配置 `DEEPSEEK_API_KEY`

## 反馈流程

1. 输入作文题目 + 正文
2. 选择目标 HSK 等级（HSK3/4/5/6）
3. 选择本次重点反馈维度
4. 点击「提交诊断」
5. 输出：雷达图 + 详细诊断报告 + 语料溯源 + HSK 词汇分析
