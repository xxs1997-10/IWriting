import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import json
import requests
import jieba
import jieba.posseg as pseg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import faiss
import pickle
from datetime import datetime
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import gradio as gr

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print(">>> 正在加载向量模型...")
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print(">>> 正在加载本地 FAISS 索引...")
faiss_index = faiss.read_index("bcc_faiss.index")
with open("bcc_metadata.pkl", "rb") as f:
    bcc_metadata = pickle.load(f)
print(f">>> 本地语料库加载完成，共 {faiss_index.ntotal} 条")

# ---------- 1. 分级评分量表（HSK3/4/5） ----------

RUBRICS = {
    "HSK3": {
        "内容与切题": {
            "说明": "内容是否符合题目要求；能否表达基本事实与简单感受",
            5: "完全切题，内容完整，包含具体事实或细节，有简单但清晰的个人感受或观点。",
            4: "基本切题，内容较完整，有一定具体描述，感受或观点基本清晰。",
            3: "大体切题，内容尚完整，但描述笼统，缺乏具体细节，感受较表面。",
            2: "部分偏题，内容简单重复，细节极少，观点模糊。",
            1: "大部分偏题或内容极度空洞，几乎无实质性表达。",
        },
        "篇章结构": {
            "说明": "是否有基本的开头、主体、结尾；段落是否清晰",
            5: "有清晰的开头、主体、结尾，段落划分合理，层次分明。",
            4: "三部分基本具备，段落划分基本合理，结构较清晰。",
            3: "三部分存在但界限不清，段落划分较随意，层次一般。",
            2: "缺少开头或结尾，段落划分混乱，结构不完整。",
            1: "无明显结构，内容堆砌，无段落划分意识。",
        },
        "语篇连贯性": {
            "说明": "句子之间是否有基本的衔接；表达是否连贯流畅",
            5: "句子衔接自然，能使用基本过渡词（然后、但是、所以等），表达连贯流畅。",
            4: "大部分句子衔接较自然，有一定过渡意识，偶有生硬但不影响理解。",
            3: "句子之间基本连贯，过渡词使用较少，部分表达略显跳跃。",
            2: "句子之间缺乏衔接，逻辑联系薄弱，理解有一定困难。",
            1: "句子堆砌，前后逻辑混乱，严重影响理解。",
        },
        "词汇运用": {
            "说明": "是否能正确使用HSK3级核心词汇（约600词）；选词是否基本准确",
            5: "正确使用HSK3级核心词汇，有少量HSK4级词汇，几乎无明显用词错误，有初步的同义替换意识。",
            4: "正确使用HSK3级词汇为主，用词基本准确，用词错误≤1处，不影响理解。",
            3: "以HSK3级常用词为主，词汇较单一，用词错误2-3处，整体能表达意思。",
            2: "词汇贫乏，频繁重复使用少量词语，用词错误4-5处，影响部分表达。",
            1: "词汇严重不足，大量用词错误（≥6处），严重影响理解。",
        },
        "语法准确性": {
            "说明": "是否能正确使用HSK3级基本语法（把字句、被字句、比较句、简单补语、基本连词）；语序是否正确；标点是否基本恰当",
            5: "语法错误0处；标点使用基本正确（0-1处错误）；语序自然。",
            4: "语法错误1-2处；标点错误1-2处；不影响整体理解。",
            3: "语法错误3-4处；标点错误3-4处；对阅读有一定影响。",
            2: "语法错误5-6处；标点错误5-6处；影响理解。",
            1: "语法错误≥7处；标点错误≥7处；严重影响理解。",
        },
    },
    "HSK4": {
        "内容与切题": {
            "说明": "内容是否符合题目要求；材料是否具体充实；是否有较清晰的观点或感悟",
            5: "紧扣题意，内容充实，有具体细节支撑，观点或感悟较清晰且有一定深度。",
            4: "基本切题，内容较充实，有一定细节，观点基本清晰。",
            3: "大体切题但有明显偏离，内容尚完整，细节不足，观点较表面。",
            2: "部分内容偏离题目，材料单薄，观点模糊或缺乏支撑。",
            1: "大部分内容与题目无关，内容空洞，几乎无实质性观点。",
        },
        "篇章结构": {
            "说明": "是否有完整的开头、主体、结尾；段落是否合理；是否有基本的首尾照应",
            5: "开头、主体、结尾功能明确，段落划分合理，有首尾照应，布局清晰。",
            4: "三部分基本齐全，段落划分较合理，有一定首尾照应意识。",
            3: "三部分存在但某部分功能不清，段落划分有不当，首尾照应不明显。",
            2: "缺少开头或结尾，段落划分随意，层次不清。",
            1: "完全缺乏结构，无明显段落划分，内容堆砌。",
        },
        "语篇连贯性": {
            "说明": "句与句、段与段之间是否衔接自然；是否能使用连接词组织文章",
            5: "全文衔接自然，能恰当使用多种连接词（不仅…而且、虽然…但是、因此等），逻辑清晰。",
            4: "大部分衔接自然，连接词使用基本准确，个别处略显生硬但不影响理解。",
            3: "衔接基本可以，连接词使用不足或有误，部分内容有跳跃感。",
            2: "衔接生硬，连接词缺失或使用错误，句段间逻辑联系薄弱。",
            1: "几乎无衔接，内容堆砌，前后逻辑混乱，严重影响理解。",
        },
        "词汇运用": {
            "说明": "是否能恰当使用HSK4级词汇（约1200词）；是否有一定词汇多样性",
            5: "HSK4级词汇使用准确，有同义替换意识，有少量HSK5级词汇，用词错误0处。",
            4: "HSK4级词汇使用基本准确，有一定词汇多样性，用词错误≤1处。",
            3: "以HSK3-4级常用词为主，词汇较单一，用词错误2-3处。",
            2: "词汇量不足，停留在HSK3级以下水平为主，用词错误4-5处。",
            1: "词汇贫乏，大量错误用词（≥6处），严重影响表达。",
        },
        "语法准确性": {
            "说明": "是否能正确使用HSK4级语法（复句结构、趋向补语、程度补语、副词的准确使用）；标点是否恰当",
            5: "语法错误0处（含复句结构、补语、虚词）；标点使用恰当（0-1处错误）。",
            4: "语法错误1-2处；标点错误1-2处；不影响整体理解。",
            3: "语法错误3-4处；标点错误3-4处；对阅读有一定影响。",
            2: "语法错误5-6处；标点错误5-6处；影响理解。",
            1: "语法错误≥7处；标点错误≥7处；严重影响理解。",
        },
    },
    "HSK5": {
        "内容与切题": {
            "说明": "内容是否紧扣题意；论述是否充分有力；是否有深刻的见解或分析",
            5: "紧扣题意，论述充分有力，有具体例证或深刻分析，见解独到且有说服力。",
            4: "基本切题，论述较充分，有一定分析，观点较清晰，有一定说服力。",
            3: "大体切题，论述尚完整，但分析较表面，缺乏有力的例证支撑。",
            2: "部分偏题，论述单薄，缺乏有效分析，观点模糊或论据不足。",
            1: "大部分偏题或内容空洞，几乎无实质性论述或分析。",
        },
        "篇章结构": {
            "说明": "是否有严谨的论述结构（提出观点—分析论证—总结）；段落功能是否明确；首尾是否照应",
            5: "论述结构严谨，各段落功能明确，首尾照应清晰，布局合理，层次分明。",
            4: "结构基本严谨，段落功能较明确，有首尾照应，偶有过渡略显生硬。",
            3: "结构基本完整，但某部分功能不清，首尾照应不明显，层次一般。",
            2: "结构不够完整，段落功能混乱，缺乏首尾照应，层次不清。",
            1: "完全缺乏论述结构，内容堆砌，无法辨认论述逻辑。",
        },
        "语篇连贯性": {
            "说明": "句段衔接是否紧密流畅；是否能灵活使用多种连接手段；逻辑推进是否清晰",
            5: "全文衔接紧密流畅，能灵活使用多种连接手段（连词、指代、省略等），逻辑推进清晰，读者无需费力理解。",
            4: "大部分衔接自然，连接手段使用较多样，个别处略显生硬但整体流畅。",
            3: "衔接基本可以，连接手段较单一，部分段落之间逻辑跳跃感明显。",
            2: "衔接较生硬，连接手段缺乏或使用错误，逻辑推进不清晰，理解有困难。",
            1: "几乎无衔接，内容堆砌，逻辑混乱，严重影响理解。",
        },
        "词汇运用": {
            "说明": "是否能准确使用HSK5级词汇（约2500词）；是否有丰富的词汇多样性；是否能区分近义词",
            5: "HSK5级词汇使用准确丰富，有明显的同义替换和词汇变化，能正确区分近义词，用词错误0处。",
            4: "HSK5级词汇使用较准确，有一定词汇多样性，偶有近义词混用，用词错误≤1处。",
            3: "词汇较单一，停留在HSK4级水平为主，近义词区分能力不足，用词错误2-3处。",
            2: "词汇量明显不足，以HSK3-4级基础词为主，用词错误4-5处，影响表达质量。",
            1: "词汇贫乏，大量错误用词（≥6处），严重影响表达与理解。",
        },
        "语法准确性": {
            "说明": "是否能正确使用HSK5级复杂语法（复杂复句、固定格式、虚词的精准使用、特殊句式）；句式是否多样；标点是否恰当",
            5: "语法错误0处；句式多样，能使用复杂句式；标点使用准确（0-1处错误）。",
            4: "语法错误1-2处；句式有一定变化；标点错误1-2处，不影响理解。",
            3: "语法错误3-4处；句式较单一；标点错误3-4处，对阅读有一定影响。",
            2: "语法错误5-6处；句式单调重复；标点错误5-6处，影响理解。",
            1: "语法错误≥7处；句式混乱；标点错误≥7处，严重影响理解。",
        },
    },
    "HSK6": {
        "内容与切题": {"说明": "内容是否切合题意，论述是否深刻有力"},
        "篇章结构": {"说明": "结构是否严谨完整，布局是否合理"},
        "语篇连贯性": {"说明": "全文是否连贯流畅，逻辑是否清晰"},
        "词汇运用": {"说明": "词汇是否丰富准确，是否能灵活运用高级词汇"},
        "语法准确性": {"说明": "语法是否准确，句式是否多样复杂"},
    },
}

def build_rubric_prompt(hsk_level):
    """根据HSK等级构建评分标准说明文本"""
    level_key = hsk_level if hsk_level in RUBRICS else "HSK5"
    rubric = RUBRICS[level_key]
    dims = ["内容与切题", "篇章结构", "语篇连贯性", "词汇运用", "语法准确性"]
    lines = [f"【{hsk_level}级作文评分标准】"]
    for dim in dims:
        info = rubric.get(dim, {})
        lines.append(f"\n维度：{dim}（{info.get('说明','')}）")
        for score in [5, 4, 3, 2, 1]:
            if score in info:
                lines.append(f"  {score}分：{info[score]}")
    lines.append("\n【语法准确性特别说明】若语法错误极少（0-1处）但标点错误较多，应在语法错误对应分值基础上适当上调1分，以避免标点问题导致语法评分过低的悖论。")
    lines.append("\n【评分分布说明】本量表的正常评分分布为：大部分作文得分应在2-4分之间，少部分作文可得1分或5分。请勿集中给出高分，评分应真实反映作文水平。")
    return "\n".join(lines)

# ---------- 2. 科研数据自动保存 ----------

def save_to_research_log(title, hsk_level, scores, focus, essay):
    file_path = "research_data.csv"
    active_focus = "、".join(focus) if isinstance(focus, list) else focus
    clean_scores = [float(s) for s in scores]
    dims = ["内容与切题", "篇章结构", "语篇连贯性", "词汇运用", "语法准确性"]
    log_entry = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "题目": title,
        "目标等级": hsk_level,
        "反馈重点": active_focus,
        "学生原文内容": essay.replace('\n', ' [换行] '),
    }
    for i, dim in enumerate(dims):
        log_entry[f"{dim}得分"] = clean_scores[i]
    log_entry["总分"] = round(sum(clean_scores), 1)
    df = pd.DataFrame([log_entry])
    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False, encoding='utf-8-sig')

# ---------- 3. HSK词汇分析 ----------

def analyze_hsk_level(essay, student_level_str):
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        vocab_path = os.path.join(base_path, "hsk_vocab.csv")
        if not os.path.exists(vocab_path):
            return f"<div style='color:red;'>[系统预警] 未能找到词表文件：{vocab_path}</div>"

        df_vocab = pd.read_csv(vocab_path, encoding='utf-8-sig')
        df_vocab = df_vocab.map(lambda x: str(x).strip() if isinstance(x, str) else x)

        col_word = next((c for c in df_vocab.columns if '词' in c), None)
        col_level = next((c for c in df_vocab.columns if '级' in c), None)
        col_pinyin = next((c for c in df_vocab.columns if '音' in c), None)
        col_pos = next((c for c in df_vocab.columns if '性' in c), None)

        word_mapping = {}
        for _, row in df_vocab.iterrows():
            w = row[col_word]
            lv = row[col_level]
            py = row[col_pinyin] if col_pinyin else ""
            ps = row[col_pos] if col_pos else "通用"
            if w not in word_mapping:
                word_mapping[w] = {}
            word_mapping[w][ps] = {"level": lv, "pinyin": py}
            jieba.add_word(w)

        words_with_pos = pseg.lcut(essay)
        analysis_results = []
        target_lvl_num = int(''.join(filter(str.isdigit, student_level_str))) if any(c.isdigit() for c in student_level_str) else 0
        seen_words = set()
        level_weight = {"一":1, "二":2, "三":3, "四":4, "五":5, "六":6, "七":7, "高":8, "九":9, "优":10}

        for w, pos_tag in words_with_pos:
            w = w.strip()
            if w in word_mapping and w not in seen_words and len(w) > 0:
                s_pos = "名" if pos_tag.startswith('n') else "动" if pos_tag.startswith('v') else "形" if pos_tag.startswith('a') else "通用"
                info = word_mapping[w].get(s_pos) or list(word_mapping[w].values())[0]
                lvl_str = info['level']
                cur_lvl_num = next((v for k, v in level_weight.items() if k in lvl_str), 0)
                star = "⭐" if cur_lvl_num > target_lvl_num else ""
                analysis_results.append({
                    "word": w, "pinyin": info['pinyin'], "level": lvl_str,
                    "pos": pos_tag, "star": star, "weight": cur_lvl_num
                })
                seen_words.add(w)

        if not analysis_results:
            return "<div style='text-align:center; padding:15px; color:#999; border:1px dashed #ccc;'>暂未匹配到大纲词汇，建议检查CSV编码或增加作文长度。</div>"

        html = "<table style='width:100%; border-collapse:collapse; font-size:14px;'>"
        html += "<tr style='background:#f8f9fa; border-bottom:2px solid #4A90E2;'><th>词汇（拼音）</th><th>等级</th><th>词性</th><th>评估</th></tr>"
        for item in sorted(analysis_results, key=lambda x: x['weight'], reverse=True):
            html += f"<tr style='border-bottom:1px solid #eee;'>"
            html += f"<td style='padding:10px;'><b>{item['word']}</b><br><small style='color:#666;'>{item['pinyin']}</small></td>"
            html += f"<td>{item['level']}</td>"
            html += f"<td><span style='color:#4A90E2;'>{item['pos']}</span></td>"
            html += f"<td>{item['star']} {'超纲表达' if item['star'] else '运用达标'}</td></tr>"
        return html + "</table>"
    except Exception as e:
        return f"<div style='color:red;'>词库对齐异常: {e}</div>"

# ---------- 4. 雷达图（5分制） ----------

def create_radar_chart(scores):
    labels = ['内容切题', '篇章结构', '语篇连贯', '词汇运用', '语法准确']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    scores = [float(s) for s in scores]
    plot_scores = scores + [scores[0]]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, plot_scores, color='#4A90E2', alpha=0.3)
    ax.plot(angles, plot_scores, color='#4A90E2', linewidth=2, marker='o')
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], size=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return Image.open(buf)

# ---------- 5. 核心反馈逻辑 ----------

def get_feedback(title, essay, hsk_level, focus_areas):
    print(f">>> 正在执行写作诊断: 《{title}》 [{hsk_level}]")
    try:
        # RAG检索
        # 过滤关键词集合：单字高频虚词直接过滤，保留有语法意义的功能词
        FILTER_KEYWORDS = {
            '的', '地', '得', '了', '着', '过', '是', '在', '有', '到',
            '大', '小', '多', '少', '好', '坏', '上', '下', '里', '外',
            '来', '去', '说', '看', '想', '做', '用', '把', '被', '让',
            '给', '跟', '和', '与', '或', '也', '都', '就', '才', '又',
            '还', '再', '很', '太', '最', '更', '不', '没', '别', '每',
            '这', '那', '哪', '什么', '怎么', '为什么', '可以', '应该'
        }

        # 保留有语法意义的功能词（HSK核心语法点）
        KEEP_KEYWORDS = {
            '虽然', '但是', '因为', '所以', '如果', '就', '既然', '于是',
            '不但', '而且', '不仅', '也', '即使', '连', '把', '被', '比',
            '对', '向', '往', '离', '从', '到', '在', '给', '跟', '和',
            '或者', '还是', '要么', '宁可', '与其', '尽管', '然而', '因此',
            '所以', '从而', '进而', '反而', '却', '竟然', '居然', '幸好',
            '难道', '究竟', '到底', '简直', '几乎', '恐怕', '也许', '大概'
        }

        def should_keep_keyword(keyword):
            """判断关键词是否值得检索"""
            if not keyword or not isinstance(keyword, str):
                return False
            keyword = keyword.strip()
            # 保留列表里的直接保留
            if keyword in KEEP_KEYWORDS:
                return True
            # 单字且在过滤列表里则过滤
            if len(keyword) == 1 and keyword in FILTER_KEYWORDS:
                return False
            # 两字以上的词汇一般都保留
            if len(keyword) >= 2:
                return True
            return False

        # 按句子切分作文（只在句号、问号、感叹号处切分）
        import re
        sentences = re.split(r'(?<=[。！？])', essay)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]

        # 对每个句子分别检索，汇总结果去重
        all_candidates = []
        seen_ids = set()

        for sent in sentences:
            sent_vector = embed_model.encode([sent]).astype('float32')
            faiss.normalize_L2(sent_vector)
            sent_scores, sent_indices = faiss_index.search(sent_vector, 6)
            for score, idx in zip(sent_scores[0], sent_indices[0]):
                if score >= 0.5 and idx not in seen_ids:
                    meta = bcc_metadata[idx]
                    keyword = str(meta.get("keyword", "")).strip()
                    if should_keep_keyword(keyword):
                        all_candidates.append({
                            "text": meta["sentence"],
                            "keyword": keyword,
                            "is_error": meta["is_error"],
                            "sentence_clean": meta.get("sentence_clean", ""),
                            "sentence_display": meta.get("sentence_display", ""),
                            "score": round(float(score), 3)
                        })
                        seen_ids.add(idx)

        # 按相似度排序取前4条
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        similar_cases = all_candidates[:4]

        active_focus = "、".join(focus_areas) if focus_areas else "综合写作能力"
        corpus_text = "\n".join([
            f"- 原句：{c['text']}（关键词：{c['keyword']}，{'错误用法' if str(c['is_error']) == 'True' else '正确用法'}）"
            for c in similar_cases
        ]) if similar_cases else "暂无高相关语料"

        # 构建分级评分标准
        rubric_text = build_rubric_prompt(hsk_level)

        # 构建五个维度的评分要求
        dims = ["内容与切题", "篇章结构", "语篇连贯性", "词汇运用", "语法准确性"]
        focus_note = f"本次反馈重点关注【{active_focus}】，在该维度请给出更详细的分析与建议。"

        # 第一步：严格评分（temperature=0，不含反馈重点）
        scoring_prompt = f"""你是一名专业的国际中文教育评估专家。
请严格依据以下评分标准对该{hsk_level}级作文进行评分。

{rubric_text}

【学生作文题目】：{title}
【学生原文】：{essay}

【评分步骤，必须按顺序执行】
第一步：逐维度严格对照描述语，列出该维度发现的具体错误或不足，每条错误必须引用原文中的具体句子或词语。
第二步：根据错误数量和严重程度，对照描述语确定初步分数。
第三步：自我验证——重新阅读该维度的描述语，确认初步分数对应的描述语与作文实际表现吻合，如不吻合则修正。
第四步：输出最终分数。

【重要评分规则】
1. 评分分布：大部分作文得分应在2-4分之间，少部分作文可得1分或5分，请勿集中给出高分。
2. 词汇运用维度：必须依据评分标准中的HSK等级词汇要求判断，不得依赖模型自身的语感经验。
3. 语法准确性维度：若语法错误极少（0-1处）但标点错误较多，在语法分值基础上适当上调1分。
4. 汉字书写错误不在本次评分范围内，请忽略。
5. 每个维度必须先列出具体错误，再给出1-5分的整数评分。
6. 语法准确性维度：不能因为句子意思可以理解就忽略语法错误，必须检查：句式结构是否正确（如把字句、被字句是否误用）、动词与宾语搭配是否恰当、语序是否符合汉语规范，即使句意可懂也必须标注语法问题。

请严格以JSON格式输出，不得包含任何额外文字：
{{
  "scoring_process": {{
    "内容与切题": {{"errors": "<列出具体错误或不足，引用原文>", "initial_score": <初步分数>, "verification": "<验证说明>", "final_score": <最终分数>}},
    "篇章结构": {{"errors": "<列出具体错误或不足，引用原文>", "initial_score": <初步分数>, "verification": "<验证说明>", "final_score": <最终分数>}},
    "语篇连贯性": {{"errors": "<列出具体错误或不足，引用原文>", "initial_score": <初步分数>, "verification": "<验证说明>", "final_score": <最终分数>}},
    "词汇运用": {{"errors": "<列出具体错误或不足，引用原文>", "initial_score": <初步分数>, "verification": "<验证说明>", "final_score": <最终分数>}},
    "语法准确性": {{"errors": "<列出具体错误或不足，引用原文>", "initial_score": <初步分数>, "verification": "<验证说明>", "final_score": <最终分数>}}
  }},
  "scores": {{
    "内容与切题": <最终分数，1-5整数>,
    "篇章结构": <最终分数，1-5整数>,
    "语篇连贯性": <最终分数，1-5整数>,
    "词汇运用": <最终分数，1-5整数>,
    "语法准确性": <最终分数，1-5整数>
  }}
}}"""

        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

        payload_score = {
            "model": "deepseek-chat",
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "你是一个严格依据评分标准输出JSON的国际中文写作评估专家。评分必须先列出错误，再打分，再自我验证。scores字段只能是1-5的整数。"},
                {"role": "user", "content": scoring_prompt}
            ],
            "response_format": {"type": "json_object"}
        }

        resp_score = requests.post("https://api.deepseek.com/chat/completions", json=payload_score, headers=headers, timeout=90)
        data_score = json.loads(resp_score.json()['choices'][0]['message']['content'])

        # 评分结果拿到之后，再构建feedback_prompt
        score_summary = "\n".join([
            f"{dim}：{int(data_score.get('scores', {}).get(dim, 3))}分"
            for dim in ["内容与切题", "篇章结构", "语篇连贯性", "词汇运用", "语法准确性"]
        ])

        feedback_prompt = f"""你是一名专业的国际中文教育评估专家。
请依据以下评分标准和已确定的评分结果，对该{hsk_level}级作文的各维度给出具体反馈建议。

{rubric_text}

【已确定的评分结果】（反馈内容必须与此分数保持一致，不得矛盾）：
{score_summary}

【学生作文题目】：{title}
【学生原文】：{essay}
【语料库参考示例】：
{corpus_text}
【本次反馈重点】：{active_focus}（请在该维度给出更详细的分析与建议）

请严格以JSON格式输出：
{{
  "feedback": {{
    "内容与切题": "<针对该维度的具体评价与建议>",
    "篇章结构": "<针对该维度的具体评价与建议>",
    "语篇连贯性": "<针对该维度的具体评价与建议>",
    "词汇运用": "<针对该维度的具体评价与建议。请明确对比学生原文中的词汇表达与语料库参考示例的差距，指出学生哪些表达可以改进，并给出具体的改写方向>",
    "语法准确性": "<针对该维度的具体评价与建议>"
  }},
  "strengths": "<文章整体亮点，3-4句>",
  "suggestions": "<最需要改进的方向，结合语料库给出引导性建议，3-4句>"
}}"""

        payload_feedback = {
            "model": "deepseek-chat",
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": "你是一个严格依据评分标准输出JSON的国际中文写作评估专家。"},
                {"role": "user", "content": feedback_prompt}
            ],
            "response_format": {"type": "json_object"}
        }

        resp_feedback = requests.post("https://api.deepseek.com/chat/completions", json=payload_feedback, headers=headers, timeout=90)
        data = json.loads(resp_feedback.json()['choices'][0]['message']['content'])
        data['scores'] = data_score.get('scores', {})

        # 解析分数
        raw_scores = data.get('scores', {})
        dims = ["内容与切题", "篇章结构", "语篇连贯性", "词汇运用", "语法准确性"]
        current_scores = []
        for dim in dims:
            try:
                s = float(raw_scores.get(dim, 3))
                s = max(1, min(5, s))
                current_scores.append(s)
            except:
                current_scores.append(3.0)

        total = sum(current_scores)
        if total >= 23:
            total_label = "优秀"
        elif total >= 19:
            total_label = "良好"
        elif total >= 14:
            total_label = "一般"
        elif total >= 9:
            total_label = "较差"
        else:
            total_label = "很差"

        save_to_research_log(title, hsk_level, current_scores, focus_areas, essay)

        # 构建反馈报告
        feedback = data.get('feedback', {})
        dims_display = ["内容与切题", "篇章结构", "语篇连贯性", "词汇运用", "语法准确性"]
        dim_scores = {
            "内容与切题": int(current_scores[0]),
            "篇章结构": int(current_scores[1]),
            "语篇连贯性": int(current_scores[2]),
            "词汇运用": int(current_scores[3]),
            "语法准确性": int(current_scores[4]),
        }

        feedback_sections = ""
        for i, dim in enumerate(dims_display):
            is_focus = dim in focus_areas
            label = f"**{'⭐ ' if is_focus else ''}{'①②③④⑤'[i]} {dim}{'（重点反馈）' if is_focus else ''}**"
            feedback_sections += f"\n{label}\n{feedback.get(dim, '')}\n"

        report = f"""## 📊 综合评分：{int(total)} / 25　　等级：{total_label}

| 维度 | 得分 |
|------|------|
| 内容与切题 | {dim_scores['内容与切题']} / 5 |
| 篇章结构 | {dim_scores['篇章结构']} / 5 |
| 语篇连贯性 | {dim_scores['语篇连贯性']} / 5 |
| 词汇运用 | {dim_scores['词汇运用']} / 5 |
| 语法准确性 | {dim_scores['语法准确性']} / 5 |

---

### ✅ 文章亮点
{data.get('strengths', '')}

---

### 📝 各维度详细反馈
{feedback_sections}
---

### 🔍 改进建议
{data.get('suggestions', '')}
"""

        # 语料展示
        ref_html = "<div style='background:#fdf6e3; padding:15px; border-radius:8px; border-left:5px solid #e6db74;'>"
        ref_html += "<b>语料库参考示例：</b><ul style='margin-top:10px;'>"
        for c in similar_cases:
            label = "❌ 错误用法" if str(c['is_error']) == 'True' else "✅ 正确用法"
            display_text = c.get('sentence_display') or c.get('text', '')
            ref_html += f"<li style='margin-bottom:8px; color:#555;'>{label}｜关键词：<b>{c['keyword']}</b>｜{display_text}</li>"
        if not similar_cases:
            ref_html += "<li style='color:#999;'>暂未检索到相关语料</li>"
        ref_html += "</ul></div>"

        return create_radar_chart(current_scores), report, ref_html, analyze_hsk_level(essay, hsk_level)

    except Exception as e:
        return create_radar_chart([0]*5), f"## ⚠️ 诊断中断\n错误码：{e}", "检索失效", "匹配失效"

# ---------- 6. Gradio界面 ----------

with gr.Blocks(theme=gr.themes.Soft(), title="IWriting") as demo:
    gr.Markdown("# ✍️ IWriting　国际中文写作智能反馈系统")

    with gr.Row():
        with gr.Column(scale=2):
            t_in = gr.Textbox(label="📌 作文题目", placeholder="请输入题目...")
            e_in = gr.Textbox(label="📝 正文内容", lines=14, placeholder="请输入留学生作文正文...")
            with gr.Row():
                l_in = gr.Dropdown(["HSK3", "HSK4", "HSK5", "HSK6"], label="目标HSK等级", value="HSK4")
                f_in = gr.CheckboxGroup(
                    ["内容与切题", "篇章结构", "语篇连贯性", "词汇运用", "语法准确性"],
                    label="本次重点反馈维度", value=["词汇运用", "语法准确性"]
                )
            btn = gr.Button("🚀 提交诊断", variant="primary")

        with gr.Column(scale=3):
            plot_out = gr.Image(label="📊 五维能力雷达图（1-5分制）")
            text_out = gr.Markdown(label="💬 写作诊断报告")

    with gr.Tab("📖 语料溯源语料库参考示例"):
        html_out = gr.HTML()

    with gr.Tab("📊 HSK词汇分析"):
        standard_out = gr.HTML()

    btn.click(get_feedback, [t_in, e_in, l_in, f_in], [plot_out, text_out, html_out, standard_out])

if __name__ == "__main__":
    print(">>> 正在启动，请关注下方链接...")
    demo.launch(share=False)
