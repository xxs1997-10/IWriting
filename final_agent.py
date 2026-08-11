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
BAIDU_APP_ID = os.getenv("BAIDU_APP_ID")
BAIDU_SECRET_KEY = os.getenv("BAIDU_SECRET_KEY")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print(">>> 正在加载向量模型...")
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print(">>> 正在加载本地 FAISS 索引...")
faiss_index = faiss.read_index("bcc_faiss_v2.index")
with open("bcc_metadata_v2.pkl", "rb") as f:
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
        lines.append(f"\n维度:{dim}（{info.get('说明','')}）")
        for score in [5, 4, 3, 2, 1]:
            if score in info:
                lines.append(f"  {score}分:{info[score]}")
    lines.append("\n【语法准确性特别说明】若语法错误极少（0-1处）但标点错误较多，应在语法错误对应分值基础上适当上调1分，以避免标点问题导致语法评分过低的悖论。")
    lines.append("\n【评分分布说明】本量表的正常评分分布为:大部分作文得分应在2-4分之间，少部分作文可得1分或5分。请勿集中给出高分，评分应真实反映作文水平。")
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
            return f"<div style='color:red;'>未能找到词表文件</div>"

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
        level_weight = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"高":8,"九":9,"优":10}
        level_name = {1:"一级",2:"二级",3:"三级",4:"四级",5:"五级",6:"六级",7:"七级",8:"高等"}

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
            return "<div style='text-align:center;padding:15px;color:#999;'>暂未匹配到词汇。</div>"

        # 统计各等级词汇数量（用于柱形图）
        level_counts = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 8:0}
        for item in analysis_results:
            if item['weight'] in level_counts:
                level_counts[item['weight']] += 1

        # 生成柱形图
        fig, ax = plt.subplots(figsize=(7, 3))
        level_keys = [1, 2, 3, 4, 5, 6, 8]
        levels = [level_name[i] for i in level_keys]
        counts = [level_counts[i] for i in level_keys]
        colors = ['#C5DAFA' if i <= target_lvl_num else '#4CAF50' for i in level_keys]
        bars = ax.bar(levels, counts, color=colors, edgecolor='white', linewidth=0.5)

        ax.set_xlabel('HSK等级', fontsize=10)
        ax.set_ylabel('词汇数量', fontsize=10)
        ax.set_title('词汇等级分布', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom', fontsize=9)

        import base64
        buf_chart = BytesIO()
        plt.savefig(buf_chart, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        chart_b64 = base64.b64encode(buf_chart.getvalue()).decode()

        # 图例说明
        chart_html = f"""
<div style='margin-bottom:12px;'>
  <img src='data:image/png;base64,{chart_b64}' style='width:100%;max-width:600px;'/>
  <div style='margin-top:6px;display:flex;gap:16px;font-size:11px;color:#666;'>
    <span><span style='display:inline-block;width:12px;height:12px;background:#C5DAFA;margin-right:4px;border-radius:2px;'></span>达标词汇</span>
    <span><span style='display:inline-block;width:12px;height:12px;background:#4CAF50;margin-right:4px;border-radius:2px;'></span>超纲词汇</span>
  </div>
</div>"""

        # 表格部分：只显示目标等级及以上的词汇
        table_results = [item for item in analysis_results if item['weight'] >= target_lvl_num]
        table_results = sorted(table_results, key=lambda x: x['weight'], reverse=True)

        if not table_results:
            table_html = "<div style='color:#999;font-size:12px;padding:8px;'>当前等级及以上暂无匹配词汇。</div>"
        else:
            table_html = "<table style='width:100%;border-collapse:collapse;font-size:12px;'>"
            table_html += "<tr style='background:#C5DAFA;'><th style='padding:6px 8px;color:#1D5FA5;'>词汇（拼音）</th><th style='padding:6px 8px;color:#1D5FA5;'>等级</th><th style='padding:6px 8px;color:#1D5FA5;'>词性</th><th style='padding:6px 8px;color:#1D5FA5;'>评估</th></tr>"
            for item in table_results:
                html_row = f"<tr style='border-bottom:0.5px solid #E8F1FB;'>"
                html_row += f"<td style='padding:6px 8px;'><b>{item['word']}</b><br><small style='color:#888;'>{item['pinyin']}</small></td>"
                html_row += f"<td style='padding:6px 8px;color:#1D5FA5;'>{item['level']}</td>"
                html_row += f"<td style='padding:6px 8px;color:#4A90E2;'>{item['pos']}</td>"
                html_row += f"<td style='padding:6px 8px;'>{'⭐ 超纲表达' if item['star'] else '运用达标'}</td></tr>"
                table_html += html_row
            table_html += "</table>"

        return chart_html + table_html

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

# ---------- 翻译函数 ----------

import hashlib
import random

def translate_text(text, to_lang='en'):
    """调用百度翻译API，将中文翻译成目标语言"""
    if not text or not text.strip():
        return text
    
    url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    salt = str(random.randint(32768, 65536))
    sign_str = BAIDU_APP_ID + text + salt + BAIDU_SECRET_KEY
    sign = hashlib.md5(sign_str.encode('utf-8')).hexdigest()
    
    params = {
        'q': text,
        'from': 'zh',
        'to': to_lang,
        'appid': BAIDU_APP_ID,
        'salt': salt,
        'sign': sign
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        result = resp.json()
        if 'trans_result' in result:
            return '\n'.join([item['dst'] for item in result['trans_result']])
        else:
            print("翻译错误:", result)
            return text
    except Exception as e:
        print("翻译异常:", e)
        return text
# ---------- 5. 核心反馈逻辑 ----------

def get_feedback(title, essay, hsk_level, focus_areas):
    print(f">>> 正在执行写作诊断: 《{title}》 [{hsk_level}]")
    try:
        # RAG检索
        # 过滤关键词集合:单字高频虚词直接过滤，保留有语法意义的功能词
        # 按句子切分作文（只在句号、问号、感叹号处切分）
        import re
        sentences = re.split(r'(?<=[。！？])', essay)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]

        # 对每个句子检索最相似的语料，找到就配对，找不到就跳过
        similar_cases = []
        seen_ids = set()

        for sent in sentences:
            sent_vector = embed_model.encode([sent]).astype('float32')
            faiss.normalize_L2(sent_vector)
            sent_scores, sent_indices = faiss_index.search(sent_vector, 5)

            best_match = None
            for score, idx in zip(sent_scores[0], sent_indices[0]):
                if score >= 0.65 and idx not in seen_ids:
                    meta = bcc_metadata[idx]
                    keyword = str(meta.get("keyword", "")).strip()
                    display_text = meta.get("sentence_display") or meta.get("sentence_clean", "") or meta.get("sentence", "")
                    if display_text.strip():
                        best_match = {
                            "student_sentence": sent,
                            "text": display_text,
                            "keyword": keyword,
                            "is_error": meta["is_error"],
                            "score": round(float(score), 3)
                        }
                        seen_ids.add(idx)
                        break

            if best_match:
                similar_cases.append(best_match)

            if len(similar_cases) >= 4:
                break

        active_focus = "、".join(focus_areas) if focus_areas else "综合写作能力"
        if similar_cases:
            corpus_lines = []
            for c in similar_cases:
                usage = "错误用法" if str(c['is_error']) == 'True' else "正确用法"
                corpus_lines.append(
                    f"- 学生原句:「{c['student_sentence']}」\n"
                    f"  语料库对应示例（{usage}，关键词:{c['keyword']}）:「{c['text']}」\n"
                    f"  提示:{'这是一个错误用法，请告诉学生如何修改' if str(c['is_error']) == 'True' else '这是正确用法，学生可以模仿这个句子的写法'}"
                )
            corpus_text = "\n".join(corpus_lines)
        else:
            corpus_text = "暂无高相关语料，请仅依据评分标准对作文进行反馈。"

        # 构建分级评分标准
        rubric_text = build_rubric_prompt(hsk_level)

        # 构建五个维度的评分要求
        dims = ["内容与切题", "篇章结构", "语篇连贯性", "词汇运用", "语法准确性"]
        focus_note = f"本次反馈重点关注【{active_focus}】，在该维度请给出更详细的分析与建议。"

        # 第一步:严格评分（temperature=0，不含反馈重点）
        scoring_prompt = f"""你是一名专业的国际中文教育评估专家。
请严格依据以下评分标准对该{hsk_level}级作文进行评分。

{rubric_text}

【学生作文题目】:{title}
【学生原文】:{essay}

【注意】评分阶段只依据量表标准和学生原文判断，不参考任何语料库内容。

【评分步骤，必须按顺序执行】
第一步:逐维度严格对照描述语，列出该维度发现的具体错误或不足，每条错误必须引用原文中的具体句子或词语。
第二步:根据错误数量和严重程度，对照描述语确定初步分数。
第三步:自我验证——重新阅读该维度的描述语，确认初步分数对应的描述语与作文实际表现吻合，如不吻合则修正。
第四步:输出最终分数。

【重要评分规则】
1. 评分分布:大部分作文得分应在2-4分之间，少部分作文可得1分或5分，请勿集中给出高分。
2. 词汇运用维度:必须依据评分标准中的HSK等级词汇要求判断，不得依赖模型自身的语感经验。
3. 语法准确性维度:若语法错误极少（0-1处）但标点错误较多，在语法分值基础上适当上调1分。
4. 汉字书写错误不在本次评分范围内，请忽略。
5. 每个维度必须先列出具体错误，再给出1-5分的整数评分。
6. 语法准确性维度:不能因为句子意思可以理解就忽略语法错误，必须检查:句式结构是否正确（如把字句、被字句是否误用）、动词与宾语搭配是否恰当、语序是否符合汉语规范，即使句意可懂也必须标注语法问题。

请严格以JSON格式输出，不得包含任何额外文字:
{{
  "scoring_process": {{
    "内容与切题": {{"errors": "<列出具体错误或不足，引用原文>", "initial_score": <初步分数>, "verification": "<验证说明>", "final_score": <最终分数>}},
    "篇章结构": {{"errors": "<列出具体错误或不足，引用原文>", "initial_score": <初步分数>, "verification": "<验证说明>", "final_score": <最终分数>}},
    "语篇连贯性": {{"errors": "<列出具体错误或不足，引用原文>", "initial_score": <初步分数>, "verification": "<验证说明>", "final_score": <最终分数>}},
    "词汇运用": {{"errors": "<只列出有明确语言学依据的词汇错误，包括：动宾搭配不当、近义词混用、褒贬误用、量词错误、书面语口语混用。每个错误词语必须用「」包裹并说明原因，例如：「吃」与牛奶搭配不当应改为喝。不要标注重复使用或词汇单一等丰富度问题。如无明确错误请填写无。>", "initial_score": <初步分数>, "verification": "<验证说明>", "final_score": <最终分数>}},
    "语法准确性": {{"errors": "<列出具体语法错误，每个错误句子必须用「」包裹，例如：「我很努力学习汉语」语序有误>", "initial_score": <初步分数>, "verification": "<验证说明>", "final_score": <最终分数>}}
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
        if not data_score.get('scores'):
            raise ValueError("评分结果格式异常")

        # 评分结果拿到之后，再构建feedback_prompt
        score_summary = "\n".join([
            f"{dim}:{int(data_score.get('scores', {}).get(dim, 3))}分"
            for dim in ["内容与切题", "篇章结构", "语篇连贯性", "词汇运用", "语法准确性"]
        ])

        feedback_prompt = f"""你是一名专业的国际中文教育评估专家。
请依据以下评分标准和已确定的评分结果，对该{hsk_level}级作文的各维度给出具体反馈建议。

{rubric_text}

【已确定的评分结果】（反馈内容必须与此分数保持一致，不得矛盾）:
{score_summary}

【学生作文题目】:{title}
【学生原文】:{essay}
【语料库参考示例】:
{corpus_text}
【本次反馈重点】:{active_focus}（请在该维度给出更详细的分析与建议）

请严格以JSON格式输出:
{{
  "feedback": {{
    "内容与切题": "<依据已确定的评分分数对应的量表描述语，指出学生作文在内容与切题维度的具体表现，必须引用原文中至少一处具体句子或词语作为证据，说明符合或不符合描述语的理由，再给出针对性的改进建议。最后一句格式为：本维度得X分，依据量表X分标准：【引用对应分数的描述语】，作文的实际表现为……>",
    "篇章结构": "<依据已确定的评分分数对应的量表描述语，指出学生作文在篇章结构维度的具体表现，必须引用原文中至少一处具体句子或段落作为证据，说明符合或不符合描述语的理由，再给出针对性的改进建议。最后一句格式为：本维度得X分，依据量表X分标准：【引用对应分数的描述语】，作文的实际表现为……>",
    "语篇连贯性": "<依据已确定的评分分数对应的量表描述语，指出学生作文在语篇连贯性维度的具体表现，必须引用原文中至少一处具体的衔接词或句子转换处作为证据，说明符合或不符合描述语的理由，再给出针对性的改进建议。最后一句格式为：本维度得X分，依据量表X分标准：【引用对应分数的描述语】，作文的实际表现为……>",
    "词汇运用": "<依据已确定的评分分数对应的量表描述语，指出学生作文在词汇运用维度的具体表现，必须引用原文中至少一处具体词语作为证据，结合HSK等级词汇要求说明符合或不符合描述语的理由，再给出针对性的改进建议。最后一句格式为：本维度得X分，依据量表X分标准：【引用对应分数的描述语】，作文的实际表现为……>",
    "语法准确性": "<依据已确定的评分分数对应的量表描述语，指出学生作文在语法准确性维度的具体表现，必须引用原文中至少一处具体的语法错误作为证据，说明符合或不符合描述语的理由，再给出针对性的改进建议。最后一句格式为：本维度得X分，依据量表X分标准：【引用对应分数的描述语】，作文的实际表现为……>",
    "grammar_examples": "<针对上述语法错误，每个错误生成一个示范句，示范句必须基于学生原句改写，保留原句话题和大意，只修正语法问题。格式为：原句：【引用学生原句】→示范句：【修改后的正确句子】。每个示范句之间用换行分隔。如无语法错误则填写无。>"
  }},
  "strengths": "<文章整体亮点与情感鼓励，必须包含以下三个层次：第一，真诚指出文章中具体用得好的词语或句子（引用原文）；第二，肯定学生的努力和进步，用温暖的语气表达认可；第三，用一句激励性的话鼓励学生继续写作，让学生感受到学习中文的成就感和动力。语气温暖自然，避免套话，像一位真正关心学生的老师。>",
  "opening": "<你是一位真正在阅读这篇作文的国际中文教师，不是评分机器。请按照以下四个部分生成情感反馈，每个部分2-3句话，总字数控制在170-230字之间，语气真诚自然，绝对不能出现'你很棒''继续加油''写得不错'这类空洞套话，要像一个真实的人在回应学生的表达。第一部分，看见内容：从学生作文中提取一个具体的细节、场景或表达，告诉学生你读到了什么，让他感受到有人真的在读他的文章，而不是在走流程。第二部分，看见努力：观察学生在词汇选择、句式尝试或结构安排上做了哪些努力，哪怕结果不完美，也要肯定这种尝试本身，因为学习者最希望被看见的是努力而不是成绩。第三部分，降低失败感：用温和的方式提到作文中存在的不足，把问题正常化，告诉学生这是语言发展过程中自然出现的现象，让学生读完之后愿意继续写下去，而不是感到受挫。第四部分，给予成长期待：用面向未来的语言结尾，告诉学生下一步可以朝哪个方向成长，语气是充满信心的期待，而不是命令或要求。>",
  "suggestions": "<改进建议，语气要积极正面，不要直接说'你错了'，而是说'如果能做到X，你的作文会更好'。先肯定学生的努力，再给出1-2个最重要的改进方向，最后用一句充满信心的话结尾，比如'相信你下次一定能写得更好'之类的鼓励。>",
  "corrections": [
    {{
      "original": "<学生原句，直接引用>",
      "error_type": "<错误类型，从以下选择:语法错误/词汇建议/语篇问题/标点问题>",
      "problem": "<具体问题说明，1-2句>",
      "suggestion": "<修改后的句子>",
      "corpus_ref": "<如有语料库参考则填入对应语料句子，没有则填空字符串>",
      "corpus_is_error": <true或false，对应语料是错误用法还是正确用法>,
      "corpus_keyword": "<对应语料的关键词，没有则填空字符串>",
      "corpus_tip": "<基于语料给学生的一句提示，没有语料则填空字符串>"
    }}
  ]
}}"""

        payload_feedback = {
            "model": "deepseek-chat",
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": "你是一个严格依据量表标准输出JSON的国际中文写作评估专家。每个维度的反馈必须严格对照已确定的评分分数所对应的量表描述语撰写，不得依赖模型自身经验判断，必须引用原文证据，最后必须用规定格式说明评分理由。"},
                {"role": "user", "content": feedback_prompt}
            ],
            "response_format": {"type": "json_object"}
        }

        resp_feedback = requests.post("https://api.deepseek.com/chat/completions", json=payload_feedback, headers=headers, timeout=90)
        try:
            raw_content = resp_feedback.json()['choices'][0]['message']['content']
            raw_content = raw_content.strip()
            if raw_content.startswith('```'):
                raw_content = raw_content.split('```')[1]
                if raw_content.startswith('json'):
                    raw_content = raw_content[4:]
            data = json.loads(raw_content)
        except Exception:
            data = {
                'feedback': {dim: '系统繁忙，请重新提交。' for dim in ["内容与切题","篇章结构","语篇连贯性","词汇运用","语法准确性"]},
                'strengths': '系统繁忙，请重新提交。',
                'opening': '系统繁忙，请重新提交。',
                'suggestions': '系统繁忙，请重新提交。',
                'corrections': [],
                'grammar_examples': '无'
            }
        print("feedback keys:", list(data.keys()))
        print("feedback content:", data.get('feedback', {}))
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
        else:
            total_label = "继续加油"

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

        feedback = data.get('feedback', {})
        dims_display = ["内容与切题", "篇章结构", "语篇连贯性", "词汇运用", "语法准确性"]

        feedback_html_parts = ""
        for i, dim in enumerate(dims_display):
            is_focus = dim in focus_areas
            focus_tag = "<span style='font-size:9px;background:#1D5FA5;color:white;padding:1px 6px;border-radius:3px;margin-left:6px;'>⭐ 重点</span>" if is_focus else ""
            dim_score = dim_scores.get(dim, 3)
            dim_content = feedback.get(dim, '（暂无）')
            uid_dim = f"dim-{i}-{int(__import__('time').time()*1000)%100000}"
            
            grammar_example_html = ""
            if dim == "语法准确性":
                grammar_examples = feedback.get('grammar_examples', '')
                if grammar_examples and grammar_examples != '无':
                    lines = grammar_examples.replace('→示范句：', '\n示范句：').split('\n')
                    formatted = ""
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith('原句：'):
                            formatted += f"<div style='margin-bottom:2px;direction:ltr;text-align:left;'><span style='color:#E24B4A;font-weight:600;'>原句：</span>{line[3:]}</div>"
                        elif line.startswith('示范句：'):
                            formatted += f"<div style='margin-bottom:8px;direction:ltr;text-align:left;'><span style='color:#1D9E75;font-weight:600;'>示范句：</span>{line[4:]}</div>"
                        else:
                            formatted += f"<div style='margin-bottom:4px;'>{line}</div>"
                    grammar_example_html = f"""
<div style='background:#F0F6FF;border:0.5px solid #85B7EB;border-radius:6px;padding:10px 12px;margin-top:8px;direction:ltr;text-align:left;'>
  <div style='font-size:11px;font-weight:500;color:#1D5FA5;margin-bottom:8px;'>📝 参考示范句</div>
  <div style='font-size:11px;color:#2A4A7F;line-height:1.8;'>{formatted}</div>
  <div style='font-size:10px;color:#9BB5D0;margin-top:6px;'>以上例句仅供参考，鼓励你根据自己的理解尝试改写。</div>
</div>"""

            btn_bg = "#E6F1FB" if is_focus else "#F8FBFF"
            btn_border = "#85B7EB" if is_focus else "#C5DAFA"
            feedback_html_parts += f"""
<div style='margin-bottom:6px;'>
  <div onclick="
    var c=document.getElementById('{uid_dim}');
    var shown=c.style.display!=='none';
    c.style.display=shown?'none':'block';
    this.querySelector('.arr').textContent=shown?'▾':'▴';"
    style='width:100%;background:{btn_bg};border:0.5px solid {btn_border};border-radius:8px;padding:10px 14px;display:flex;align-items:center;justify-content:space-between;cursor:pointer;'>
    <div style='display:flex;align-items:center;gap:6px;'>
      <span style='font-size:12px;color:#5F7FA5;'>{'①②③④⑤'[i]}</span>
      <span style='font-size:12px;font-weight:500;color:#0C447C;'>{dim}</span>
      {focus_tag}
    </div>
    <div style='display:flex;align-items:center;gap:8px;'>
      <span style='font-size:12px;font-weight:500;color:#0C447C;'>{dim_score}/5</span>
      <div style='width:36px;height:3px;background:#E6F1FB;border-radius:2px;'>
        <div style='height:3px;background:#378ADD;border-radius:2px;width:{dim_score*20}%;'></div>
      </div>
      <span class='arr' style='font-size:10px;color:#5F7FA5;'>▾</span>
    </div>
  </div>
  <div id='{uid_dim}' style='display:none;background:#F5F9FF;border:0.5px solid #C5DAFA;border-radius:0 0 8px 8px;padding:12px 14px;border-top:none;font-size:11px;color:#444;line-height:1.7;'>
    {dim_content}
    {grammar_example_html}
  </div>
</div>"""

        strengths_text = data.get('strengths', '') or '（暂无）'
        opening_text = data.get('opening', '') or '（暂无）'
        suggestions_text = data.get('suggestions', '') or '（暂无）'

        import base64
        fig2, ax2 = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
        angles2 = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
        vals = [float(s) for s in current_scores]
        vals2 = vals + [vals[0]]
        angles2 += angles2[:1]
        ax2.fill(angles2, vals2, color='#4A90E2', alpha=0.3)
        ax2.plot(angles2, vals2, color='#378ADD', linewidth=1.5, marker='o', markersize=4)
        ax2.set_ylim(0, 5)
        ax2.set_yticks([1, 2, 3, 4, 5])
        ax2.set_yticklabels(['1','2','3','4','5'], size=7)
        ax2.set_xticks(angles2[:-1])
        ax2.set_xticklabels(['内容切题','篇章结构','语篇连贯','词汇运用','语法准确'], size=7)
        buf2 = BytesIO()
        plt.savefig(buf2, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        img_b64 = base64.b64encode(buf2.getvalue()).decode()

        # 生成词汇柱形图
        try:
            level_name = {1:"一级",2:"二级",3:"三级",4:"四级",5:"五级",6:"六级",8:"高等"}
            vocab_html_result = analyze_hsk_level(essay, hsk_level)
            # 从standard_html里提取柱形图部分（img标签）
            import re as re2
            chart_match = re2.search(r"<img src='data:image/png;base64,([^']+)'[^>]*/>", vocab_html_result)
            if chart_match:
                vocab_chart_section = f"<img src='data:image/png;base64,{chart_match.group(1)}' style='width:100%;'/>"
                vocab_chart_section += """
<div style='display:flex;gap:12px;margin-top:6px;font-size:10px;color:#5F7FA5;'>
  <span><span style='display:inline-block;width:10px;height:10px;background:#C5DAFA;border-radius:2px;margin-right:3px;'></span>达标词汇</span>
  <span><span style='display:inline-block;width:10px;height:10px;background:#4CAF50;border-radius:2px;margin-right:3px;'></span>超纲词汇</span>
</div>"""
            else:
                vocab_chart_section = "<div style='color:#B0C4DE;font-size:12px;text-align:center;padding:16px;'>暂无词汇数据</div>"
        except:
            vocab_chart_section = "<div style='color:#B0C4DE;font-size:12px;text-align:center;padding:16px;'>暂无词汇数据</div>"
        score_html = f"""
<div style='display:flex;flex-direction:column;gap:12px;'>
  <div style='background:white;border:0.5px solid #C5DAFA;border-radius:12px;padding:16px;'>
    <div style='font-size:10px;font-weight:500;color:#5F7FA5;margin-bottom:10px;letter-spacing:0.03em;'>评分概览</div>
    <div style='display:flex;gap:14px;align-items:center;'>
      <div style='flex:1.1;'>
        <img src='data:image/png;base64,{img_b64}' style='width:100%;'/>
      </div>
      <div style='flex:1;display:flex;flex-direction:column;justify-content:center;gap:0;padding-left:8px;'>
        {"".join([
          f"<div style='display:flex;align-items:center;padding:7px 0;border-bottom:0.5px solid #E6F1FB;'>"
          f"<span style='font-size:11px;color:#444;width:70px;flex-shrink:0;'>{d}</span>"
          f"<div style='flex:1;margin:0 8px;height:3px;background:#E6F1FB;border-radius:2px;'><div style='height:3px;background:#378ADD;border-radius:2px;width:{int(current_scores[i])*20}%;'></div></div>"
          f"<span style='font-size:11px;color:#0C447C;font-weight:500;width:24px;text-align:right;'>{int(current_scores[i])}/5</span></div>"
          for i, d in enumerate(["内容与切题","篇章结构","语篇连贯性","词汇运用","语法准确性"])
        ])}
        <div style='background:#E6F1FB;border-radius:6px;padding:7px 10px;display:flex;justify-content:space-between;align-items:center;margin-top:8px;'>
          <span style='font-size:11px;color:#0C447C;font-weight:500;'>总分</span>
          <span style='font-size:12px;color:#0C447C;font-weight:500;'>{int(total)}/25 {total_label}</span>
        </div>
      </div>
    </div>
  </div>
  <div style='background:white;border:0.5px solid #C5DAFA;border-radius:12px;padding:16px;'>
    <div style='font-size:10px;font-weight:500;color:#5F7FA5;margin-bottom:8px;letter-spacing:0.03em;'>词汇等级分布</div>
    {vocab_chart_section}
  </div>
</div>"""
        corrections = data.get('corrections', [])
        if similar_cases:
            ref_html = "<div style='padding:0.5rem 0;font-family:var(--font-sans);'>"
            ref_html += "<div style='font-size:10px;color:#9BB5D0;margin-bottom:10px;'>以下为语料库中与您作文相似的句子，这些表达在留学生写作中也曾出现过，仅供参考。</div>"
            for c in similar_cases:
                student_sent = c.get('student_sentence', '')
                corpus_sent = c.get('text', '')
                ref_html += f"""
<div style='background:white;border:0.5px solid #C5DAFA;border-radius:10px;padding:12px 14px;margin-bottom:10px;'>
  <div style='display:flex;align-items:flex-start;gap:8px;margin-bottom:8px;'>
    <span style='background:#FFEBEE;color:#B71C1C;font-size:10px;font-weight:500;padding:2px 8px;border-radius:4px;white-space:nowrap;margin-top:1px;'>学生原句</span>
    <span style='font-size:12px;color:#2A4A7F;line-height:1.6;'>{student_sent}</span>
  </div>
  <div style='border-left:2px solid #E6F1FB;padding-left:10px;'>
    <div style='font-size:10px;color:#5F7FA5;margin-bottom:4px;'>语料库中有人写过类似的句子：</div>
    <div style='font-size:11px;color:#2A4A7F;line-height:1.6;font-style:italic;'>「{corpus_sent}」</div>
    <div style='font-size:10px;color:#9BB5D0;margin-top:6px;'>💡 类似的表达在留学生写作中也曾出现过，建议结合教师反馈判断是否需要修改。</div>
  </div>
</div>"""
            ref_html += "</div>"
        else:
            ref_html = "<div style='padding:1rem;color:#999;font-size:13px;'>暂无高相关语料参考</div>"

        standard_html = analyze_hsk_level(essay, hsk_level)
        # 生成HighlightedText标注数据
        # 第一步：从scoring_process提取黄色和红色标注
        scoring_process = data_score.get('scoring_process', {})
        
        yellow_words = []  # 词汇问题
        red_sentences = []  # 语法问题
        
        vocab_errors = scoring_process.get('词汇运用', {}).get('errors', '')
        grammar_errors = scoring_process.get('语法准确性', {}).get('errors', '')
        
        import re
        if vocab_errors and vocab_errors != '无':
            yellow_words = re.findall(r'[「""](.+?)[」""]', vocab_errors)
            if not yellow_words:
                yellow_words = re.findall(r'"(.+?)"', vocab_errors)
            if not yellow_words:
                yellow_words = re.findall(r'【(.+?)】', vocab_errors)
            yellow_words = [w for w in yellow_words if len(w) <= 6]

        if grammar_errors and grammar_errors != '无':
            red_sentences = re.findall(r'[「""](.+?)[」""]', grammar_errors)
            if not red_sentences:
                red_sentences = re.findall(r'"(.+?)"', grammar_errors)
            if not red_sentences:
                red_sentences = re.findall(r'【(.+?)】', grammar_errors)
            red_sentences = [s for s in red_sentences if len(s) > 4]
        
        # 第二步：从hsk_vocab提取超纲词汇（绿色）
        green_words = []
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
            vocab_path = os.path.join(base_path, "hsk_vocab.csv")
            df_vocab = pd.read_csv(vocab_path, encoding='utf-8-sig')
            df_vocab = df_vocab.map(lambda x: str(x).strip() if isinstance(x, str) else x)
            col_word = next((c for c in df_vocab.columns if '词' in c), None)
            col_level = next((c for c in df_vocab.columns if '级' in c), None)
            level_weight = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"高":8}
            target_lvl_num = int(''.join(filter(str.isdigit, hsk_level))) if any(c.isdigit() for c in hsk_level) else 0
            word_mapping = {}
            for _, row in df_vocab.iterrows():
                w = row[col_word]
                lv = row[col_level]
                cur_lvl_num = next((v for k, v in level_weight.items() if k in str(lv)), 0)
                word_mapping[w] = cur_lvl_num
            # 用jieba找超纲词汇，但只记录词语本身，不用分词结果拼接原文
            words = jieba.lcut(essay)
            seen = set()
            for w in words:
                if w in word_mapping and word_mapping[w] > target_lvl_num and w not in seen:
                    green_words.append(w)
                    seen.add(w)
        except:
            green_words = []

        # 第三步：在原文字符串上用字符位置构建标注，完全不改变原文结构
        annotations = []

        # 绿色：超纲词汇
        for w in green_words:
            start = 0
            while True:
                pos = essay.find(w, start)
                if pos == -1:
                    break
                annotations.append((pos, pos + len(w), '超纲词汇'))
                start = pos + len(w)

        # 黄色：词汇问题
        for w in set(yellow_words):
            if len(w) == 0:
                continue
            start = 0
            while True:
                pos = essay.find(w, start)
                if pos == -1:
                    break
                annotations.append((pos, pos + len(w), '词汇问题'))
                start = pos + len(w)

        # 红色：语法问题
        for s in set(red_sentences):
            if len(s) == 0:
                continue
            pos = essay.find(s)
            if pos != -1:
                annotations.append((pos, pos + len(s), '语法问题'))

        # 按位置排序，去除重叠（优先级：语法问题>词汇问题>超纲词汇）
        priority = {'语法问题': 3, '词汇问题': 2, '超纲词汇': 1}
        annotations.sort(key=lambda x: (x[0], -priority[x[2]]))

        merged = []
        last_end = 0
        for start, end, label in annotations:
            if start >= last_end:
                merged.append((start, end, label))
                last_end = end

        # 按字符位置切分原文，未标注部分原样保留
        highlighted_data = []
        last_end = 0
        for start, end, label in merged:
            if start > last_end:
                highlighted_data.append((essay[last_end:start], None))
            highlighted_data.append((essay[start:end], label))
            last_end = end
        if last_end < len(essay):
            highlighted_data.append((essay[last_end:], None))

        if not highlighted_data:
            highlighted_data = [(essay, None)]
        zh_content = {
            'strengths': strengths_text,
            'opening': opening_text,
            'suggestions': suggestions_text,
            'feedback_html': feedback_html_parts,
            'ref_html': ref_html,
            'standard_html': standard_html,
            'feedback_dims': {dim: feedback.get(dim, '') for dim in dims_display},
            'grammar_examples': feedback.get('grammar_examples', ''),
            'focus_areas': focus_areas,
            'dim_scores': dim_scores,
        }
        
        return strengths_text, opening_text, suggestions_text, feedback_html_parts, ref_html, standard_html, score_html, highlighted_data, zh_content

    except Exception as e:
        import traceback
        traceback.print_exc()

        return '生成失败', '', '', '', '<div>生成失败</div>', '<div>生成失败</div>', '', [(essay, None)], {}

# ---------- 6. Gradio界面 ----------

css = """
.gradio-container { margin: 0 auto !important; background: #F0F6FF !important; }
.gr-row { align-items: flex-start !important; }
.gradio-row { align-items: flex-start !important; }
#annotation-fixed { min-height: 200px; max-height: 400px; overflow-y: auto; }

/* 左侧输入区整体 */
.input-col { 
    background: white !important; 
    border-radius: 12px !important; 
    padding: 0 !important;
    border: 0.5px solid #C5DAFA !important;
    overflow: hidden !important;
    align-self: flex-start !important;
}

/* 覆盖Gradio输入框默认样式 */
.input-col input, .input-col textarea {
    border: 0.5px solid #C5DAFA !important;
    border-radius: 8px !important;
    background: #F8FBFF !important;
    font-size: 13px !important;
}
.input-col input:focus, .input-col textarea:focus {
    border-color: #378ADD !important;
    box-shadow: 0 0 0 2px #E6F1FB !important;
}
.input-col select, .input-col .wrap {
    border: 0.5px solid #C5DAFA !important;
    border-radius: 8px !important;
    background: #F8FBFF !important;
}
.input-col label span {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #5F7FA5 !important;
}
.input-col .gr-button-primary {
    background: #1D5FA5 !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
.input-col .gr-button-primary:hover {
    background: #185FA5 !important;
    opacity: 0.92 !important;
}
/* HighlightedText样式 */
.input-col .highlighted-text {
    font-size: 12px !important;
    line-height: 1.8 !important;
    border: 0.5px solid #C5DAFA !important;
    border-radius: 8px !important;
    background: #F8FBFF !important;
    padding: 8px !important;
}
#lang-zh, #lang-en, #lang-ru, #lang-ar {
    min-width: 0 !important;
    flex: 1 !important;
}
#lang-zh button, #lang-en button, #lang-ru button, #lang-ar button {
    border-radius: 20px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    border: 0.5px solid #C5DAFA !important;
    background: white !important;
    color: #5F7FA5 !important;
    padding: 5px 0 !important;
    min-width: 0 !important;
}
#lang-zh button {
    background: #E6F1FB !important;
    color: #0C447C !important;
    border-color: #85B7EB !important;
}
"""

EMPTY_RIGHT_HTML = """
<div style='font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;display:flex;flex-direction:column;gap:12px;'>
  <div style='background:white;border:0.5px solid #C5DAFA;border-radius:12px;padding:16px;'>
    <div style='font-size:10px;font-weight:500;color:#5F7FA5;margin-bottom:10px;letter-spacing:0.03em;'>评分概览</div>
    <div style='display:flex;align-items:center;justify-content:center;height:160px;color:#B0C4DE;font-size:12px;'>提交作文后显示评分</div>
  </div>
  <div style='background:white;border:0.5px solid #C5DAFA;border-radius:12px;padding:16px;'>
    <div style='font-size:10px;font-weight:500;color:#5F7FA5;margin-bottom:8px;'>词汇等级分布</div>
    <div style='display:flex;align-items:center;justify-content:center;height:80px;color:#B0C4DE;font-size:12px;'>提交作文后显示</div>
  </div>
</div>"""

def build_translated_html(strengths, opening, suggestions, feedback_html, ref_html, standard_html, score_html, lang, zh_content=None):
    if lang == 'en':
        lang_code = 'en'
    elif lang == 'ru':
        lang_code = 'ru'
    elif lang == 'ar':
        lang_code = 'ara'
    else:
        lang_code = 'zh'
    t_strengths = translate_text(strengths, lang_code)
    t_opening = translate_text(opening, lang_code)
    t_suggestions = translate_text(suggestions, lang_code)

    # 翻译各维度反馈
    t_feedback_html = feedback_html
    if zh_content and zh_content.get('feedback_dims'):
        dims_display = ["内容与切题", "篇章结构", "语篇连贯性", "词汇运用", "语法准确性"]
        focus_areas = zh_content.get('focus_areas', [])
        t_feedback_html = ""
        for i, dim in enumerate(dims_display):
            is_focus = dim in focus_areas
            focus_badge = f"<span style='font-size:10px;padding:1px 6px;background:#C5DAFA;color:#1D5FA5;border-radius:3px;margin-left:4px;'>{'重点反馈' if lang_code == 'zh' else 'Focus' if lang_code == 'en' else 'تركيز' if lang_code == 'ara' else 'Акцент'}</span>" if is_focus else ""
            prefix = "⭐ " if is_focus else ""
            dim_content = zh_content['feedback_dims'].get(dim, '')
            t_dim_content = translate_text(dim_content, lang_code)
            grammar_example_html = ""
            if dim == "语法准确性":
                grammar_examples = zh_content.get('grammar_examples', '')
                if grammar_examples and grammar_examples != '无':
                    lines = grammar_examples.replace('→示范句：', '\n示范句：').split('\n')
                    formatted = ""
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith('原句：'):
                            formatted += f"<div style='margin-bottom:2px;direction:ltr;text-align:left;'><span style='color:#E24B4A;font-weight:600;'>原句：</span>{line[3:]}</div>"
                        elif line.startswith('示范句：'):
                            formatted += f"<div style='margin-bottom:8px;direction:ltr;text-align:left;'><span style='color:#1D9E75;font-weight:600;'>示范句：</span>{line[4:]}</div>"
                        else:
                            formatted += f"<div style='margin-bottom:4px;'>{line}</div>"
                    if lang_code == 'en':
                        note = 'For reference only. Try rewriting in your own words.'
                        example_title = '📝 Reference Examples'
                    elif lang_code == 'ara':
                        note = 'للاستخدام المرجعي فقط. حاول إعادة الكتابة بأسلوبك الخاص.'
                        example_title = '📝 أمثلة مرجعية'
                    else:
                        note = 'Только для справки. Попробуйте переписать своими словами.'
                        example_title = '📝 Примеры для справки'
                    grammar_example_html = f"""
<div style='background:#F0F6FF;border:0.5px solid #85B7EB;border-radius:6px;padding:10px 12px;margin-top:8px;direction:ltr;text-align:left;'>
  <div style='font-size:11px;font-weight:500;color:#1D5FA5;margin-bottom:8px;'>{example_title}</div>
  <div style='font-size:11px;color:#2A4A7F;line-height:1.8;'>{formatted}</div>
  <div style='font-size:10px;color:#9BB5D0;margin-top:6px;'>{note}</div>
</div>"""
            is_focus = dim in zh_content.get('focus_areas', [])
            focus_tag = f"<span style='font-size:9px;background:#1D5FA5;color:white;padding:1px 6px;border-radius:3px;margin-left:6px;'>⭐ {'重点' if lang_code=='zh' else 'Focus' if lang_code=='en' else 'Акцент' if lang_code=='ru' else 'تركيز'}</span>" if is_focus else ""
            btn_bg = "#E6F1FB" if is_focus else "#F8FBFF"
            btn_border = "#85B7EB" if is_focus else "#C5DAFA"
            uid_dim = f"dim-{i}-{lang_code}-{int(__import__('time').time()*1000)%100000}"
            dim_score = zh_content.get('dim_scores', {}).get(dim, 3)
            t_feedback_html += f"""
<div style='margin-bottom:6px;'>
  <div onclick="
    var c=document.getElementById('{uid_dim}');
    var shown=c.style.display!=='none';
    c.style.display=shown?'none':'block';
    this.querySelector('.arr').textContent=shown?'▾':'▴';"
    style='width:100%;background:{btn_bg};border:0.5px solid {btn_border};border-radius:8px;padding:10px 14px;display:flex;align-items:center;justify-content:space-between;cursor:pointer;'>
    <div style='display:flex;align-items:center;gap:6px;'>
      <span style='font-size:12px;color:#5F7FA5;'>{'①②③④⑤'[i]}</span>
      <span style='font-size:12px;font-weight:500;color:#0C447C;'>{dim}</span>
      {focus_tag}
    </div>
    <div style='display:flex;align-items:center;gap:8px;'>
      <span style='font-size:12px;font-weight:500;color:#0C447C;'>{dim_score}/5</span>
      <div style='width:36px;height:3px;background:#E6F1FB;border-radius:2px;'>
        <div style='height:3px;background:#378ADD;border-radius:2px;width:{dim_score*20}%;'></div>
      </div>
      <span class='arr' style='font-size:10px;color:#5F7FA5;'>▾</span>
    </div>
  </div>
  <div id='{uid_dim}' style='display:none;background:#F5F9FF;border:0.5px solid #C5DAFA;border-radius:0 0 8px 8px;padding:12px 14px;border-top:none;font-size:11px;color:#444;line-height:1.7;'>
    {t_dim_content}
    {grammar_example_html}
  </div>
</div>"""

    labels = {
        'en': {
            'strengths': '✅ Strengths',
            'opening': '🌟 Emotional Feedback',
            'report_btn': 'View Report ▾',
            'collapse_btn': 'Collapse Report ▴',
            'corpus_btn': '📖 Corpus Examples ▾',
            'vocab_btn': '📊 Vocabulary Analysis ▾',
            'feedback_title': 'Detailed Feedback by Dimension',
            'suggestions_title': '🔍 Suggestions',
        },
        'ru': {
            'strengths': '✅ Достоинства текста',
            'opening': '🌟 Эмоциональный отклик',
            'report_btn': 'Просмотр отчёта ▾',
            'collapse_btn': 'Свернуть отчёт ▴',
            'corpus_btn': '📖 Примеры корпуса ▾',
            'vocab_btn': '📊 Анализ лексики ▾',
            'feedback_title': 'Подробная обратная связь',
            'suggestions_title': '🔍 Рекомендации',
        },
        'ar': {
            'strengths': '✅ نقاط قوة المقال',
            'opening': '🌟 الاستجابة العاطفية',
            'report_btn': 'عرض تقرير التقييم ▾',
            'collapse_btn': 'طي التقرير ▴',
            'corpus_btn': '📖 أمثلة المدونة اللغوية ▾',
            'vocab_btn': '📊 تحليل المفردات ▾',
            'feedback_title': 'تغذية راجعة تفصيلية',
            'suggestions_title': '🔍 اقتراحات التحسين',
        },
    }
    if lang == 'en':
        lang_code = 'en'
        lb = labels['en']
    elif lang == 'ru':
        lang_code = 'ru'
        lb = labels['ru']
    elif lang == 'ar':
        lang_code = 'ara'
        lb = labels['ar']
    else:
        lang_code = 'zh'
        lb = labels.get('zh', {})
    return build_report_html(t_strengths, t_opening, t_suggestions, t_feedback_html, ref_html, standard_html, score_html, lb)

def build_report_html(strengths, opening, suggestions, feedback_html, ref_html, standard_html, score_html, labels=None):
    import time, json
    uid = str(int(time.time() * 1000))[-6:]
    if labels is None:
        labels = {
            'strengths': '✅ 文章亮点',
            'opening': '🌟 情感回应',
            'report_btn': '查看评分报告 ▾',
            'collapse_btn': '收起评分报告 ▴',
            'corpus_btn': '📖 语料参考示例 ▾',
            'vocab_btn': '📊 词汇等级分析 ▾',
            'feedback_title': '各维度详细反馈',
            'suggestions_title': '🔍 改进建议',
        }
    
    print("labels corpus_btn:", labels.get('corpus_btn'))
    return f"""
<div style='font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;display:flex;flex-direction:column;gap:12px;{";direction:rtl;text-align:right;" if labels.get("feedback_title") == "تغذية راجعة تفصيلية" else ""}'>

  {score_html}

  <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
    <div style='background:white;border:0.5px solid #C5DAFA;border-radius:12px;padding:14px;'>
      <div style='font-size:11px;font-weight:500;color:#0C447C;margin-bottom:6px;'>{labels['strengths']}</div>
      <div style='font-size:12px;color:#444;line-height:1.7;'>{strengths or "（暂无）"}</div>
    </div>
    <div style='background:white;border:0.5px solid #C5DAFA;border-radius:12px;padding:14px;'>
      <div style='font-size:11px;font-weight:500;color:#0C447C;margin-bottom:6px;'>{labels['opening']}</div>
      <div style='font-size:12px;color:#444;line-height:1.7;'>{opening or "（暂无）"}</div>
    </div>
  </div>

  <button onclick="
    var r=document.getElementById('rpt-{uid}');
    var shown=r.style.display!=='none';
    r.style.display=shown?'none':'flex';
    this.textContent=shown?'{labels["report_btn"]}':'{labels["collapse_btn"]}';"
    style='background:#1D5FA5;color:white;border:none;border-radius:8px;padding:11px 0;font-size:13px;font-weight:500;cursor:pointer;width:100%;'>
    {labels['report_btn']}
  </button>

  <div id='rpt-{uid}' style='display:none;flex-direction:column;gap:10px;background:white;border:0.5px solid #C5DAFA;border-radius:12px;padding:14px;'>

    <div style='display:flex;gap:8px;'>
      <button onclick="
        var p=document.getElementById('corpus-{uid}');
        var shown=p.style.display!=='none';
        p.style.display=shown?'none':'block';
        this.style.background=shown?'#E6F1FB':'#B5D4F4';
        this.textContent=shown?'{labels['corpus_btn']}':'{labels['corpus_btn'].replace('▾','▴')}';"
        style='flex:1;background:#E6F1FB;color:#0C447C;border:0.5px solid #85B7EB;border-radius:8px;padding:8px 12px;font-size:11px;font-weight:500;cursor:pointer;text-align:left;'>
        {labels['corpus_btn']}
      </button>
      <button onclick="
        var p=document.getElementById('vocab-{uid}');
        var shown=p.style.display!=='none';
        p.style.display=shown?'none':'block';
        this.style.background=shown?'#E6F1FB':'#B5D4F4';
        this.textContent=shown?'{labels['vocab_btn']}':'{labels['vocab_btn'].replace('▾','▴')}';"
        style='flex:1;background:#E6F1FB;color:#0C447C;border:0.5px solid #85B7EB;border-radius:8px;padding:8px 12px;font-size:11px;font-weight:500;cursor:pointer;text-align:left;'>
        {labels['vocab_btn']}
      </button>
    </div>

    <div id='corpus-{uid}' style='display:none;'>{ref_html}</div>
    <div id='vocab-{uid}' style='display:none;font-size:11px;'>{standard_html}</div>

    <div style='font-size:10px;font-weight:500;color:#5F7FA5;letter-spacing:0.03em;'>各维度详细反馈</div>
    {feedback_html}

    <div style='margin-bottom:6px;'>
      <div onclick="
        var c=document.getElementById('sug-{uid}');
        var shown=c.style.display!=='none';
        c.style.display=shown?'none':'block';
        this.querySelector('.arr').textContent=shown?'▾':'▴';"
        style='width:100%;background:#E6F1FB;border:0.5px solid #85B7EB;border-radius:8px;padding:10px 14px;display:flex;align-items:center;justify-content:space-between;cursor:pointer;'>
        <span style='font-size:12px;font-weight:500;color:#0C447C;'>{labels['suggestions_title']}</span>
        <span class='arr' style='font-size:10px;color:#0C447C;'>▾</span>
      </div>
      <div id='sug-{uid}' style='display:none;background:#E6F1FB;border-left:2px solid #378ADD;border-radius:0 8px 8px 0;padding:12px 14px;border-top:none;font-size:11px;color:#185FA5;line-height:1.7;'>
        {suggestions or "（暂无）"}
      </div>
    </div>

  </div>
</div>"""

with gr.Blocks(theme=gr.themes.Soft(), title="IWriting", css=css) as demo:

    with gr.Row():
        gr.HTML("""
<div style='background:#1D5FA5;padding:16px 24px;border-radius:12px;margin-bottom:4px;'>
  <div style='color:white;font-size:18px;font-weight:500;'>IWriting 国际中文写作智能反馈系统</div>
  <div style='color:#B5D4F4;font-size:12px;margin-top:3px;'>International Chinese Writing Feedback System</div>
</div>""")

    with gr.Row(equal_height=False):
        with gr.Column(scale=2, elem_classes=["input-col"]):
            with gr.Column(elem_id="input-inner", scale=1):
                gr.HTML("<div style='padding:16px 16px 0;font-size:13px;font-weight:500;color:#1D5FA5;'>作文输入</div>")
                t_in = gr.Textbox(label="作文题目", placeholder="请输入题目...", container=True)
                e_in = gr.Textbox(label="正文内容", lines=10, placeholder="请输入留学生作文正文...", container=True)
                with gr.Row():
                    l_in = gr.Dropdown(["HSK3","HSK4","HSK5","HSK6"], label="目标HSK等级", value="HSK4")
                    f_in = gr.CheckboxGroup(
                        ["内容与切题","篇章结构","语篇连贯性","词汇运用","语法准确性"],
                        label="重点反馈维度", value=[]
                    )
                with gr.Accordion("📋 评分维度说明", open=False):
                    with gr.Tabs():
                        with gr.TabItem("中文"):
                            gr.HTML("""
<div style='display:flex;flex-direction:column;gap:6px;padding:8px 0;'>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>内容与切题</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>内容是否符合题目要求；能否表达基本事实与观点</div>
  </div>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>篇章结构</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>是否有完整的开头、主体、结尾；段落是否合理</div>
  </div>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>语篇连贯性</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>句子之间是否有基本的衔接；表达是否连贯流畅</div>
  </div>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>词汇运用</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>是否能正确使用目标等级词汇；选词是否准确多样</div>
  </div>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>语法准确性</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>是否能正确使用目标等级语法；语序是否正确；标点是否恰当</div>
  </div>
</div>""")
                        with gr.TabItem("English"):
                            gr.HTML("""
<div style='display:flex;flex-direction:column;gap:6px;padding:8px 0;'>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>Content & Relevance</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>Whether content meets the topic requirement and expresses basic facts and opinions</div>
  </div>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>Text Structure</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>Whether there is a complete introduction, body, and conclusion with logical paragraphs</div>
  </div>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>Discourse Coherence</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>Whether sentences are connected naturally and expression is fluent</div>
  </div>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>Vocabulary Use</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>Whether target-level vocabulary is used correctly with variety and accuracy</div>
  </div>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>Grammatical Accuracy</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>Whether target-level grammar is used correctly with proper word order and punctuation</div>
  </div>
</div>""")
                        with gr.TabItem("Русский"):
                            gr.HTML("""
<div style='display:flex;flex-direction:column;gap:6px;padding:8px 0;'>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>Содержание и тема</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>Соответствует ли содержание теме; выражены ли основные факты и мнения</div>
  </div>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>Структура текста</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>Есть ли полное введение, основная часть и заключение; логичны ли абзацы</div>
  </div>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>Связность текста</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>Связаны ли предложения естественно; является ли текст последовательным</div>
  </div>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>Использование лексики</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>Правильно ли используется лексика целевого уровня; разнообразна ли она</div>
  </div>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>Грамматическая точность</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>Правильно ли используется грамматика; верен ли порядок слов и пунктуация</div>
  </div>
</div>""")
                        with gr.TabItem("العربية"):
                            gr.HTML("""
<div style='display:flex;flex-direction:column;gap:6px;padding:8px 0;direction:rtl;text-align:right;'>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>المحتوى والموضوع</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>هل يتوافق المحتوى مع متطلبات الموضوع؟ هل يعبر عن الحقائق الأساسية والآراء؟</div>
  </div>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>بنية النص</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>هل يحتوي النص على مقدمة وعرض وخاتمة؟ هل تقسيم الفقرات منطقي؟</div>
  </div>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>تماسك النص</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>هل الجمل مترابطة بشكل طبيعي؟ هل التعبير متسق وسلس؟</div>
  </div>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>استخدام المفردات</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>هل يستخدم المفردات المناسبة للمستوى المستهدف؟ هل الاختيار اللغوي دقيق ومتنوع؟</div>
  </div>
  <div style='background:#F8FBFF;border:0.5px solid #E6F1FB;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:12px;font-weight:500;color:#0C447C;margin-bottom:3px;'>الدقة النحوية</div>
    <div style='font-size:10px;color:#5F7FA5;line-height:1.5;'>هل يستخدم القواعد النحوية للمستوى المستهدف بشكل صحيح؟ هل ترتيب الكلمات والترقيم مناسبان؟</div>
  </div>
</div>""")
                btn = gr.Button("提交诊断", variant="primary")
                gr.HTML("<div style='padding:0 16px 4px;font-size:11px;font-weight:500;color:#5F7FA5;'>作文标注分析</div><div style='padding:0 16px 6px;font-size:10px;color:#9BB5D0;'>以下标注由AI自动生成，仅供参考，具体以教师反馈为准</div>")
                annotation_out = gr.HighlightedText(
                    color_map={"超纲词汇":"#4CAF50","词汇问题":"#FFC107","语法问题":"#F44336"},
                    show_legend=True, visible=False, container=False,
                    elem_id="annotation-fixed"
                )
                gr.HTML("<div style='height:12px;'></div>")

        with gr.Column(scale=3):
            with gr.Row():
                lang_zh = gr.Button("中文", size="sm", scale=1, elem_id="lang-zh")
                lang_en = gr.Button("EN", size="sm", scale=1, elem_id="lang-en")
                lang_ru = gr.Button("RU", size="sm", scale=1, elem_id="lang-ru")
                lang_ar = gr.Button("AR", size="sm", scale=1, elem_id="lang-ar")
            combined_out = gr.HTML(value=EMPTY_RIGHT_HTML)

    zh_state = gr.State({})
    score_state = gr.State('')

    def process(t, e, l, f):
        results = get_feedback(t, e, l, f)
        zh_content = results[-1]
        score_html = results[-3]
        html = build_report_html(*results[:-2])
        highlighted = results[-2]
        return html, highlighted, gr.update(visible=True), zh_content, score_html

    def switch_to_en(zh_content, score_html):
        if not zh_content:
            return gr.update()
        return build_translated_html(
            zh_content.get('strengths', ''),
            zh_content.get('opening', ''),
            zh_content.get('suggestions', ''),
            zh_content.get('feedback_html', ''),
            zh_content.get('ref_html', ''),
            zh_content.get('standard_html', ''),
            score_html, 'en', zh_content
        )

    def switch_to_ru(zh_content, score_html):
        if not zh_content:
            return gr.update()
        return build_translated_html(
            zh_content.get('strengths', ''),
            zh_content.get('opening', ''),
            zh_content.get('suggestions', ''),
            zh_content.get('feedback_html', ''),
            zh_content.get('ref_html', ''),
            zh_content.get('standard_html', ''),
            score_html, 'ru', zh_content
        )

    def switch_to_zh(zh_content, score_html):
        if not zh_content:
            return gr.update()
        return build_report_html(
            zh_content.get('strengths', ''),
            zh_content.get('opening', ''),
            zh_content.get('suggestions', ''),
            zh_content.get('feedback_html', ''),
            zh_content.get('ref_html', ''),
            zh_content.get('standard_html', ''),
            score_html
        )
    
    def switch_to_ar(zh_content, score_html):
        if not zh_content:
            return gr.update()
        return build_translated_html(
            zh_content.get('strengths', ''),
            zh_content.get('opening', ''),
            zh_content.get('suggestions', ''),
            zh_content.get('feedback_html', ''),
            zh_content.get('ref_html', ''),
            zh_content.get('standard_html', ''),
            score_html, 'ar', zh_content
        )

    btn.click(
        fn=process,
        inputs=[t_in, e_in, l_in, f_in],
        outputs=[combined_out, annotation_out, annotation_out, zh_state, score_state]
    )

    lang_en.click(
        fn=switch_to_en,
        inputs=[zh_state, score_state],
        outputs=[combined_out]
    )

    lang_ru.click(
        fn=switch_to_ru,
        inputs=[zh_state, score_state],
        outputs=[combined_out]
    )

    lang_zh.click(
        fn=switch_to_zh,
        inputs=[zh_state, score_state],
        outputs=[combined_out]
    )

    lang_ar.click(
        fn=switch_to_ar,
        inputs=[zh_state, score_state],
        outputs=[combined_out]
    )

if __name__ == "__main__":
    print(">>> 正在启动，请关注下方链接...")
    demo.launch(share=False)
