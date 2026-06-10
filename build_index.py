import pandas as pd
import numpy as np
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer

CSV_FILE = "bcc_corpus_clean.csv"
INDEX_FILE = "bcc_faiss.index"
META_FILE = "bcc_metadata.pkl"

def clean_sentence(text):
    """清洗句子，去除人工标注痕迹"""
    if not isinstance(text, str):
        return ""
    # 去除例编号，如"例1：""例12："
    text = re.sub(r'例\d+：', '', text)
    # 去除【】括号及其内容（人工标注的关键词）
    text = re.sub(r'【.*?】', '', text)
    # 去除分词空格（语料里词与词之间的空格）
    text = re.sub(r' +', '', text)
    # 去除多余标点重复
    text = re.sub(r'[。．.]{2,}', '。', text)
    text = text.strip()
    return text

print(">>> 正在读取语料...")
df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
df = df.dropna(subset=["sentence"])
df = df[df["sentence"].str.len() < 1000]
df = df.drop_duplicates(subset=["sentence"])
df = df.reset_index(drop=True)
print(f">>> 有效语料共 {len(df)} 条")

# 清洗sentence_clean字段，作为展示用
df["sentence_display"] = df["sentence_clean"].apply(clean_sentence)
# 对空的展示字段用清洗后的sentence补充
df["sentence_display"] = df.apply(
    lambda row: row["sentence_display"] if row["sentence_display"].strip() else clean_sentence(row["sentence"]),
    axis=1
)

print(">>> 正在加载多语言向量模型（paraphrase-multilingual-MiniLM-L12-v2）...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print(">>> 正在生成向量，请稍候（约10-20分钟）...")
sentences = df["sentence_display"].tolist()
embeddings = model.encode(sentences, batch_size=64, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

print(">>> 正在建立 FAISS 索引...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)

print(">>> 正在保存索引文件...")
faiss.write_index(index, INDEX_FILE)

metadata = df[["sentence", "keyword", "is_error", "sentence_clean", "sentence_display"]].to_dict(orient="records")
with open(META_FILE, "wb") as f:
    pickle.dump(metadata, f)

print(f">>> 完成！共索引 {index.ntotal} 条语料")
print(f">>> 已生成：{INDEX_FILE} 和 {META_FILE}")