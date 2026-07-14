import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

print(">>> 正在加载向量模型...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print(">>> 正在读取清洗后的语料库...")
df = pd.read_csv("bcc_corpus_v2.csv", encoding='utf-8-sig')
print(f">>> 共 {len(df)} 条")

sentences = df['sentence_display'].tolist()

print(">>> 正在编码，请耐心等待（可能需要10-30分钟）...")
batch_size = 512
all_embeddings = []
for i in range(0, len(sentences), batch_size):
    batch = sentences[i:i+batch_size]
    embeddings = model.encode(batch, show_progress_bar=False)
    all_embeddings.append(embeddings)
    if i % 5000 == 0:
        print(f">>> 已处理 {i}/{len(sentences)} 条")

embeddings = np.vstack(all_embeddings).astype('float32')
faiss.normalize_L2(embeddings)

print(">>> 正在构建FAISS索引...")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "bcc_faiss_v2.index")

metadata = []
for _, row in df.iterrows():
    metadata.append({
        "sentence_display": row['sentence_display'],
        "keyword": row.get('keyword', ''),
        "is_error": row.get('is_error', True)
    })

with open("bcc_metadata_v2.pkl", "wb") as f:
    pickle.dump(metadata, f)

print(f">>> 完成，共索引 {index.ntotal} 条")
print(">>> 已保存 bcc_faiss_v2.index 和 bcc_metadata_v2.pkl")