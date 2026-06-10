from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import sys

# 1. 初始化
PINECONE_API_KEY = "pcsk_3C1BoM_5ZszDLArtCSHfAs2SJxb6hZpcwBhFz3rXTrM6Db1VqjrCNys5uCPg44PD4VGkD3"
INDEX_NAME = "hsk-bcc-corpus"

print("正在初始化 Pinecone 和语义模型（约需 10 秒）...")

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    # 加载本地已下载的模型
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ 系统就绪！")
except Exception as e:
    print(f"❌ 初始化失败: {e}")
    sys.exit()

def find_similar_cases(query_text, top_k=3):
    print(f"\n🔍 正在检索与 '{query_text}' 最相似的案例...")
    
    # 向量化
    query_vector = model.encode(query_text).tolist()
    
    # 查询
    results = index.query(
        vector=query_vector, 
        top_k=top_k, 
        include_metadata=True
    )
    
    print(f"--- 检索结果 ---")
    if not results['matches']:
        print("未找到相似案例。")
    for i, res in enumerate(results['matches']):
        print(f"案例 {i+1} [相似度: {res['score']:.4f}]: {res['metadata']['text']}")

# ============ 必须包含以下代码，程序才会执行 ============
if __name__ == "__main__":
    # 你可以把这里换成任何你想测试的错句
    test_sentence = "我是学习在学校。" 
    find_similar_cases(test_sentence)