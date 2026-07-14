import pandas as pd
import re

print(">>> 正在读取语料库...")
df = pd.read_csv("bcc_corpus_clean.csv", encoding='utf-8-sig')
print(f">>> 原始数据共 {len(df)} 条")

def clean_sentence(s):
    if not isinstance(s, str):
        return ''
    # 去掉例N：前缀
    s = re.sub(r'^例\d+[：:]', '', s).strip()
    # 去掉【】但保留词语本身
    s = re.sub(r'【(.+?)】', r'\1', s)
    # 去掉词语之间多余的空格
    s = re.sub(r'\s+', '', s)
    return s.strip()

# 生成干净的句子
df['sentence_display'] = df['sentence'].apply(clean_sentence)

# 过滤掉太短或空的句子
df = df[df['sentence_display'].str.len() > 8]
df = df[df['sentence_display'].notna()]

print(f">>> 清洗后共 {len(df)} 条")
print(">>> 示例：")
for s in df['sentence_display'].head(5):
    print(f"  {s}")

# 保存新的CSV
df.to_csv("bcc_corpus_v2.csv", index=False, encoding='utf-8-sig')
print(">>> 已保存到 bcc_corpus_v2.csv")