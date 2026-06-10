vectors.append({
    "id": str(i + j),
    "values": emb,
    "metadata": {
        "text": text,
        "keyword": str(df.iloc[i + j]["keyword"]) if "keyword" in df.columns else "",
        "is_error": str(df.iloc[i + j]["is_error"]) if "is_error" in df.columns else "",
        "sentence_clean": str(df.iloc[i + j]["sentence_clean"]) if "sentence_clean" in df.columns else ""
    }
})