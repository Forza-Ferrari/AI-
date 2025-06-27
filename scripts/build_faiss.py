# scripts/build_faiss.py

import sys
from pathlib import Path
import json
from tqdm import tqdm
import torch

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 加载 config
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from config import settings

def build_index():
    # 0. 检查 GPU
    if torch.cuda.is_available():
        print(f"✅ 检测到 GPU：{torch.cuda.get_device_name(0)}，将用于嵌入计算")
        device = "cuda"
    else:
        print("⚠️ 未检测到 GPU，将使用 CPU，嵌入速度可能较慢")
        device = "cpu"

    # 1. 读取 contexts.json
    print("[1/4] 加载 contexts.json ...")
    path = Path(project_root) / "data" / "contexts.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    print(f"    共载入 {len(raw['contexts'])} 条知识")

    # 2. 构造 Document 对象
    print("[2/4] 构造 Document 对象 ...")
    documents = []
    for item in tqdm(raw["contexts"], desc="处理条目"):
        content = f"示例问：{item.get('ask','').strip()}\n示例答：{item.get('answer','').strip()}"
        meta = {
            "department": item.get("department", "").strip(),
            "title": item.get("title", "").strip()
        }
        documents.append(Document(page_content=content, metadata=meta))

    # 3. 嵌入并构建索引
    print("[3/4] 生成文本嵌入并构建 FAISS 向量索引 ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": device}
    )

    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    vectors = []

    batch_size = 512
    for i in tqdm(range(0, len(texts), batch_size), desc="生成嵌入"):
        batch = texts[i:i + batch_size]
        vec = embeddings.embed_documents(batch)
        vectors.extend(vec)

    # ✅ 核心：将 texts 和 vectors 配对
    text_embeddings = list(zip(texts, vectors))

    vector_db = FAISS.from_embeddings(
        text_embeddings,
        embedding=embeddings,
        metadatas=metadatas
    )



    # 4. 保存到本地
    print(f"[4/4] 保存索引到：{settings.VECTOR_DB_PATH}")
    vector_db.save_local(settings.VECTOR_DB_PATH)
    print("✅ FAISS 索引构建完成！")

if __name__ == "__main__":
    build_index()
