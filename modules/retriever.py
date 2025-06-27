import json
from pathlib import Path
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from config import settings
import torch
from modules.web_searcher import BaiduSearcher

class MedicalRetrieverOffline:
    def __init__(self):
        # 1. 初始化嵌入模型（用 GPU 如果可用）
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": device}
        )

        # 2. 加载 FAISS 向量索引（已由 build_faiss.py 构建）
        self.vector_db = FAISS.load_local(
            folder_path=settings.VECTOR_DB_PATH,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )


        # 3. 构建 BM25（仍从 contexts.json 加载）
        self.bm25 = self._build_bm25()

    def _build_bm25(self):
        json_path = Path(__file__).parent.parent / "data" / "contexts.json"
        with json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        documents = []
        for item in raw["contexts"]:
            content = f"示例问：{item.get('ask','').strip()}\n示例答：{item.get('answer','').strip()}"
            metadata = {
                "department": item.get("department", "").strip(),
                "title": item.get("title", "").strip()
            }
            documents.append(Document(page_content=content, metadata=metadata))

        return BM25Retriever.from_documents(documents)

    def hybrid_retrieve(self, query: str, top_k: int = 5):
        # 向量检索
        faiss_docs = self.vector_db.similarity_search(query, k=top_k)
        # BM25 检索
        bm25_docs = self.bm25.invoke(query)[:top_k]

        # 合并、去重
        combined = {doc.page_content: doc for doc in faiss_docs + bm25_docs}
        return list(combined.values())
class MedicalRetrieverOnline:
    def __init__(self):
        # 移除FAISS和BM25相关初始化
        self.searcher = BaiduSearcher()
        
    def hybrid_retrieve(self, query: str, top_k: int = 5):
        return self.searcher.search_medical_info(query)