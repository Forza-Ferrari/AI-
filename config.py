# config.py

import os
from dotenv import load_dotenv
from pathlib import Path

# 加载 .env 文件中的环境变量
load_dotenv()

class Settings:
    # FAISS 索引文件存放路径
    VECTOR_DB_PATH: str = str(Path(__file__).parent / "data" / "faiss_index")

    # 从环境变量中读取 API Key
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

    # 嵌入模型（可选中文模型）
    EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"

    # 重试次数
    MAX_RETRIES = 3

    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

settings = Settings()
