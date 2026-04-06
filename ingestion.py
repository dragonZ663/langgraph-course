from dotenv import load_dotenv

load_dotenv()

import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 创建Embeddings模型实例
embedding_llm = OllamaEmbeddings(model=os.environ.get("EMBEDDING_MODEL"), temperature=0)

# 待加载的网页
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 进行网页加载
pages = [WebBaseLoader(url).load() for url in urls]
doc_list = [item for subList in pages for item in subList]

# 开始分块
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=0
)
chunks = text_splitter.split_documents(doc_list)

# 进行向量化并存储到向量数据库
vector_store = Chroma.from_documents(
    documents=chunks,
    collection_name=os.environ.get("INDEX_NAME"),
    embedding=embedding_llm,
    persist_directory=os.environ.get("PERSIST_DIRECTORY"),
)
