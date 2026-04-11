from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()
import os

# 向量化模型
embedding_llm = OllamaEmbeddings(model=os.environ.get("EMBEDDING_MODEL"), temperature=0)

# gemma4 对话模型
# chat_llm = ChatOllama(model="gemma4:e4b", temperature=0)
chat_llm = ChatOpenAI(
    model="qwen3.6-plus",
    api_key=os.environ.get("ALIBABA_API_KEY"),
    base_url=os.environ.get("ALIBABA_BASE_URL"),
    temperature=0,
)
