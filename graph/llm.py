from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()
import os

# 向量化模型
model = os.environ.get("EMBEDDING_MODEL")
embedding_llm = OllamaEmbeddings(model= model if model else "qwen3-embedding:0.6b", temperature=0)

# gemma4 对话模型
# chat_llm = ChatOllama(model="gemma4:e4b", temperature=0)

api_key=os.environ.get("ALIBABA_API_KEY")
chat_llm = ChatOpenAI(
    model="qwen3.6-plus",
    api_key=SecretStr(api_key) if api_key else None,
    base_url=os.environ.get("ALIBABA_BASE_URL"),
    temperature=0,
)
