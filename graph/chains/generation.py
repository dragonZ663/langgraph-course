from langchain_core.output_parsers import StrOutputParser
from langsmith import Client

from graph.llm import chat_llm

client = Client()
prompt = client.pull_prompt("rlm/rag-prompt")

generation_chain = prompt | chat_llm | StrOutputParser()
