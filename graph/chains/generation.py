from langchain_core.output_parsers import StrOutputParser
from langsmith import Client

from graph.llm import gemma4_llm

client = Client()
prompt = client.pull_prompt("rlm/rag-prompt")

generation_chain = prompt | gemma4_llm | StrOutputParser()
