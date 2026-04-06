from dotenv import load_dotenv

load_dotenv()
import os
from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """对检索到的文档进行相关性检查的二元评分。"""

    binary_score: Literal["yes", "no"] = Field(
        description="文档与问题“yes”或“no”相关。"
    )


pydantic_parser = PydanticOutputParser(pydantic_object=GradeDocuments)
format_instruction = pydantic_parser.get_format_instructions()
llm = ChatOpenAI(
    model="qwen/qwen3.5-9b",
    api_key=os.environ.get("LM_STUDIO_API_KEY"),
    base_url=os.environ.get("LM_STUDIO_BASE_URL"),
    temperature=0,
)

system_prompt = """
    您是一名评分员，负责评估检索到的文档与用户问题的相关性。\n
    如果文档包含与问题相关的关键词或语义信息，则将其评为相关。\n
    给出“yes”或“no”的二元评分，以表明文档是否与问题相关。
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "检索的文档：\n\n {documents} \n\n 用户的问题: {question}"),
        ("system", "你需要按照如下格式进行回答：{format_instruction}"),
    ]
).partial(format_instruction=format_instruction)

retrieval_grader = prompt | llm | pydantic_parser
