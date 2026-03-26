from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()
import os

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一位推特爆款博主，正在给一条推文打分。请对用户的推文进行点评并提出建议。"
            "可以从推文长度、传播潜力、风格等方面进行打分。请使用简短的语言回复，切忌长篇大论。"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名推特科技达人助理，任务是撰写优质的推特帖子。"
            "请根据用户的要求，尽可能生成最佳的推特帖子。"
            "如果用户提出修改建议，请回复修改后的版本。"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(
    model="qwen3.5-flash",
    api_key=os.environ.get("ALIBABA_API_KEY"),
    base_url=os.environ.get("ALIBABA_BASE_URL"),
)

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm
