import datetime
import os
from dotenv import load_dotenv
load_dotenv()


from langchain_core.messages  import HumanMessage
from langchain_core.output_parsers import (
    JsonOutputToolsParser,
    PydanticToolsParser
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from schemas import AnswerQuestion, ReviseAnswer

llm = ChatOpenAI(
    model="Pro/moonshotai/Kimi-K2.5",
    api_key=os.environ.get("SILICON_API_KEY"),
    base_url=os.environ.get("SILICON_BASE_URL")
)

parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """您是一位资深研究员。
            当前时间：{time}
            1. {first_instruction}
            2. 反思并批判性地审视您的答案。务必认真对待，以最大限度地改进。
            3. 推荐一些搜索查询，以查找信息并改进您的答案。""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "请使用规定的格式回答用户提出的上述问题。"
        )
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="请提供一份约250字的详细答案。"
)

# 一个 Pydantic 类 = 一个 “工具”
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion]
)

revise_instructions = """根据新信息修改之前的答案。
    - 你应该利用之前的反馈意见，在答案中添加重要信息。
        - 你必须在修改后的答案中包含数字引用，以确保其可验证性。
        - 在答案末尾添加“参考文献”部分（不计入字数限制）。格式如下：
            - [1] https://example.com
            - [2] https://example.com
    - 你应该利用之前的反馈意见，删除答案中多余的信息，并确保答案不超过 250 字。
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer])


if __name__ == "__main__":
    human_message = HumanMessage(
        content="撰写关于人工智能驱动的 SOC / autonomous SOC 问题领域的文章,"
        " 列出从事该领域研究并已获得融资的初创公司。"
    )

    chain = (
        first_responder_prompt_template 
        | llm.bind_tools(tools=[AnswerQuestion])
        | parser_pydantic
    )

    res = chain.invoke({"messages": [human_message]})
    print(res)