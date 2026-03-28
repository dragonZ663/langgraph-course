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
from schemas import AnswerQuestion

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
            """You are expert researcher.
            Current time: {time}

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Answer the user's question above using the required format."
        )
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

# 一个 Pydantic 类 = 一个 “工具”
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion]
)

if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc  problem domain,"
        " list startups that do that and raised capital."
    )

    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion])
        | parser_pydantic
    )

    res = chain.invoke({"messages": [human_message]})
    print(res)