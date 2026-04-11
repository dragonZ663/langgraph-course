from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from graph.llm import chat_llm
from langchain_core.output_parsers import PydanticOutputParser



class GradeAnswer(BaseModel):
    """Binary score for whether the answer addresses the question."""

    binary_score: bool = Field(
        description="Whether the answer addresses the question, 'True' or 'False'"
    )

pydantic_parser = PydanticOutputParser(pydantic_object=GradeAnswer)

system = """You are a grader assessing whether the LLM generation addresses / resolves the User question \n 
     Give a binary score 'True' or 'False'. 'True' means that the answer resolves the question.\n\n
     instruction: {instruction}
     """

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question}\n\n LLLM generation: {generation}\n\n"),
    ]
).partial(
    instruction = pydantic_parser.get_format_instructions()
)

answer_grader = prompt | chat_llm | pydantic_parser
