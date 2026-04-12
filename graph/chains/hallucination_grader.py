from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from graph.llm import chat_llm


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'True' or 'False'"
    )


pydantic_parser = PydanticOutputParser(pydantic_object=GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'True' or 'False'. 'True' means that the answer is grounded in / supported by the set of facts.\n\n
     instruction: {instruction}
     """

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: {documents}\n\n LLLM generation: {generation}\n\n"),
    ]
).partial(instruction=pydantic_parser.get_format_instructions())

hallucination_grader = prompt | chat_llm | pydantic_parser
