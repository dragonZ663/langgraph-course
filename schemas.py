from typing import List
from pydantic import BaseModel, Field

class Reflection(BaseModel):
    # 对缺失部分的批判。
    # description 是给大模型看的指令！Pydantic 会把这些 description 转换成 工具描述 发给 AI。
    missing: str = Field(description="Critique of what is missing.")
    # 对冗余部分的批判
    superfluous: str = Field(description="Critique of what is superfluous.")

class AnswerQuestion(BaseModel):
    """Answer the question."""
    # 250字左右详细回答
    answer: str = Field(description="~250 word detailed answer to the question.")
    # 反思（嵌套上面的 Reflection）
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    # 1~3 条搜索查询
    search_queries: List[str] = Field(description="1-3 search queries for researching improvements to address the critique of your current answer.")
