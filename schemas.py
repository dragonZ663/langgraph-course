from typing import List
from pydantic import BaseModel, Field

class Reflection(BaseModel):
    # 对缺失部分的批判。
    # description 是给大模型看的指令！Pydantic 会把这些 description 转换成 工具描述 发给 AI。
    missing: str = Field(description="对缺失部分的批判。")
    # 对冗余部分的批判
    superfluous: str = Field(description="对冗余部分的批判")

class AnswerQuestion(BaseModel):
    """回答问题。"""
    # 250字左右详细回答
    answer: str = Field(description="250字左右详细回答")
    # 反思（嵌套上面的 Reflection）
    reflection: Reflection = Field(description="你对最初答案的反思。")
    # 1~3 条搜索查询
    search_queries: List[str] = Field(description="1-3 个搜索查询，用于研究改进方案，以解决对您当前答案的批评。")

class ReviseAnswer(AnswerQuestion):
    """修改你最初对问题的回答。"""

    references: List[str] = Field(description="支持您更新答案的引证材料。")