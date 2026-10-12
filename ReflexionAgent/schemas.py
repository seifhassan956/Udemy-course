from typing import List
from pydantic import BaseModel,Field

class Reflection(BaseModel):
    missing: str = Field(description="critique of what is missing in the answer")
    superfluous: str = Field(description="critique of what is superfluous in the answer")

class AnswerQuestion(BaseModel):
    answer: str = Field(description="the answer to the question")
    reflection: Reflection = Field(description="reflection on the answer, including critique of what is missing and what is superfluous")
    search_queries: List[str] = Field(description="list of recommended search queries to research information and improve the answer")

class RevisedAnswer(AnswerQuestion):
    references: List[str] = Field(description="list of references used in the revised answer")