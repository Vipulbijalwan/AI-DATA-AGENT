
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from operator import add


class AgentSchema(BaseModel):
    messages: Annotated[list, add] = Field(
        default_factory=list,
        description="List of messages exchanged with the agent"
    )

    user_question: str = Field(
        ...,
        description="User's original question"
    )

    curated_ques: str = Field(
        default="",
        description="Curated question for the agent"
    )

    prompt_quert_text: str = Field(
        default="",
        description="A detailed prompt with SQL database context for the agent"
    )
    generated_sql_query : str = Field(
        default="",
        description="The SQL query generated y the  agent"
    )
    is_safe_sql_response: Literal["Yes", "No"] = Field(
        default="No",
        description="Determine wheather the query is safe or not"
    )
    comments : str = Field(
        default="",
        description="Comments regarding the safety to generated sql query"
    )

    sql_query_execution_result: str = Field(
        default="",
        description="SQL query executed by the agent"
    )

    final_answer: str = Field(
        default="",
        description="Final answer provided by the agent"
    )


class JudgeSchema(BaseModel):
    answer: Literal["Yes", "No"] = Field(
        ...,
        description="Indicates if generated SQL query is safe or not"
    )

    comments: str = Field(
        ...,
        description="Comments on the safety of the SQL query"
    )

class ETLAgentSchema(BaseModel):
    messages: Annotated[list, add] = Field(
        default_factory=list,
        description="List of messages exchanged with the ETL agent"
    )