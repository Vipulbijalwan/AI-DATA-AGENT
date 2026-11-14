
import os
import re
import sys
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# Project imports
from utils.database import DatabaseUtil
from utils.llm_pick import pick_llm
from models.schema import AgentSchema, JudgeSchema


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def get_database():
    env_config = dotenv_values(BASE_DIR / ".env")
    conn_details = {
        "host": os.getenv("DB_HOST") or env_config.get("host") or os.getenv("host"),
        "port": os.getenv("DB_PORT") or env_config.get("port") or os.getenv("port"),
        "database": os.getenv("DB_NAME") or env_config.get("database") or os.getenv("database"),
        "user": (
            os.getenv("DB_USER")
            or os.getenv("DB_USERNAME")
            or env_config.get("username")
            or env_config.get("user")
        ),
        "password": (
            os.getenv("DB_PASSWORD")
            or env_config.get("password")
            or os.getenv("password")
        )
    }

    missing = [key for key, value in conn_details.items() if not value]
    if missing:
        raise ValueError(
            "Missing database configuration: " + ", ".join(missing)
        )

    db = DatabaseUtil(conn_details)
    db.connect()
    return db


# --------------------------------------------------
# NODE 1: CURATE QUESTION
# --------------------------------------------------

def curate_question(state: AgentSchema) -> dict:

    user_question = state.user_question

    llm = pick_llm("low")

    response = llm.invoke(
        f"""
        Rewrite the following user question into a clear,
        specific question suitable for SQL analysis.

        User question: {user_question}
        """
    )

    curated = response.content

    return {
        "curated_ques": curated,
        "messages": [
            HumanMessage(content=curated)
        ]
    }


# --------------------------------------------------
# NODE 2: GET DATABASE SCHEMA AND CREATE PROMPT
# --------------------------------------------------

def prompt_query_context(state: AgentSchema) -> dict:

    curated_question = state.curated_ques

    db = get_database()
    try:
        schema_info = db.schema_details("public")
    finally:
        db.close()

    prompt = f"""
    You are an expert SQL analyst.

    Your task is to generate a PostgreSQL SELECT query
    based on the user's question.

    USER QUESTION:
    {curated_question}

    DATABASE SCHEMA:
    {schema_info}

    RULES:
    1. Generate only a SELECT query.
    2. Do not generate INSERT, UPDATE, DELETE or DROP.
    3. Use only tables and columns present in the schema.
    4. Do not invent table or column names.
    5. Return only executable SQL.
    6. Never generate multiple SQL statements.
    """

    return {
        "prompt_quert_text": prompt
    }


# --------------------------------------------------
# NODE 3: GENERATE SQL
# --------------------------------------------------

def generate_sql(state: AgentSchema) -> dict:

    prompt = state.prompt_quert_text

    llm = pick_llm("high")

    response = llm.invoke(prompt)

    sql_query = response.content.strip()

    # Remove Markdown SQL code fences
    sql_query = re.sub(
        r"^```(?:sql)?\s*|\s*```$",
        "",
        sql_query,
        flags=re.IGNORECASE
    ).strip()

    return {
        "generated_sql_query": sql_query,
        "messages": [
            AIMessage(content=f"Generated SQL:\n{sql_query}")
        ]
    }


# --------------------------------------------------
# NODE 4: CHECK SQL SAFETY
# --------------------------------------------------

def is_safe_sql(state: AgentSchema) -> dict:

    sql_query = state.generated_sql_query

    llm = pick_llm("low")

    llm_judge = llm.with_structured_output(JudgeSchema)

    prompt = f"""
    You are an SQL security judge.

    Determine whether the SQL query is safe.

    A safe query must:
    1. Be a single SELECT statement.
    2. Only retrieve data.
    3. Not modify database records.
    4. Not modify database structure.
    5. Not execute stored procedures or dangerous commands.

    SQL QUERY:
    {sql_query}

    Return:
    answer = Yes or No
    comments = Explanation
    """

    response = llm_judge.invoke(prompt)

    return {
        "is_safe_sql_response": response.answer,
        "comments": response.comments
    }


# --------------------------------------------------
# NODE 5: CANCEL UNSAFE SQL
# --------------------------------------------------

def canceled_sql(state: AgentSchema) -> dict:

    comments = state.comments

    final_answer = f"""
    The generated SQL query was considered unsafe.

    Reason: {comments}

    The query was not executed.
    """

    return {
        "final_answer": final_answer,
        "messages": [
            AIMessage(content=final_answer)
        ]
    }


# --------------------------------------------------
# NODE 6: EXECUTE SAFE SQL
# --------------------------------------------------

def execute_sql(state: AgentSchema) -> dict:

    sql_query = state.generated_sql_query

    db = get_database()
    try:
        result = db.execute_query(sql_query)
    finally:
        db.close()

    return {
        "sql_query_execution_result": str(result),
        "messages": [
            AIMessage(content=f"SQL execution result: {result}")
        ]
    }


# --------------------------------------------------
# NODE 7: REPRESENT FINAL ANSWER
# --------------------------------------------------

def represent_final_answer(state: AgentSchema) -> dict:

    executed_result = state.sql_query_execution_result

    curated_question = state.curated_ques

    llm = pick_llm("low")

    prompt = f"""
    You are a data analyst assistant.

    Explain the SQL query results in simple language.

    USER QUESTION:
    {curated_question}

    QUERY RESULT:
    {executed_result}

    Provide a clear and concise answer.
    Do not invent information that is not present in the results.
    """

    response = llm.invoke(prompt)

    final_answer = response.content

    return {
        "final_answer": final_answer,
        "messages": [
            AIMessage(content=final_answer)
        ]
    }


# --------------------------------------------------
# CONDITIONAL EDGE
# --------------------------------------------------

def is_safe_condition(state: AgentSchema) -> str:

    if state.is_safe_sql_response == "Yes":
        return "execute_sql"

    return "canceled_sql"


# --------------------------------------------------
# BUILD LANGGRAPH
# --------------------------------------------------

sql_agent_graph = StateGraph(AgentSchema)

# Add nodes
sql_agent_graph.add_node("curate_question", curate_question)

sql_agent_graph.add_node(
    "prompt_query_context",
    prompt_query_context
)

sql_agent_graph.add_node("generate_sql", generate_sql)

sql_agent_graph.add_node("is_safe_sql", is_safe_sql)

sql_agent_graph.add_node("canceled_sql", canceled_sql)

sql_agent_graph.add_node("execute_sql", execute_sql)

sql_agent_graph.add_node(
    "represent_final_answer",
    represent_final_answer
)


# Add edges
sql_agent_graph.add_edge(START, "curate_question")

sql_agent_graph.add_edge(
    "curate_question",
    "prompt_query_context"
)

sql_agent_graph.add_edge(
    "prompt_query_context",
    "generate_sql"
)

sql_agent_graph.add_edge(
    "generate_sql",
    "is_safe_sql"
)


# Conditional routing
sql_agent_graph.add_conditional_edges(
    "is_safe_sql",
    is_safe_condition,
    {
        "execute_sql": "execute_sql",
        "canceled_sql": "canceled_sql"
    }
)


sql_agent_graph.add_edge("execute_sql", "represent_final_answer")

sql_agent_graph.add_edge("represent_final_answer", END)

sql_agent_graph.add_edge("canceled_sql", END)


# --------------------------------------------------
# RUN AGENT
# --------------------------------------------------

if __name__ == "__main__":

    sql_analyst = sql_agent_graph.compile()

    input_schema = {
        "messages": [],
        "user_question": (
            "What are the different types of payment method we have in our db"
          
        )
    }

    sql_analyst_response = sql_analyst.invoke(input_schema)

    print(sql_analyst_response["messages"])
    print("*****************************")

    print("\nGenerated SQL Query:")
    print(sql_analyst_response.get("generated_sql_query", ""))  

    print("****************************")

    print(sql_analyst_response.get("sql_query_execution_result", "")  )
   
    print("****************") 

    print(sql_analyst_response.get("prompt_query_context", ""))

    print("****************") 
 
    print(sql_analyst_response.get("final_answer", ""))

    print("\n=======================================")
    
