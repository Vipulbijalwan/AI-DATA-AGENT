
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
load_dotenv(BASE_DIR / ".env")

# Add project root to Python path
sys.path.insert(0, str(BASE_DIR))

# Import project modules
from utils.llm_pick import pick_llm
from models.schema import JudgeSchema


# SQL query to analyze
sql_query = "DELETE * FROM users WHERE age > 30;"

# Select LLM based on difficulty
llm = pick_llm("low")

# Enable structured output
llm_judge = llm.with_structured_output(JudgeSchema)

# Create SQL judge prompt
prompt = """
You are an SQL Judge Agent.

Your task is to analyze the given SQL query and determine
whether it is SAFE or UNSAFE.

Check for:
1. SQL injection risks.
2. DROP, DELETE, TRUNCATE or UPDATE operations.
3. Unauthorized data modification.
4. Dangerous SQL commands.
5. Attempts to access sensitive information.

SQL Query:
{sql_query}

Return your judgment using the required structured schema.
"""

# Insert SQL query into prompt
formatted_prompt = prompt.format(sql_query=sql_query)

# Get LLM response
response = llm_judge.invoke(formatted_prompt).model_dump()


# Print result
print("\n========== SQL JUDGE RESULT ==========")
print(response)
print("======================================")
