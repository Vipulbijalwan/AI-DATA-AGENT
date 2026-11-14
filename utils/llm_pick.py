
import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def pick_llm(level: str):
    """
    Selects an LLM based on the question difficulty.
    """

    if level.lower() == "low":
        model = "openai/gpt-oss-20b"

    elif level.lower() == "high":
        model = "openai/gpt-oss-120b"

    else:
        raise ValueError("Invalid level. Choose 'low' or 'high'.")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set the GROQ_API_KEY environment variable before using the LLM.")

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0
    )

    return llm

if __name__ == "__main__":
    # Example usage
    llm = pick_llm("low")
    response = llm.invoke("What is the capital of France?")
    print(response.content)
