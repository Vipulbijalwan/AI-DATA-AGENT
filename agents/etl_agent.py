import os
import json
import sys

from pathlib import Path
# Project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

sys.path.append(os.path.dirname(os.path.abspath(__file__))) 

from utils.llm_pick import pick_llm
from utils.etl_tools import ETLTools
from models.schema import AgentSchema, JudgeSchema ,ETLAgentSchema
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langchain_openai import ChatOpenAI

#--------------------------------------------------

@tool
def extract_load_tool(url: str, output_folder: str, format: str):
    """
    This tool extracts the data from the API (url) and loads it into a 
    desired location(output_folder).

    Args:
        url (str): The API endpoint to extract data from.
        output_folder (str): The folder where the extracted data will be saved.
    
        Returns:
        str: The path to the saved data file.
     """
    etl_tools = ETLTools()
    return etl_tools.extract_lode(url, output_folder, format)


@tool
def transform_load_tool(file_path:str, output_file:str, output_format:str):
    """
    This tool transforms the data from the input file and loads it into a 
    desired location(output_file).

    Args:
        file_path (str): The path to the input file.
        output_file (str): The file path where the transformed data will be saved.
        output_format (str): The format of the output file ('csv' or 'json').

    Returns:
        str: The path to the saved transformed data file.

        """
    etl_tools = ETLTools()
    top_3_rows = etl_tools.transform_load_context(file_path, output_file, user_question, output_format)

    llm = pick_llm("high")

    prompt = f"""
       you are a python data analyst who uses pandas to analyze data. You have been given a dataset in the form of a file located at {top_3_rows}.
       Your task is to analyze the data and provide insights. Please provide a summary of the dataset

       create the pandas dataframe from data source in the file {file_path} and provide a summary of the dataset.
       write the code to transform and save the data at {output_file} in {output_format} format.
       here the user question is {user_question} and the file path is {top_3_rows}. Please provide the code and output in json format.
       here the context of the dataset is {top_3_rows}. Please provide the code and output in json format."""

    response = llm.invoke(prompt).content

    pandas_code= response.split().strip('```').strip().lstrip('python').strip()

    result = etl_tools.execute_code(pandas_code)

    return f"Data transformed and saved to {output_file}. Here is the code used for transformation: {pandas_code}. The result of the transformation is: {result}"


tools = [extract_load_tool, transform_load_tool]

llm = pick_llm("high")
llm_bind=llm.bind_tools(tools)



def llm_node(state: ETLAgentSchema) :
    system_message = SystemMessage(
        content=(
            "You are a data extraction assistant. Use the available extraction "
            "tool when the user asks to fetch API data, then report the saved file path."
        )
    )
    final_answer = llm_bind.invoke([system_message, *state.messages])
    return {"messages": [final_answer]}

def tool_node(state: ETLAgentSchema) :
    """Execute the tools requested by the latest assistant message."""
    tools_results = []
    tools_by_name = {tool.name: tool for tool in tools}
    tool_calls = state.messages[-1].tool_calls

    for tool_call in tool_calls:
        selected_tool = tools_by_name[tool_call["name"]]
        observation = selected_tool.invoke(tool_call["args"])
        tools_results.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )

    return {"messages": tools_results}



etl_analyst_graph = StateGraph(ETLAgentSchema)
etl_analyst_graph.add_node("llm_node", llm_node)
etl_analyst_graph.add_node("tool_node", tool_node)

etl_analyst_graph.add_edge(START,"llm_node")

def is_tool_call(state: ETLAgentSchema) -> str:
    if state.messages and state.messages[-1].tool_calls:
        return "tool_node"
    else:
        return "end"

etl_analyst_graph.add_conditional_edges("llm_node", is_tool_call, {"tool_node": "tool_node", "end": END})

etl_analyst_graph.add_edge("tool_node", "llm_node")

if __name__ == "__main__":
    etl_analyst = etl_analyst_graph.compile()

    response = etl_analyst.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Extract data from https://pokeapi.co/api/v2/pokemon/ "
                        "and save it as CSV in data/extracted."
                    )
                )
            ],
        }
    )

    print(response["messages"][-1].content)