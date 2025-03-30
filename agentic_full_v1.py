import os
import getpass
import json
import logging
from dotenv import load_dotenv
from typing import Annotated, List, Tuple, Union
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field
import operator
from langgraph.graph import StateGraph, END
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import Response
from fastapi.responses import JSONResponse

# Configure logging to a file
log_file = "app.log"
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get OpenAI API Key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Initialize models
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# RAG Configuration
index_folder = "faiss_indexes"
conversation_history = []
conversation_summary = ""

# Initialize or load FAISS index
if os.path.exists(index_folder) and os.path.isdir(index_folder):
    logger.info("***************APP Started*****************")
    retrieval_vector_store = FAISS.load_local(index_folder, embeddings_model, allow_dangerous_deserialization=True)
else:
    # Load and index PDFs
    logger.info("Creating New FAISS Indexes.....please wait....")
    pdf_directory = "my_pdfs"
    all_splits = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_directory, filename))
            docs = loader.load()
            for doc in docs:
                all_splits.append(Document(
                    page_content=doc.page_content,
                    metadata={"source": filename, "page_number": doc.metadata.get("page_label", "Unknown")}
                ))
    retrieval_vector_store = FAISS.from_documents(all_splits, embeddings_model)
    retrieval_vector_store.save_local(index_folder)

# Agent State Definition
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    context: List[Document]
    conversation_summary: str

# Planning Models
class Plan(BaseModel):
    steps: List[str] = Field(description="Ordered steps to achieve objective")

class Response(BaseModel):
    response: str

class Act(BaseModel):
    action: Union[Response, Plan] = Field(description="Next action to take")

# Custom RAG Tools
def rag_retriever(query: str) -> str:
    """Retrieve relevant course content using RAG"""
    docs = retrieval_vector_store.similarity_search(query, k=5)
    return "\n\n".join(f"Source: {d.metadata['source']} (Page {d.metadata['page_number']})\nContent: {d.page_content}" for d in docs)

def optimize_query(query: str) -> str:
    """Optimize search query"""
    return llm.invoke(f"Improve this query for course content retrieval: '{query}'").content

# Agent Prompts
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """Create a study plan for the user's objective. Include steps for:
1. Content retrieval from course materials
2. Key concept explanation
3. Practice questions analysis
4. Self-assessment guidance"""),
    ("user", "{input}")
])

replanner_prompt = ChatPromptTemplate.from_template("""
Update study plan based on progress:

Original Objective: {input}
Current Plan: {plan}
Completed Steps: {past_steps}
Context: {context}

Update plan considering what's been learned. Focus on remaining gaps.""")

# Agent Workflow
def format_context_messages(state: PlanExecute):
    context = rag_retriever(state["input"])
    history = "\n".join(f"Q: {h['user']}\nA: {h['assistant']}" for h in conversation_history[-3:])
    return [
        SystemMessage(content=f"""
        You are a course study assistant. Use these resources:
        Course Content: {context}
        Conversation History: {history}
        Current Summary: {state['conversation_summary']}
        """),
        HumanMessage(content=state["input"])
    ]

def plan_step(state: PlanExecute):
    logger.debug(f"Plan step state: {state}")
    logger.info("\nPLANNING PHASE\n")
    logger.info(state["input"])
    plan = llm.with_structured_output(Plan).invoke(planner_prompt.format(input=state["input"]))
    logger.info("\nGenerated Plan:")
    logger.info(json.dumps({"plan": plan.steps}, indent=2))
    return {
        "plan": plan.steps,
        "context": rag_retriever(state["input"]),
        "conversation_summary": state.get("conversation_summary", "")
    }

def execute_step(state: PlanExecute):
    logger.debug(f"Execute step state: {state}")
    logger.info("\nEXECUTION PHASE\n")
    logger.info(f"Current Task: {state['plan'][0]}")
    task = state["plan"][0]
    response = llm.invoke(format_context_messages(state))
    logger.info("\nExecution Result:")
    logger.info(json.dumps({
        "task": task,
        "result": response.content
    }, indent=2))
    return {
        "past_steps": [(task, response.content)],
        "response": response.content
    }

def replan_step(state: PlanExecute):
    logger.debug(f"Replan step state: {state}")
    logger.info("\nREPLANNING PHASE\n")
    logger.info("Current State:")
    logger.info(json.dumps({
        "completed_steps": [step[0] for step in state["past_steps"]],
        "remaining_plan": state["plan"]
    }, indent=2))
    updated_plan = llm.with_structured_output(Act).invoke(
        replanner_prompt.format(**state)
    )
    if isinstance(updated_plan.action, Response):
        logger.info("\nFinal Response Generated:")
        logger.info(json.dumps({"response": updated_plan.action.response}, indent=2))
        return {"response": updated_plan.action.response}
    completed_steps = {step[0] for step in state["past_steps"]}
    new_plan = [step for step in updated_plan.action.steps if step not in completed_steps]
    logger.info("\nUpdated Plan:")
    logger.info(json.dumps({"new_plan": new_plan}, indent=2))
    if not new_plan:
        final_response = {"response": f"Final answer: {state['past_steps'][-1][1]}"}
        logger.info("\nAuto-generated Final Response:")
        logger.info(json.dumps(final_response, indent=2))
        return final_response
    return {"plan": new_plan, "context": rag_retriever(state["input"])}

def should_continue(state: PlanExecute):
    logger.debug(f"Should continue state: {state}")
    if "response" in state and state["response"]:
        return END
    return "agent"

# Build Workflow
workflow = StateGraph(PlanExecute)
workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)
workflow.set_entry_point("planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges("replan", should_continue, {"agent": "agent", END: END})
app_workflow = workflow.compile()

def generate_summary(history: list, previous_summary: str) -> str:
    logger.debug(f"Generate summary history: {history}, previous summary: {previous_summary}")
    """Generate/update conversation summary using LLM"""
    if not history:
        return ""
    recent_history = "\n".join(
        f"Q: {h['user']}\nA: {h['assistant']}"
        for h in history[-3:]
    )
    summary_prompt = (
        "Condense this conversation history into a brief summary. "
        "Focus on key study topics and learning objectives.\n\n"
        f"Previous Summary: {previous_summary}\n\n"
        f"New Interactions:\n{recent_history}\n\n"
        "Updated Summary:"
    )
    return llm.invoke(summary_prompt).content

# FastAPI App
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
async def query_agent(request: QueryRequest):
    global conversation_history, conversation_summary
    query = request.query
    logger.debug(f"Query received: {query}")
    initial_state = {
        "input": query,
        "conversation_summary": conversation_summary
    }

    try:
        response = app_workflow.invoke(
            initial_state,
            {"recursion_limit": 10}
        )
        logger.debug(f"Workflow response: {response}")
    except Exception as e:
        logger.error(f"Error in workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    conversation_summary = generate_summary(
        conversation_history[-3:],
        conversation_summary
    )

    conversation_history.append({"user": query, "assistant": response["response"]})
    conversation_history = conversation_history[-10:]
    logger.info(f"response : {response['response']}")
    return {"response": response["response"]}

@app.get("/logs_json/")
async def get_log_file():
    try:
        logging.info("Received request for log file (JSON)")
        log_lines = []
        with open(log_file, 'r') as file:
            for line in file:
                log_lines.append(line.strip())
        return JSONResponse(content={'log_lines': log_lines})
    except FileNotFoundError:
        logging.error(f"Log file not found: {log_file}")
        raise HTTPException(status_code=404, detail="Log file not found")
    except Exception as e:
        logging.error(f"Error reading log file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading log file: {str(e)}")

@app.post("/clear_logs/")
async def clear_logs():
    try:
        with open(log_file, "w") as f:
            f.write("")  # Clear the file
        return {"message": "Logs cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing logs: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing logs: {str(e)}")