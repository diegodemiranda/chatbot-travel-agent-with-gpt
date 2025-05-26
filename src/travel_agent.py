from config import api_key

if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please configure it before running the application.")
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import AIMessage
import bs4
import json

# Chat agent
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)


# Research agent
def researchAgent(query, llm):
    tools = load_tools(['ddg-search', 'wikipedia'], llm=llm)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt)
    try:
        web_context = agent_executor.invoke({"input": query})
        return web_context["output"]
    except Exception as e:
        print(f"Error in researchAgent for query '{query}': {e}") # For logging/debugging
        return {"output": f"Failed to get web context due to: {e}"}


# Load data from the web (RAG)
def loadData():
    loader = WebBaseLoader(["https://www.dicasdeviagem.com/brasil/", "https://www.dicasdeviagem.com/europa/"],
                           bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=(
                               "post-template-default single single-post postid-54460 single-format-standard edition "
                               "desktop-device regular-nav chrome windows")))
                           )
    try:
        docs = loader.load()
    except Exception as e:
        print(f"Error loading data from web: {e}")
        raise e

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    splits = text_splitter.split_documents(docs)

    try:
        vector_store = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-large"))
    except Exception as e:
        print(f"Error creating vector store: {e}")
        raise e

    retriever = vector_store.as_retriever()
    return retriever


def getRelevantDocs(query):
    retriever = loadData()
    relevant_documents = retriever.invoke(query)
    return relevant_documents


# Supervisor agent
def supervisorAgent(query, llm, web_context, relevant_documents):
    prompt_template = """You are a manager at a travel agency specializing in providing complete itineraries and
    tips and use the context of events and ticket prices, user input and relevant documents to prepare your answers.
    Context: {web_context}
    Documents relevantes: {relevant_documents}
    User input: {query}
    Assist:
    """
    
    prompt = PromptTemplate(
        input_variables={"web_context", "relevant_documents", "query"},
        template=prompt_template
    )
    sequence = RunnableSequence(prompt | llm)
    
    try:
        response = sequence.invoke({"web_context": web_context, "relevant_documents": relevant_documents, "query": query})
    except Exception as e:
        print(f"Error in supervisorAgent for query '{query}': {e}")
        # Return an AIMessage object with error content
        return AIMessage(content=f"Error in supervisor agent: {e}")
    return response


def getResponse(query, llm):
    web_context = researchAgent(query, llm)
    relevant_documents = getRelevantDocs(query)
    response = supervisorAgent(query, llm, web_context, relevant_documents)
    return response


# Lambda handler
def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
    except json.JSONDecodeError:
        return {
            "statusCode": 400,
            "headers": {
                "Content-Type": "application/json",
            },
            "body": json.dumps({
                "message": "Invalid JSON in request body."
            })
        }
    query = body.get("question")
    if query is None:
        return {
            "statusCode": 400,
            "headers": {
                "Content-Type": "application/json",
            },
            "body": json.dumps({
                "message": "Missing 'question' parameter in request body."
            })
        }
    try:
        response_content = getResponse(query, llm).content
        return {
            "statusCode": 200,
            "headers":{
                "Content-Type": "application/json",
            },
            "body": json.dumps({
                "message": "task completed successfully",
                "details": response_content
            })
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
            },
            "body": json.dumps({
                "message": "An unexpected error occurred.",
                "error_details": str(e)
            })
        }
