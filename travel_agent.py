from langchain_openai.chat_models import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import bs4

# Chat agent
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Query
query = "What are the best destinations in Brazil?"


# Research agent
def researchAgent(query, llm):
    tools = load_tools(['ddg-search', 'wikipedia'], llm=llm)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt, verbose=True)
    web_context = agent_executor.invoke({"input": query})
    return web_context["output"]


# Load data from the web
def loadData():
    loader = WebBaseLoader(["https://www.dicasdeviagem.com/brasil/", "https://www.dicasdeviagem.com/europa/"],
                           bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=(
                               "post-template-default single single-post postid-54460 single-format-standard edition "
                               "desktop-device regular-nav chrome windows")))
                           )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    splits = text_splitter.split_documents(docs)
    vector_store = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings)
    retriever = vector_store.as_retriever()
    return retriever


# Get relevant documents
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
        imput_variables={"web_context", "relevant_documents", "query"},
        template=prompt_template
    )
    sequence = RunnableSequence(prompt | llm)
    
    response = sequence.invoke({"web_context": web_context, "relevant_documents": relevant_documents, "query": query})
    return response
