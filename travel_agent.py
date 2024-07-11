from langchain_openai.chat_models import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4

# Chat agent
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Query
query = "What is the capital of France?"


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


print(researchAgent(query, llm))
