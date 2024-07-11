from langchain import hub
from langchain_openai.chat_models import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools


llm = ChatOpenAI(model="gpt-3.5-turbo")

query = "What is the capital of France?"


def researchAgent(query, llm):
    tools = load_tools(['ddg-search', 'wikipedia'], llm=llm)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt, verbose=True)
    web_context = agent_executor.invoke({"input": query})
    return web_context["output"]


print(researchAgent(query, llm))
