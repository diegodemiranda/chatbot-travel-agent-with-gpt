import os
from langchain_openai.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools

llm = ChatOpenAI(model="gpt-3.5-turbo")
tools = load_tools(['ddg-search','wikipedia'], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent= 'zero-shot-react-description',
    verbose=True
)

query = "What is the capital of France?"

agent.run(query)
