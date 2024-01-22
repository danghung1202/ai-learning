from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app_config import AppConfig

llm = ChatOpenAI(openai_api_key=AppConfig.getOpenAIKey())

#Prompt templates are used to convert raw user input to a better input to the LLM.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

#We can now combine these into a simple LLM chain:
chain = prompt | llm 
chain.invoke({"input": "how can langsmith help with testing?"})
