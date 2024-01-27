from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app_config import AppConfig

# Basic example: prompt + model + output parser

# Prompt templates are used to convert raw user input to a better input to the LLM.
prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
prompt_value = prompt.invoke({"topic": "ice cream"})
print(prompt_value)
# -> ChatPromptValue(messages=[HumanMessage(content='tell me a short joke about ice cream')])
print(prompt_value.to_messages())
# -> [HumanMessage(content='tell me a short joke about ice cream')]
print(prompt_value.to_string())
# -> 'Human: tell me a short joke about ice cream'

# Model LLM
model = ChatOpenAI(openai_api_key=AppConfig.getOpenAIKey())
message = model.invoke(prompt_value)
print(message)
# -> AIMessage(content="Why don't ice creams ever get invited to parties?\n\nBecause they always bring a melt down!")

# output
output_parser = StrOutputParser()

# We can now combine these into a simple LLM chain:
# In this chain the user input is passed to the prompt template, 
# then the prompt template output is passed to the model, 
# then the model output is passed to the output parser. 
# Let’s take a look at each component individually to really understand what’s going on.
chain = prompt | model | output_parser
chain.invoke({"topic": "ice cream"})
