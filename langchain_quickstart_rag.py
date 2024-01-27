from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from app_config import AppConfig
OPENAI_API_KEY = AppConfig.getOpenAIKey()
faiss_store = AppConfig.getFaissStore()

# Setup vector store as retriever
vectorstore = FAISS.load_local(faiss_store, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
docs = retriever.invoke("ngăn đá tủ lạnh?")
print(docs[0])

# # Basic example: prompt + model + output parser with RAG
# template = """Answer the question based only on the following context:
# context

# Question: {question}
# """
# # Prompt templates are used to convert raw user input to a better input to the LLM.
# prompt = ChatPromptTemplate.from_template(template)

# # Model LLM
# model = ChatOpenAI(openai_api_key=AppConfig.getOpenAIKey())

# # output
# output_parser = StrOutputParser()

# # We can now combine these into a simple LLM chain:
# # In this chain the user input is passed to the prompt template, 
# # then the prompt template output is passed to the model, 
# # then the model output is passed to the output parser. 
# # Let’s take a look at each component individually to really understand what’s going on.
# chain = prompt | model | output_parser
# chain.invoke({"question": "ice cream"})
