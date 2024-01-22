import argparse
# import faiss
import os
import pickle

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chains import create_retrieval_chain

from app_config import AppConfig

import langchain
langchain.debug = False


parser = argparse.ArgumentParser(description='Paepper.com Q&A')
parser.add_argument('question', type=str, help='Your question for Paepper.com')
args = parser.parse_args()

# with open("faiss_store.pkl", "rb") as f:
#     store = pickle.load(f)

store = FAISS.load_local("faiss_store", OpenAIEmbeddings(openai_api_key=AppConfig.getOpenAIKey()))

docs = store.similarity_search(args.question)

print(docs[0].page_content)
print('-------------------------------')
retriever = store.as_retriever();
docs = retriever.invoke(args.question)
print(docs[0].page_content)

#retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
#retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")
#print(retrieved_docs[0].text)
#llm = ChatOpenAI(openai_api_key=AppConfig.getOpenAIKey(), temperature=0, verbose=True)
# chain = VectorDBQAWithSourcesChain.from_llm(
#         llm = ChatOpenAI(openai_api_key=AppConfig.getOpenAIKey(), temperature=0, verbose=True), 
#         vectorstore=store, 
#         verbose=True)
# result = chain({"question": args.question})

# print(f"Answer: {result['answer']}")
# print(f"Sources: {result['sources']}")
