import pickle

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser

from langchain.chains import ChatVectorDBChain
from app_config import AppConfig

import langchain
langchain.debug = False
OPENAI_API_KEY = AppConfig.getOpenAIKey()
faiss_store = AppConfig.getFaissStore()

_template = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

#You are an AI assistant for answering questions about machine learning and technical blog posts. 

template = """You are an AI assistant for answering questions about machine learning
and technical blog posts. You are given the following extracted parts of 
a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure.".
Don't try to make up an answer. If the question is not about
machine learning or technical topics, politely inform them that you are tuned
to only answer questions about machine learning and technical topics.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA = PromptTemplate(template=template, input_variables=["question", "context"])
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    else:
        return input["question"]

def get_chain(vectorstore):
    
    # qa_chain = ChatVectorDBChain.from_llm(
    #     llm,
    #     vectorstore,
    #     qa_prompt=QA,
    #     condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    # )
    qa_chain = (
            RunnablePassthrough.assign(
            context=contextualized_question | vectorstore.as_retriever() | format_docs
        )
        #{"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | QA
        | llm
        #| StrOutputParser()
    )
    return qa_chain


if __name__ == "__main__":
    # with open("faiss_store.pkl", "rb") as f:
    #     vectorstore = pickle.load(f)

    vectorstore = FAISS.load_local(faiss_store, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    qa_chain = get_chain(vectorstore)
    chat_history = []
    print("Chat with the Paepper.com bot:")
    while True:
        print("Your question:")
        question = input()
        ai_msg  = qa_chain.invoke({"question": question, "chat_history": chat_history})
        chat_history.append((question, ai_msg ))
        print(f"AI: {ai_msg }")
