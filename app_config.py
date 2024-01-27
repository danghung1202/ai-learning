import os
import dotenv

import langchain
langchain.debug = False

dotenv.load_dotenv()

class AppConfig:
    @staticmethod
    def getOpenAIKey() -> str:
        openAIKey = os.getenv('OPENAI_API_KEY')
        if openAIKey == "":
            print(f"Open API Key must be configured")
        return openAIKey

    @staticmethod
    def getFaissStore() -> str:
        openAIKey = os.getenv('FAISS_STORE')
        if openAIKey == "":
            openAIKey = "faiss_store"
        return openAIKey