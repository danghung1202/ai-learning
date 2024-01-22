import os
import dotenv

dotenv.load_dotenv()
print(f"{os.getenv('OPENAI_API_KEY')}")

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