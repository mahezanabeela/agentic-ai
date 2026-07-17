from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 
load_dotenv()
class GroqLLM:
    def __init__(self):
        self.llm = ChatGroq(
            model_name="openai/gpt-oss-120b",
            temperature=0.5,
            max_tokens=512,
            api_key=os.getenv("GROQ_API_KEY")
        )

    def run_llm(self, query: str) -> str:
        response = self.llm.invoke(query)

        return response.content

    def get_llm(self):
        return self.llm

if __name__ == "__main__":
    user_input = input("Ask a question to the LLM:\n")

    llm = GroqLLM()
    response = llm.run_llm(user_input)
    print("\n\n\nCLEAN RESPONSE\n\n\n", response)