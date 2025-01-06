from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from typing import List, Dict
from ..lib.doc_manager import StructuredDocManager

class GeminiClient:
    def __init__(self, api_key: str, api_docs: str):
        self.doc_manager = StructuredDocManager(api_docs, api_key)
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=ConversationBufferWindowMemory(k=5),
            verbose=True
        )

    def get_response(self, query: str, conversation_history: List[Dict[str, str]], api_docs: str) -> str:
        relevant_docs = self.doc_manager.get_relevant_chunks(query)
        
        prompt = f"""You are a technical assistant for Crustdata's API.
        Your role is to help users understand and use the API effectively.
        
        Relevant API Documentation:
        {relevant_docs}
        
        User query: {query}
        
        Provide a clear and accurate response based on the relevant API documentation."""

        try:
            response = self.conversation.predict(input=prompt)
            return response
        except Exception as e:
            return f"I apologize, but I encountered an error. Please try rephrasing your question. Error: {str(e)}"