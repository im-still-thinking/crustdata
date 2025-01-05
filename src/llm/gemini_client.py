import google.generativeai as genai
from typing import List, Dict
from ..lib.doc_manager import DocChunkManager

class GeminiClient:
    def __init__(self, api_key: str, api_docs: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.doc_manager = DocChunkManager(api_docs)

    def get_response(self, query: str, conversation_history: List[Dict[str, str]], api_docs: str) -> str:
        relevant_docs = self.doc_manager.get_relevant_chunks(query)
        
        context = f"""You are a technical assistant for Crustdata's API.
        Your role is to help users understand and use the API effectively.
        
        Relevant API Documentation:
        {relevant_docs}
        
        Current conversation:
        {self._format_conversation(conversation_history)}
        
        User query: {query}
        
        Provide a clear and accurate response based on the relevant API documentation."""

        try:
            response = self.model.generate_content(context)
            return response.text
        except Exception as e:
            return f"I apologize, but I encountered an error. Please try rephrasing your question. Error: {str(e)}"

    def _format_conversation(self, history: List[Dict[str, str]]) -> str:
        formatted = ""
        for msg in history[-5:]:
            formatted += f"{msg['role']}: {msg['content']}\n"
        return formatted