import streamlit as st
from src.llm.gemini_client import GeminiClient
from src.ui.chat_interface import ChatInterface
from src.config.settings import GEMINI_API_KEY, API_DOCS

def main():
    gemini_client = GeminiClient(GEMINI_API_KEY, API_DOCS)
    chat_interface = ChatInterface(gemini_client)
    chat_interface.render()

if __name__ == "__main__":
    main()