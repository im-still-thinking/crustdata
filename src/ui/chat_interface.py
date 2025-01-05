import streamlit as st
from ..llm.gemini_client import GeminiClient
from ..lib.prompt_builder import PromptBuilder
from ..config.settings import API_DOCS

class ChatInterface:
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client
        self.prompt_builder = PromptBuilder()

    def render(self):
        st.title("Crustdata API Assistant")
        st.write("Ask me anything about the Crustdata API!")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("How can I help you with the Crustdata API?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self.gemini_client.get_response(
                        prompt,
                        st.session_state.messages,
                        API_DOCS
                    )
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
