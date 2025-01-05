class PromptBuilder:
    @staticmethod
    def build_technical_prompt(query: str, relevant_docs: str) -> str:
        return f"""As a Crustdata API technical assistant, help with the following query:

Query: {query}

Use this relevant API documentation for reference:
{relevant_docs}

If the query is about API usage, include a practical example in the response.
If relevant, include curl commands or code snippets.
Explain any technical terms used in the response."""
