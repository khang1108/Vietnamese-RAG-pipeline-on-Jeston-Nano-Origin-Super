from core.prompt_builder import TemplatePrompt

class GeneralPrompt(TemplatePrompt):
    def __init__(self):
        self.msg_system = (
            "You are a helpful Vietnamese educational assistant. "
            "Answer only from the provided context. "
            "If the context is not enough, say you do not have enough information. "
            "For academic questions, explain step by step when useful."
        )
        
    def create_prompt(self, query: str, context: str) -> list[dict[str, str]]:
        msg = f"""
        Use the context below to answer the user's question carefully.

        Context:
        {context}

        Question:
        {query}
        """
        
        return [
            {"role": "system", "content": self.msg_system},
            {"role": "user", "content": msg.strip()}
        ]
