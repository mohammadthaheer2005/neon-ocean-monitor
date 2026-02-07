import os
from groq import Groq

class GroqOceanAgent:
    """
    Interfaces with Groq API for sub-second AI responses.
    """
    def __init__(self, api_key=None):
        self.client = None
        if api_key:
            try:
                self.client = Groq(api_key=api_key)
            except Exception as e:
                print(f"Groq Init Error: {e}")

    def analyze(self, context_text, user_query):
        """
        Sends telemetry to Groq (Llama 3).
        """
        if not self.client:
            return "⚠️ Groq Offline. Enter API Key in Sidebar."

        system_prompt = """
        You are NEON-AI, an advanced Oceanographic Decision Support System.
        Your output must be concise, professional, and military-scientific.
        Focus on 'Root Cause' of algae blooms based on provided data (Nitrates/Phosphates).
        """
        
        user_message = f"""
        TELEMETRY DATA:
        {context_text}
        
        USER QUERY:
        {user_query}
        """
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model="llama-3.3-70b-versatile", # Updated to latest stable model
                temperature=0.7,
                max_tokens=200,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"❌ Groq Error: {str(e)} (Try checking model name or API key)"
