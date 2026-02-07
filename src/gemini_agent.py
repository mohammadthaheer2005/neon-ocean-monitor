import google.generativeai as genai
import os

class GeminiOceanAgent:
    """
    Interfaces with Google's Gemini API to provide tactical analysis of ocean data.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.model = None
        if api_key:
            try:
                genai.configure(api_key=api_key)
                # Fallback to 'gemini-pro' which is widely available
                self.model = genai.GenerativeModel('gemini-pro')
            except Exception as e:
                print(f"Stats: Gemini Init Failed: {e}")

    def analyze(self, context_text, user_query):
        """
        Sends ocean stats and user question to Gemini.
        """
        if not self.model:
            return "⚠️ Gemini Offline. Please enter a valid API Key and hit Enter."

        prompt = f"""
        You are NEON-AI, an Ocean Intelligence System.
        
        DATA CONTEXT:
        {context_text}
        
        USER QUESTION:
        "{user_query}"
        
        INSTRUCTIONS:
        1. Answer the question based on the data.
        2. If Risk is HIGH, give a warning.
        3. Keep it short (2 sentences).
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ AI Error: {str(e)} (Try checking your API Key)"
