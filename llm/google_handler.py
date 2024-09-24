import google.generativeai as genai
import os 
from dotenv import load_dotenv

load_dotenv() 

class Google_LLM:
    def __init__(self, model="gemini-1.5-flash") -> None:
        if "GEMINI_API_KEY" in os.environ:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model

    def get_response(self, prompt):
        try:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt + " (Keep responses short).")
            return response.text 
        except:
            return "Gemini client failed."