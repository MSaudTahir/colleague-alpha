from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class OpenAI_LLM:
    def __init__(self, model="gpt-4o-mini") -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if "OPENAI_API_KEY" in os.environ else None
        self.model = model
        
    def get_response(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            return completion.choices[0].message
        except:
            return "OpenAI client failed."