from google import genai
from mistralai.client import Mistral
from dotenv import load_dotenv
import os

load_dotenv()


class LLMService:

    def __init__(self, provider):

        self.provider = provider

        if provider == "gemini":

            self.client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY")
            )

        elif provider == "mistral":

            self.client = Mistral(
                api_key=os.getenv("MISTRAL_API_KEY")
            )

        else:
            raise ValueError(
                f"Unsupported provider: {provider}"
            )

    def generate(self, prompt):

        if self.provider == "gemini":

            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

            return response.text

        elif self.provider == "mistral":

            response = self.client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            return response.choices[0].message.content