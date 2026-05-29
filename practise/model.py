from google import genai
import os

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY"))

for model in client.models.list():
    print(model.name)