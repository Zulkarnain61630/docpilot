import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
print("Asking Gemini... please wait")
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="what time is it in Delhi, India",
)

print(response.text)
print("Tokens used:", response.usage_metadata.total_token_count)