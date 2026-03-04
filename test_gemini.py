import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("No GOOGLE_API_KEY found")
    exit(1)

client = genai.Client(api_key=api_key)
try:
    print("Listing models...")
    for model in client.models.list():
        print(model.name)
except Exception as e:
    print(f"Error listing models: {e}")
