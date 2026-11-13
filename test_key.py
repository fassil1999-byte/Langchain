import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("OpenAI API key loaded successfully!")
else:
    print(" iam the coolest person in the world!")