from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load API key
load_dotenv()

# Initialize OpenAI model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Ask AI a question
response = llm.invoke(
    "What is Artificial Intelligence? Explain simply."
)

# Display AI response
print("\nAI RESPONSE:")
print(response.content)