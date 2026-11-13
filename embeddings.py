from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load API key
load_dotenv()

# Initialize embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Convert text into vector
text = "Artificial Intelligence is amazing."

vector = embeddings.embed_query(text)

# Display results
print("Original text:", text)
print("Vector dimensions:", len(vector))
print("First 10 vector values:", vector[:10])