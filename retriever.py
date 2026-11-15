from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

# Load API key
load_dotenv()

# Initialize embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Sample documents
documents = [
    Document(
        page_content="Artificial Intelligence enables machines to perform tasks that normally require human intelligence.",
        metadata={"source": "AI Notes"}
    ),
    Document(
        page_content="Machine Learning allows computers to learn from data without being explicitly programmed.",
        metadata={"source": "ML Notes"}
    ),
    Document(
        page_content="Deep Learning uses neural networks with multiple layers to learn complex patterns.",
        metadata={"source": "DL Notes"}
    ),
    Document(
        page_content="Python is a popular programming language used in AI and machine learning.",
        metadata={"source": "Python Notes"}
    )
]

# Create Chroma vector database
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="my_first_retriever"
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# Ask a question
query = "What are the benefits of the Seed Multiplication Scheme of Paddy?"

# Retrieve relevant documents
results = retriever.invoke(query)

# Display results
print("\nUSER QUESTION:", query)
print("\nRETRIEVED DOCUMENTS:")

for i, doc in enumerate(results):
    print(f"\nResult {i + 1}:")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)