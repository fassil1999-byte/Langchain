from langchain_core.documents import Document

# Sample data
data = [
    Document(
        page_content="Artificial Intelligence enables machines to perform tasks that normally require human intelligence.",
        metadata={"source": "AI Notes", "category": "Technology"}
    ),
    Document(
        page_content="Machine Learning is a branch of AI that allows computers to learn from data.",
        metadata={"source": "ML Notes", "category": "Technology"}
    ),
    Document(
        page_content="Deep Learning uses neural networks with multiple layers to learn complex patterns.",
        metadata={"source": "DL Notes", "category": "Technology"}
    )
]

# Display documents
for i, doc in enumerate(data):
    print(f"\nDocument {i + 1}")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)


    print("\n========== DOCUMENT ANALYSIS ==========")

# Total documents
print("Total Documents:", len(data))

# Access first document
first_doc = data[0]

# Access its content
print("\nFirst Document Content:")
print(first_doc.page_content)

# Access its metadata
print("\nFirst Document Metadata:")
print(first_doc.metadata)

# Access individual metadata values
print("\nSource:", first_doc.metadata["source"])
print("Category:", first_doc.metadata["category"])


from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=60,
    chunk_overlap=10
)

# Split documents
chunks = splitter.split_documents(data)

# Display results
print("\n========== TEXT SPLITTING ==========")

print("Original documents:", len(data))
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)