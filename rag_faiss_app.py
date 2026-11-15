
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. LOAD API KEY
# ==========================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please add OPENAI_API_KEY to your .env file")

# ==========================================
# 2. AGRICULTURE DATA
# ==========================================

SCHEME_TEXT = """
PASTE YOUR COMPLETE AGRICULTURE SCHEMES TEXT HERE.

Include all the text you provided:
Tamil Nadu Agriculture Schemes
Kerala Agriculture Schemes
Karnataka Agriculture Schemes
Central Government Agriculture Schemes
Institutional Credit Facilities
Crop Insurance Schemes
Farm Mechanization Schemes
Rice Development Schemes
And all remaining scheme details.
"""

# ==========================================
# 3. INITIALIZE EMBEDDINGS
# ==========================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# ==========================================
# 4. CREATE OR LOAD FAISS DATABASE
# ==========================================

DB_PATH = "_tamil-nadu-agri-scheme_db"

def create_vector_database():

    print("\nCreating vector database...")

    # Convert text into documents
    document = Document(
        page_content=SCHEME_TEXT,
        metadata={"source": "Agriculture Schemes"}
    )

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents([document])

    print("Total chunks:", len(chunks))

    # Create FAISS vector database
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Save database locally
    vectorstore.save_local(DB_PATH)

    print("Vector database saved successfully!")

    return vectorstore


def load_vector_database():

    if os.path.exists(DB_PATH):

        print("\nLoading existing FAISS database...")

        vectorstore = FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        print("Database loaded successfully!")

        return vectorstore

    else:

        return create_vector_database()


# ==========================================
# 5. INITIALIZE LLM
# ==========================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ==========================================
# 6. CREATE PROMPT
# ==========================================

prompt = ChatPromptTemplate.from_template("""
You are an Agriculture Schemes Assistant.

Answer the user's question using ONLY the
information provided in the retrieved context.

If the answer is not available in the context,
say:

"I could not find this information in the
provided agriculture schemes database."

Keep your answer simple and clear.

Context:
{context}

Question:
{question}

Answer:
""")

# ==========================================
# 7. RETRIEVER AND QUESTION ANSWERING
# ==========================================

def ask_question(vectorstore, question):

    # Retrieve relevant documents
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        print("\nNo relevant information found.")
        return

    # Combine retrieved context
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Create prompt
    messages = prompt.format_messages(
        context=context,
        question=question
    )

    # Generate answer
    response = llm.invoke(messages)

    print("\n========== ANSWER ==========\n")
    print(response.content)

    print("\n========== SOURCES ==========\n")

    for i, doc in enumerate(retrieved_docs, 1):
        print(f"Source {i}:")
        print(doc.page_content[:300])
        print("----------------------------")


# ==========================================
# 8. MAIN APPLICATION
# ==========================================

if __name__ == "__main__":

    vectorstore = load_vector_database()

    print("\n================================")
    print(" AGRICULTURE SCHEMES RAG SYSTEM")
    print("================================")

    print("\nAsk questions about agriculture schemes.")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nEnter your question: ")

        if question.lower() == "exit":
            print("Thank you!")
            break

        if not question.strip():
            continue

        ask_question(vectorstore, question)
