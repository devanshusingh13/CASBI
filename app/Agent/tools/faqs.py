import os
import pandas as pd
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# File Paths
CSV_PATH = r"CASBI\app\Agent\tools\sbi_faq.csv"
FAISS_PATH = r"D:\sbi-hackathon\CASBI\app\Agent\tools\faiss_index"

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_or_create_faiss():
    """Loads FAISS index if exists, otherwise creates and saves it."""
    if os.path.exists(FAISS_PATH):
        print(f"✅ Loading FAISS index from {FAISS_PATH}")
        return FAISS.load_local(FAISS_PATH, embedding_model)

    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV file NOT found at {CSV_PATH}")
        exit()

    print("🛠 Creating FAISS index from CSV...")

    # Load CSV and clean data
    df = pd.read_csv(CSV_PATH)
    df.dropna(inplace=True)

    # Convert to LangChain Documents
    documents = [
        Document(page_content=row["question"],
                 metadata={"answer": row["answer"]})
        for _, row in df.iterrows()
    ]

    # Create and save FAISS index
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(FAISS_PATH)
    print(f"✅ FAISS index saved at {FAISS_PATH}")

    return vectorstore


# Initialize FAISS vector store
vectorstore = load_or_create_faiss()


async def search_faq(query: str, top_k: int = 4) -> str:
    """Asynchronously searches FAISS for similar questions."""
    try:
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(None, vectorstore.similarity_search_with_score, query, top_k)

        if docs:
            return "\n".join([f"{i+1}. {doc[0].metadata['answer']}" for i, doc in enumerate(docs)])
        return "Sorry, I couldn't find an answer to that."
    except Exception as e:
        return f"Error during search: {e}"

# Async User Input Handling


async def main():
    while True:
        user_query = input(
            "Enter your question (or type 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        answer = await search_faq(user_query)
        print("Answer:\n", answer)

if __name__ == "__main__":
    asyncio.run(main())
