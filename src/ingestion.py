"""
Document Ingestion Pipeline
-----------------------------
Run once (or whenever documents change) to:
  1. Create the Pinecone index (if it doesn't exist)
  2. Load documents from docs/hr, docs/technical, docs/compliance
  3. Chunk them with RecursiveCharacterTextSplitter
  4. Embed with OpenAI text-embedding-3-small
  5. Upsert into Pinecone, one namespace per doc type

Usage:
    python -m src.ingestion
"""
import time
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    get_embeddings,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_DIMENSION,
    PINECONE_NAMESPACES,
)

# Map namespace → folder path
DOC_FOLDERS = {
    "hr":         "docs/hr",
    "technical":  "docs/technical",
    "compliance": "docs/compliance",
}


def create_pinecone_index():
    """Create the Pinecone serverless index if it doesn't already exist."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing:
        print(f"  Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            print("  Waiting for index to be ready...")
            time.sleep(3)
        print("  Index created and ready.")
    else:
        print(f"  Index '{PINECONE_INDEX_NAME}' already exists. Skipping creation.")

    return pc.Index(PINECONE_INDEX_NAME)


def load_documents_from_folder(folder_path: str, doc_type: str):
    """Load all .txt and .pdf files from a folder and tag metadata."""
    docs = []
    path = Path(folder_path)

    if not path.exists():
        print(f"  WARNING: Folder '{folder_path}' does not exist. Skipping.")
        return docs

    for file in sorted(path.glob("**/*")):
        if file.suffix == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
        elif file.suffix == ".pdf":
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(str(file))
            except ImportError:
                print(f"  WARNING: pypdf not installed. Skipping {file.name}")
                continue
        else:
            continue

        loaded = loader.load()
        for doc in loaded:
            doc.metadata["doc_type"] = doc_type
            doc.metadata["source"]   = file.name
        docs.extend(loaded)
        print(f"  Loaded: {file.name}  ({len(loaded)} page(s))")

    return docs


def chunk_documents(docs, chunk_size: int = 512, chunk_overlap: int = 64):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest_all():
    """Main ingestion entry point — processes all three namespaces."""
    print("\n" + "=" * 55)
    print("ENTERPRISE RAG BOT — DOCUMENT INGESTION")
    print("=" * 55)

    print("\n[Step 1] Setting up Pinecone index...")
    create_pinecone_index()

    embeddings = get_embeddings()

    for doc_type, folder in DOC_FOLDERS.items():
        print(f"\n[Step 2/{list(DOC_FOLDERS.keys()).index(doc_type)+2}] "
              f"Processing namespace: {doc_type.upper()}")

        docs = load_documents_from_folder(folder, doc_type)
        if not docs:
            print(f"  No documents found in '{folder}'. Skipping.")
            continue

        chunks = chunk_documents(docs)
        print(f"  Chunked into {len(chunks)} pieces "
              f"(chunk_size=512, overlap=64)")

        print(f"  Upserting to Pinecone namespace '{doc_type}'...")
        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME,
            namespace=doc_type,
        )
        print(f"  ✓ Done — {len(chunks)} chunks indexed in '{doc_type}'")

    print("\n" + "=" * 55)
    print("✅  Ingestion complete! Your knowledge base is ready.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    ingest_all()
