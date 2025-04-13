import os
import getpass
import json
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Get OpenAI API Key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Initialize LLM and embeddings model
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# FAISS index directory
index_folder = "faiss_indexes"

# Ask user whether to load an existing FAISS index or create a new one
if os.path.exists(index_folder) and os.path.isdir(index_folder):
    user_choice = input("Existing FAISS index found. Use it? (yes/no): ").strip().lower()

    if user_choice == "yes":
        try:
            # Load existing FAISS index
            retrieval_vector_store = FAISS.load_local(index_folder, embeddings_model, allow_dangerous_deserialization=True)
            print(f"FAISS index loaded successfully! Contains {retrieval_vector_store.index.ntotal} vectors.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            print("Proceeding with new FAISS index creation...")
            user_choice = "no"
    else:
        print("Creating a new FAISS index...")
else:
    print("No existing FAISS index found. Creating a new one...")
    user_choice = "no"

# Create FAISS index if needed
if user_choice == "no":
    file_path = input("Enter the PDF file path: ").strip()

    # Load PDF
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()

    # Convert to LangChain Document objects
    all_splits = [
        Document(page_content=doc.page_content, metadata={
            "source": file_path,
            "page_number": doc.metadata.get("page_label", "Unknown")
        })
        for doc in docs
    ]

    # Save metadata to JSON (Optional)
    # json_data = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in all_splits]
    # json_file_path = "metadata.json"
    # with open(json_file_path, "w", encoding="utf-8") as json_file:
    #     json.dump(json_data, json_file, ensure_ascii=False, indent=4)

    # Create FAISS index
    retrieval_vector_store = FAISS.from_documents(all_splits, embeddings_model)

    # Save FAISS index locally
    try:
        os.makedirs(index_folder, exist_ok=True)
        retrieval_vector_store.save_local(index_folder)
        print(f"FAISS index saved at {index_folder}. Contains {retrieval_vector_store.index.ntotal} vectors.")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

# Accept user query
query = input("\nEnter your query: ").strip()
print(f"\nUser Query: {query}")

# Retrieve top 6 most relevant documents
retriever = retrieval_vector_store.similarity_search_with_relevance_scores(query, k=7)

# Display retrieved documents
print("\nRetrieved Documents with Similarity Scores:")
for idx, (doc, score) in enumerate(retriever, start=1):
    print(f"\nDocument {idx}:")
    print(f"Source: {doc.metadata['source']}, Page: {doc.metadata['page_number']}")
    print(f"Content: {doc.page_content[:500]}...")  # Display first 500 chars
    print(f"Similarity Score: {score:.4f}\n")

# Format retrieved documents into a message context
def format_messages(retrieved_docs, query):
    context = "\n\n".join(
        f"Source: {doc.metadata['source']}, Page: {doc.metadata['page_number']}\nContent: {doc.page_content}"
        for doc, _ in retrieved_docs
    )

    return [
        SystemMessage(content=(
            "You are an assistant for answering questions based on retrieved documents. "
            "Use the provided context to answer accurately. "
            "If the answer is not in the context, say 'I don't know'.\n\n"
            f"{context}"
        )),
        HumanMessage(content=query)
    ]

# Query optimization
query_optimisation_template = (
    f"Improve this query: '{query}' to make it clearer and more precise for better retrieval."
)
query_optimised = llm.invoke(query_optimisation_template)

print(f"\nOptimized Query: {query_optimised}")

# Generate response from LLM based on retrieved documents
messages = format_messages(retriever, query_optimised.content)
response = llm.invoke(messages)

# Display AI's response
print("\n\nAI Response:\n")
print(response.content)


# Extract sources from retrieved docs
sources = set(
    f"Source: {doc.metadata['source']}, Page: {doc.metadata['page_number']}"
    for doc, _ in retriever
)
print(f"\n\nSources ===== \n\n{sources}\n")
# sources = sorted(list(sources))

# Format the sources into a readable string
sources_text = "\n".join(sources)
# sources_text = sorted(list(sources_text))
print("\nSources:\n")
print(sources_text)