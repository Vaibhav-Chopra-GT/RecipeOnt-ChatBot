import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM


# -------------------------------
# Load FAISS index and documents
# -------------------------------
print("Loading FAISS index and documents...")

index = faiss.read_index("faiss_index.index")
with open("all_documents.json", "r") as f:
    all_documents = json.load(f)

print(f"Index with {index.ntotal} vectors and {len(all_documents)} documents loaded successfully.")


# -------------------------------
# Load embedding model
# -------------------------------
print("Loading SentenceTransformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------------
# Load local Mistral model
# -------------------------------
print("Loading local Mistral model...")
llm = AutoModelForCausalLM.from_pretrained(
    "./",  # directory containing model file
    model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    model_type="mistral",
    # gpu_layers=20,
    max_new_tokens=1000,
    context_length=2048,
    # threads=8
)
print("Mistral model loaded successfully.")


# -------------------------------
# Function: Generate answer with context
# -------------------------------
def generate_answer_with_context(query, context):
    """Use the Mistral model to answer a question using retrieved context."""
    prompt = f"""Answer the question using the context if it’s relevant. If not, rely on your own knowledge. Prefer information from the context when possible, rewrite any raw or encoded data into clean, natural language.

Context:
{context}

Question:
{query}

Answer:"""

    response = ""
    for token in llm(prompt, stream=True):
        response += token
    return response.strip()


# -------------------------------
# Function: Retrieve relevant docs and query model
# -------------------------------
def answer_query(query, k=1):
    """Retrieve most relevant documents and answer the query using the LLM."""
    print(f"\n--- Query: '{query}' ---")
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    retrieved_docs = [all_documents[i] for i in indices[0]]

    print(f"Retrieved {len(retrieved_docs)} documents for context.")
    context = " ".join(retrieved_docs)

    answer = generate_answer_with_context(query, context)
    print("\n--- Answer ---")
    print(answer)
    return answer


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break
        answer_query(user_query)
