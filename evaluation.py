from flask import Flask, render_template, request, Response, jsonify
from ctransformers import AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

# -------------------------------
# Flask Setup
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Load FAISS index and documents
# -------------------------------
print("Loading FAISS index and documents...")
faiss_index = faiss.read_index("faiss_index.index")
with open("all_documents.json", "r") as f:
    all_documents = json.load(f)
print(f"Index with {faiss_index.ntotal} vectors and {len(all_documents)} documents loaded successfully.")

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
    "./",
    model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    model_type="mistral",
    # gpu_layers=20,
    max_new_tokens=1000,
    context_length=2048,
    # threads=8
)
print("Mistral model loaded successfully.")

# -------------------------------
# Helper: Generate answer with context
# -------------------------------
def generate_answer_with_context(query, context):
    """Use the Mistral model to answer a question using retrieved context."""
    prompt = f"""Answer the question using the context if it’s relevant. 
If not, rely on your own knowledge. Prefer information from the context when possible, 
and rewrite any raw or encoded data into clean, natural language.

Context:
{context}

Question:
{query}

Answer:"""

    # Get the full response from the model
    # (Removed stream=True)
    full_response = llm(prompt)
    
    return full_response

# -------------------------------
# Helper: Retrieve relevant docs
# -------------------------------
def retrieve_context(query, k=1):
    """Retrieve most relevant documents as context for the query."""
    query_embedding = embedder.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), k)
    retrieved_docs = [all_documents[i] for i in indices[0]]
    return " ".join(retrieved_docs)

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def index_page():
    return render_template("index2.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("user_input", "")
    print(f"\n--- User Query: {user_input} ---")

    # Retrieve relevant context
    context = retrieve_context(user_input)
    
    # Generate the full answer
    full_answer = generate_answer_with_context(user_input, context)

    # UPDATED RETURN STATEMENT: Include 'context' in the response
    return jsonify({
        "answer": full_answer, 
        "context": context
    })

# -------------------------------
# Run the Flask App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, threaded=True)