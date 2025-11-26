# RecipeOnt Chatbot

This repository contains a Retrieval-Augmented Generation (RAG) chatbot grounded in the **RecipeOnt** ontology. It uses a local **Mistral 7B** model to answer culinary questions with high precision, deriving facts from a structured RDF Knowledge Graph containing recipes, ingredients, and flavor molecules.

## File Structure & Function

### **Core Application**
* **`app.py`**: The main Flask application. It loads the FAISS vector index, the metadata (`all_documents.json`), and the local Mistral LLM. It serves the web interface at `/` and the query API at `/ask`.
* **`templates/index2.html`**: The frontend HTML/CSS/JS interface for the chatbot, featuring a dark-mode UI and real-time response interactions.

### **Setup & Data Processing**
* **`Create_VectorDB.ipynb`**: **(RUN THIS FIRST)** This Jupyter Notebook is the setup script. It:
    1.  Parses the RDF ontology (`updated_foodontology_with_instances.rdf`).
    2.  Extracts recipes and ingredients into text documents.
    3.  Generates embeddings using `all-MiniLM-L6-v2`.
    4.  Creates and saves the `faiss_index.index` and `all_documents.json`.
* **`updated_foodontology_with_instances.rdf`**: The source Knowledge Graph file containing the raw culinary data (recipes, ingredients, molecular info).

### **Evaluation**
* **`evaluate.py`**: A G-Eval (LLM-as-a-Judge) script. It runs a test dataset against the local Flask app and uses OpenAI (GPT-4o) to grade the answers for faithfulness and hallucination.
* **`test_dataset.json`**: A "Golden Dataset" containing categorized test questions (Fact Retrieval, Multi-hop Reasoning, Summarization) used by `evaluate.py`.
* **`evaluation.py`**: A variant of the app setup, likely used for testing evaluation logic independently or as a backup.

### **Utilities / Misc**
* **`LLM_inference.ipynb`**: A playground notebook for testing retrieval logic and LLM inference without running the full Flask server.
* **`ModelFile`**: A configuration file for creating a custom Ollama model.


---

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. Install the required libraries:
```bash
pip install flask ctransformers sentence-transformers faiss-cpu rdflib openai pandas tqdm requests
```

### 2. Download the Model
The repository code expects a quantized GGUF model in the root directory. Because this file is too large for GitHub, you must download it manually.

* **Model Name:** `mistral-7b-instruct-v0.1.Q4_K_M.gguf`
* **Download Link:** [TheBloke/Mistral-7B-Instruct-v0.1-GGUF on Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf?download=true)
* **Action:** Download the file and place it directly in the root folder of this project.

### 3. Initialize the Knowledge Base
Before running the chatbot, you must generate the vector index from the RDF file.

1.  Open **`Create_VectorDB.ipynb`** in Jupyter Notebook or VS Code.
2.  Run all cells.
3.  Verify that two new files have been created in your directory:
    * `faiss_index.index`
    * `all_documents.json`

---

##  How to Run

### Run the Chatbot
Start the Flask server:
```bash
python app.py
```
* The app will launch at `http://127.0.0.1:5000`.
* Open this URL in your browser to chat with the system.

### Run the Evaluation
To benchmark the system against the test dataset:

1.  Ensure `app.py` is running in one terminal.
2.  Open a new terminal and run:
    ```bash
    python evaluate.py
    ```
3.  This will generate `evaluation_report.csv` with scores and reasoning for every question.