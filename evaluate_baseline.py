import json
import requests
import pandas as pd
import time
from openai import OpenAI
from tqdm import tqdm

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
BASELINE_API_URL = "http://127.0.0.1:5000/ask_baseline"  # The new route
RAG_API_URL = "http://127.0.0.1:5000/ask"               # Normal route (to get ground truth)
JUDGE_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"             # <--- PASTE YOUR KEY

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------------------------------
# BASELINE JUDGE PROMPT
# ------------------------------------------------------------------
# We tell the Judge: "Here is the True Data (Context). Here is what the AI guessed. Score it."
BASELINE_JUDGE_PROMPT = """
You are evaluating a "Baseline" AI model (No RAG) against a ground-truth Ontology.

**User Query:** {query}

**Ground Truth (Fact Check):** {context}

**Baseline AI Answer (To Evaluate):** {answer}

**Your Goal:** Determine if the Baseline AI hallucinated or missed key details because it lacked access to the Ground Truth.

**Scoring Criteria:**
- **5 (Perfect):** The AI somehow knew the exact specific facts (e.g., exact chemical IDs or specific ingredient mappings) without context. (Rare).
- **4 (Accurate Generalization):** The AI gave a correct general answer but missed specific ontology details.
- **3 (Vague):** The AI gave a generic answer that is technically true but useless compared to the specific ground truth.
- **2 (Minor Hallucination):** The AI stated facts that contradict the ground truth.
- **1 (Major Hallucination):** The AI invented recipes, ingredients, or chemicals that do not exist in the ground truth.

**Output JSON:**
{{
    "reasoning": "Compare the specific chemicals/ingredients in Truth vs Answer...",
    "score": <integer_1_to_5>
}}
"""

def run_baseline_evaluation():
    # 1. Load Test Data
    try:
        with open("test_dataset.json", "r") as f:
            test_data = json.load(f)
        print(f"Loaded {len(test_data)} test cases.")
    except FileNotFoundError:
        print("Error: test_dataset.json not found.")
        return

    results = []
    print("Starting BASELINE evaluation...")

    for item in tqdm(test_data):
        query = item['query']
        
        # --- A. Get Baseline Answer (No RAG) ---
        try:
            # We call the NEW route
            res_base = requests.post(BASELINE_API_URL, json={"user_input": query}, timeout=500)
            baseline_answer = res_base.json().get("answer", "Error")
        except Exception as e:
            print(f"Baseline API Error: {e}")
            continue

        # --- B. Get Ground Truth Context (from RAG route) ---
        # We need to know the 'Truth' to judge the baseline, so we ask the RAG system 
        # just to fetch the documents, ignoring its answer.
        try:
            res_rag = requests.post(RAG_API_URL, json={"user_input": query}, timeout=500)
            ground_truth_context = res_rag.json().get("context", "")
        except:
            ground_truth_context = "Could not fetch ground truth context."

        # --- C. Ask the Judge ---
        formatted_prompt = BASELINE_JUDGE_PROMPT.format(
            query=query,
            context=ground_truth_context[:15000], # Truncate large context
            answer=baseline_answer
        )

        try:
            judge_res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "You are a strict data evaluator."},
                    {"role": "user", "content": formatted_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            eval_data = json.loads(judge_res.choices[0].message.content)
            score = eval_data.get("score", 0)
            reasoning = eval_data.get("reasoning", "")

        except Exception as e:
            print(f"Judge Error: {e}")
            score = 0
            reasoning = str(e)

        # --- D. Log Results ---
        results.append({
            "id": item.get('id'),
            "query": query,
            "type": "BASELINE (No RAG)",
            "mistral_answer": baseline_answer,
            "ground_truth_snippet": ground_truth_context[:200],
            "score": score,
            "reasoning": reasoning
        })
        
        time.sleep(0.5) # Avoid rate limits

    # 3. Save Results
    df = pd.DataFrame(results)
    output_file = "baseline_evaluation_report.csv"
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*40)
    print(f"BASELINE EVALUATION COMPLETE")
    print(f"Average Baseline Score: {df['score'].mean():.2f} / 5.0")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    run_baseline_evaluation()