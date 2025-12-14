import json
import requests
import pandas as pd
import time
from openai import OpenAI
from tqdm import tqdm

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
FLASK_API_URL = "http://127.0.0.1:5000/ask"  # URL of your local RAG app
JUDGE_MODEL = "gpt-4o-mini"                  # Efficient and cheap evaluator
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"  # Replace with your key

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------------------------------
# PROMPT TEMPLATE FOR THE JUDGE
# ------------------------------------------------------------------
JUDGE_PROMPT = """
You are a strict Culinary Data Scientist evaluating a RAG system grounded in an Ontology.

Your goal is to score the **Faithfulness** and **Accuracy** of the AI's answer based *only* on the provided Context.

**User Query:** {query}

**Retrieved Context (Source Data):** {context}

**AI Answer (To Evaluate):** {answer}

**Evaluation Criteria:**
1. **Triple Integrity:** If the Answer claims "Ingredient X has Flavor Y", this relationship MUST exist in the Context.
2. **Chemical Precision:** Check that specific flavor molecules (e.g., 'vanillin', 'cineole') listed in the Answer appear exactly in the Context.
3. **No Hallucination:** The AI must not invent cooking steps, ingredients, or nutritional values not present in the source text.
4. **Relevance:** Does the answer actually address the user's specific question?

**Scoring Rubric:**
- **5:** Perfect. All facts found in context. No hallucinations. Precise chemical names.
- **4:** Good. Minor details missed, but factually accurate.
- **3:** Mixed. Some correct facts, but includes minor hallucinations or generic fluff not in context.
- **2:** Poor. Major hallucinations (invented ingredients/molecules) or failed to answer.
- **1:** Terrible. Completely irrelevant or dangerous misinformation.

**Output Format:**
Return valid JSON only:
{{
    "reasoning": "Explain exactly which facts were verified or hallucinated...",
    "score": <integer_1_to_5>
}}
"""

# ------------------------------------------------------------------
# MAIN EVALUATION LOOP
# ------------------------------------------------------------------
def run_evaluation():
    # 1. Load the Test Dataset
    try:
        with open("test_dataset.json", "r") as f:
            test_data = json.load(f)
        print(f"Loaded {len(test_data)} test cases from test_dataset.json")
    except FileNotFoundError:
        print("Error: 'test_dataset.json' not found. Please create it first.")
        return

    results = []

    # 2. Iterate through each question
    print("Starting evaluation... (This may take a few minutes)")
    for item in tqdm(test_data):
        query = item['query']
        complexity = item.get('difficulty', 'Unknown')
        
        # --- A. Query your Local Mistral App ---
        try:
            # We assume your app now returns {"answer": "...", "context": "..."}
            response = requests.post(FLASK_API_URL, json={"user_input": query}, timeout=500)
            if response.status_code != 200:
                print(f"Error querying Flask: {response.status_code}")
                continue
            
            data = response.json()
            ai_answer = data.get("answer", "No answer provided")
            retrieved_context = data.get("context", "")

        except Exception as e:
            print(f"Connection Error: {e}")
            ai_answer = "ERROR"
            retrieved_context = ""

        # --- B. Ask the Judge (GPT-4o) ---
        formatted_prompt = JUDGE_PROMPT.format(
            query=query,
            context=retrieved_context[:10000], # Truncate if too massive
            answer=ai_answer
        )

        try:
            judge_response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "You are a specialized AI evaluator."},
                    {"role": "user", "content": formatted_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0 # Deterministic for consistency
            )
            
            # Parse Judge Output
            eval_content = json.loads(judge_response.choices[0].message.content)
            score = eval_content.get("score", 0)
            reasoning = eval_content.get("reasoning", "No reasoning provided")

        except Exception as e:
            print(f"Judge Error: {e}")
            score = 0
            reasoning = f"Evaluation failed: {str(e)}"

        # --- C. Log Result ---
        results.append({
            "id": item.get('id'),
            "query": query,
            "difficulty": complexity,
            "mistral_answer": ai_answer,
            "retrieved_context_snippet": retrieved_context[:200] + "...",
            "judge_score": score,
            "judge_reasoning": reasoning
        })
        
        # Small sleep to avoid rate limits if using free tier keys
        time.sleep(1.5)

    # 3. Save and Summarize
    df = pd.DataFrame(results)
    
    # Save detailed CSV
    output_filename = "evaluation_report.csv"
    df.to_csv(output_filename, index=False)
    
    print("\n" + "="*40)
    print(f"EVALUATION COMPLETE")
    print("="*40)
    print(f"Average Score: {df['judge_score'].mean():.2f} / 5.0")
    print(f"Detailed results saved to: {output_filename}")
    
    # Display low scores for immediate review
    low_scores = df[df['judge_score'] <= 3]
    if not low_scores.empty:
        print("\nTop Issues (Score <= 3):")
        print(low_scores[['query', 'judge_score', 'judge_reasoning']].to_string(index=False))

if __name__ == "__main__":
    run_evaluation()