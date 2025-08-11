system = """
You are an expert translation evaluator for English ↔ Filipino translations. Assess the quality of the given translation from English to Filipino based only on the provided inputs.

Evaluate according to these six criteria (1 point each):
1. Accuracy - Correct meaning, intent, and details.
2. Fluency - Grammatically correct, natural, and idiomatic Filipino.
3. Coherence - Logical flow and structure matching the source.
4. Cultural Appropriateness - Respects Filipino norms, idioms, and sensitivities.
5. Guideline Adherence - Follows any domain-specific terminology and style.
6. Completeness - Translates all elements without omission/addition.

Scoring:
- Total points: 0-6  
- Normalize to 1-5 scale:  
  • 5 = Excellent (5-6 points)  
  • 3-4 = Good (3-4 points)  
  • 1-2 = Poor (0-2 points)  

Output format (follow exactly):
Score: <number from 1-5>  
Label: <"excellent", "good", or "poor">  
Reasoning:  
Accuracy: <your comment>  
Fluency: <your comment>  
Coherence: <your comment>  
Cultural Appropriateness: <your comment>  
Guideline Adherence: <your comment>  
Completeness: <your comment>  

Be concise but clear in reasoning. Base your evaluation only on the provided inputs.

---

### Example 1
Source (English): The meeting will start at 9 a.m. sharp.  
Translation (Filipino): Ang pulong ay magsisimula nang eksakto alas-nwebe ng umaga.  

Score: 5  
Label: excellent  
Reasoning:  
Accuracy: Fully conveys the meaning and time detail.  
Fluency: Grammatically correct and natural phrasing.  
Coherence: Matches structure and intent of the source.  
Cultural Appropriateness: No issues; standard formal Filipino.  
Guideline Adherence: Appropriate for formal context.  
Completeness: All information is included without additions.  

---

### Example 2
Source (English): She gave him a cold look before leaving.  
Translation (Filipino): Tiningnan niya siya nang malamig bago umalis.  

Score: 3  
Label: good  
Reasoning:  
Accuracy: Literal translation captures meaning but slightly awkward.  
Fluency: Understandable, but “nang malamig” sounds unnatural; “matamang tingin” or “tingin na walang emosyon” would be smoother.  
Coherence: Sequence is clear and logical.  
Cultural Appropriateness: Acceptable, though more idiomatic options exist.  
Guideline Adherence: Fits general domain but not stylistically refined.  
Completeness: All information is present.  
"""



prompt_template = """
Source (English): {english_text}
Translation (Filipino): {filipino_text}
"""

import pandas as pd
import os
filename = "data.csv"
data = pd.read_csv(filename, encoding='utf-8')

data = data.drop(columns=['Contributor'], errors='ignore')
print(f"Loaded {len(data)} rows from {filename}")

def construct_prompt(row, prompt_template):
    english_text = row['Source Text (English)']
    filipino_text = row['Target Text (Filipino)']
    score = row.get('Score', None)  

    prompt = prompt_template.format(english_text=english_text, filipino_text=filipino_text)
    
    return prompt, filipino_text, score

sample_prompt, translated_text, score = construct_prompt(data.iloc[1], prompt_template)

print("Sample Prompt:")
print(sample_prompt) 
print(score)


from google import genai
from google.genai import types
import time
import os
import re

os.environ["GEMINI_API_KEY"] = ""

def query_gemini(prompt):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-2.0-flash"
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]
    
    generate_content_config = types.GenerateContentConfig(
        temperature=0.0,
        system_instruction=system,
        response_mime_type="text/plain",
    )
    
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    
    return response.text.strip() if response and response.text else "unknown"

results = []
# count = 48
# max = len(data)
for i, row in data.iterrows():
    prompt, translated_text, score = construct_prompt(row, prompt_template)

    #print(f"Querying Gemini for row {i}...")
    response = query_gemini(prompt)

    score_match = re.search(r"Score:\s*(\d+)", response)
    if score_match:
        score_match = int(score_match.group(1))
    else:
        score_match = None
    correct_label = "excellent" if score_match == 5 else "good" if score_match >= 3 else "poor"

    accuracy = re.search(r"Accuracy:\s*(.*)", response)
    fluency = re.search(r"Fluency:\s*(.*)", response)
    coherence = re.search(r"Coherence:\s*(.*)", response)
    cultural_appropriateness = re.search(r"Cultural Appropriateness:\s*(.*)", response)
    guideline_adherence = re.search(r"Guideline Adherence:\s*(.*)", response)
    completeness = re.search(r"Completeness:\s*(.*)", response)

    accuracy = accuracy.group(1).strip() if accuracy else ""
    fluency = fluency.group(1).strip() if fluency else ""
    coherence = coherence.group(1).strip() if coherence else ""
    cultural_appropriateness = cultural_appropriateness.group(1).strip() if cultural_appropriateness else ""
    guideline_adherence = guideline_adherence.group(1).strip() if guideline_adherence else ""
    completeness = completeness.group(1).strip() if completeness else ""

    comment1 = data.iloc[i]['Rater 1 Explanation']
    comment2 = data.iloc[i]['Rater 2 Explanation']
    

    results.append({
        "prompt": prompt,
        "response": response,
        "original_score": int(score) if score is not None else None,
        "llm_score": score_match,
        "accuracy": accuracy,
        "fluency": fluency,
        "coherence": coherence,
        "cultural_appropriateness": cultural_appropriateness,
        "guideline_adherence": guideline_adherence,
        "completeness": completeness,
        "correct_label": correct_label,
        "rater_1_comment": comment1 if pd.notna(comment1) else "",
        "rater_2_comment": comment2 if pd.notna(comment2) else "",
    })



    time.sleep(5) 

import json
with open("prompt_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

# 2.0 = results (sys 0), results_1 (sys 1), results_3 (system_3), results_5 (sys 4), results_6 (sys 5), results_7 (sys 6)
# 2.5 = results_2 (sys1), results_4 (sys 4)



human_scores = [row['original_score'] for row in results if row['original_score'] is not None]
llm_scores = [row['llm_score'] for row in results if row['llm_score'] is not None]

from scipy.stats import spearmanr
rho, pval = spearmanr(human_scores, llm_scores)
print(f"Spearman correlation between human and LLM scores: {rho:.4f}, p-value: {pval:.4f}")

agreement = sum(1 for h, l in zip(human_scores, llm_scores) if h == l) / len(human_scores)
print(f"Exact agreement between human and LLM scores: {agreement:.2%}")