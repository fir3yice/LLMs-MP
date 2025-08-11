import os

os.environ.setdefault("GEMINI_API_KEY", "")


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-3.3B",
    src_lang="tgl_Latn",  
    use_auth_token=True,
    cache_dir="D:/_GitRepos/Thesis/huggingface_cache"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-3.3B",
    use_auth_token=True,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    bnb_4bit_compute_dtype=torch.float16,
    cache_dir="D:/_GitRepos/Thesis/huggingface_cache"
)


import pandas as pd
import json
from typing import List, Dict, Tuple
from google import genai
from google.genai import types
import time
os.environ.setdefault("GEMINI_API_KEY", "")

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash-lite"   
RATE_LIMIT_SLEEP = 5 
MAX_RETRIES = 3

#from google.cloud import translate_v2 as translate
# def back_translate(filipino_text: str) -> str:
#     result = client.translate(filipino_text, target_language="en", source_language="tl")
#     return result["translatedText"]

# client = translate.Client()

import torch
import os


# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch

# tokenizer = AutoTokenizer.from_pretrained(
#     "facebook/nllb-200-3.3B",
#     src_lang="tgl_Latn",  
#     use_auth_token=True,
#     cache_dir="D:/_GitRepos/Thesis/huggingface_cache"
# )

# model = AutoModelForSeq2SeqLM.from_pretrained(
#     "facebook/nllb-200-3.3B",
#     use_auth_token=True,
#     device_map="auto",
#     load_in_4bit=True,
#     torch_dtype=torch.float16,
#     bnb_4bit_compute_dtype=torch.float16,
#     cache_dir="D:/_GitRepos/Thesis/huggingface_cache"
# )
# def back_translate(filipino_text: str) -> str:
#     inputs = tokenizer(filipino_text, return_tensors="pt")
#     if torch.cuda.is_available():
#         inputs = {k: v.to("cuda") for k, v in inputs.items()}

#     # Specify forced_bos_token_id for English output ("eng_Latn")
#     forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")

#     outputs = model.generate(
#         **inputs,
#         forced_bos_token_id=forced_bos_token_id,
#         max_length=128,
#         do_sample=False,  # greedy decoding
#     )

#     translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return translated_text



import requests
# pip install libretranslate
# libretranslate
def back_translate(text: str) -> str:
    source_lang = "en"
    target_lang = "tl"
    url = "http://127.0.0.1:5000/translate" #my localhost
    data = {
        "q": text,
        "source": source_lang,
        "target": target_lang,
        "format": "text"
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        translated = response.json().get("translatedText", "")
        return translated
    else:
        print(f"Error: {response.status_code} {response.text}")
        return ""



def call_gemini(system_instruction: str, user_prompt: str, temperature: float = 0.0) -> str:
    client = genai.Client(api_key=API_KEY)
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_prompt)],
        )
    ]
    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction,
        response_mime_type="text/plain"
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            )
            text = response.text or ""
            return text.strip()
        except Exception as e:
            print(f"[call_gemini] Attempt {attempt+1} failed: {e}")
            time.sleep(RATE_LIMIT_SLEEP * (attempt + 1))
    raise RuntimeError("call_gemini failed after retries")


def evaluate_pair(english: str, filipino: str, human_score: int, explanation1: str, explanation2: str) -> Dict:
    import re

    paraphrase_prompt = f"""
    Paraphrase the following sentences concisely. The sentences are separate, do not think about them together. Only provide one paraphrase for each:

    English sentence: {english}

    Filipino sentence (translate and paraphrase into English): {filipino}

    Respond with the paraphrases in this format:

    English paraphrase: [your paraphrase here]
    Filipino paraphrase (in English): [your paraphrase here]
    """

    paraphrase_response = call_gemini(
        system_instruction="You are a helpful paraphraser. You are intelligent. Simply do as instructed.",
        user_prompt=paraphrase_prompt
    )

    pattern = re.compile(
        r"English paraphrase:\s*(?P<eng>.+?)\s*Filipino paraphrase \(in English\):\s*(?P<fil>.+)", 
        re.DOTALL
    )

    match = pattern.search(paraphrase_response)
    if match:
        paraphrase_src = match.group("eng").strip()
        paraphrase_tr = match.group("fil").strip()
    else:
        paraphrase_src = ""
        paraphrase_tr = ""

    back_translated = back_translate(filipino).strip()

    time.sleep(5)  
    scoring_prompt = f"""
        Given the English source, Filipino translation, and various paraphrases, score the Filipino translation from 1 to 5 on these criteria:

        Accuracy, Fluency, Coherence, Cultural Appropriateness, Completeness.

        English source: {english}
        Filipino translation: {filipino}
        Back-translation: {back_translated}

        Respond clearly with each criterion’s score and justification in this format:
        Accuracy: 4.5 |  [your explanation here]
        Fluency: 4.0 |  [your explanation here]
        Coherence: 4.2 |  [your explanation here]
        Cultural Appropriateness: 5.0 |  [your explanation here]
        Completeness: 4.8 |  [your explanation here]
        """
    scoring_response = call_gemini(
        system_instruction="You are a helpful assistant. You are an expert translator and evaluator.",
        user_prompt=scoring_prompt
    )

    time.sleep(5)  
    scores_and_justifications = {}
    for line in scoring_response.splitlines():
        if ": " in line and "|" in line:
            key_part, rest = line.split(":", 1)
            score_part, justification_part = rest.split("|", 1)
            key = key_part.strip()
            try:
                score = float(score_part.strip())
            except ValueError:
                score = None
            justification = justification_part.strip()
            scores_and_justifications[key] = {
                "score": score,
                "justification": justification
            }

    summary_prompt = f"""
        Based on these scores and justifications:

        {scoring_response}

        Provide an integer score from 1-5 of the Filipino translation, and a brief summary of the overall translation quality.
        Score:
        Summary:
        """

    summary = call_gemini(
        system_instruction="You are a helpful assistant.",
        user_prompt=summary_prompt
    )

    time.sleep(5) 

    return {
        "english": english,
        "filipino": filipino,
        "paraphrase_src": paraphrase_src,
        "paraphrase_tr": paraphrase_tr,
        "back_translated": back_translated,
        "scores_and_justifications": scores_and_justifications,
        "summary": summary,
        "human_score": human_score,
        "explanation1": explanation1,
        "explanation2": explanation2
    }


results = []
traces = []
def run_agentic_on_dataframe(df: pd.DataFrame,
                             text_col_src: str,
                             text_col_tr: str,
                             out_json_path: str,
                             max_rows: int = None) -> Tuple[List[Dict], List[Dict]]:
    count = 0
    min = 55

    rows = df.head(max_rows) if max_rows else df
    for idx, row in rows.iterrows():
        # if count < min:
        #     count += 1
        #     continue
        print("Running evaluation for row:", idx)
        english = str(row[text_col_src]).strip()
        filipino = str(row[text_col_tr]).strip()
        human_score = int(row.get("Score", None))
        explanation1 = str(row.get("Rater 1 Explanation", "")).strip()
        explanation2 = str(row.get("Rater 2 Explanation", "")).strip()

        eval_result = evaluate_pair(english, filipino, human_score, explanation1, explanation2)
        results.append(eval_result)

        traces.append({
            "row_index": idx,
            "english": english,
            "filipino": filipino,
            "paraphrase_src": eval_result["paraphrase_src"],
            "paraphrase_tr": eval_result["paraphrase_tr"],
            "back_translated": eval_result["back_translated"]
        })

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results, traces

if __name__ == "__main__":
    data = pd.read_csv("data1.csv")

    out_path = "agentic_results_13_libretl.json"
    res, traces = run_agentic_on_dataframe(
        data,
        text_col_src="English",
        text_col_tr="Filipino",
        out_json_path=out_path,
        max_rows= None
    )
    print(f"Saved {len(res)} evaluation results to {out_path}")
