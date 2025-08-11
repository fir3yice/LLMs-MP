import os
import json
import re
import time
from typing import Dict
from google import genai
from google.genai import types
import requests

API_KEY = os.environ.setdefault("GEMINI_API_KEY", "")
MODEL = "gemini-2.5-flash-lite"
RATE_LIMIT_SLEEP = 5
MAX_RETRIES = 3

client = genai.Client(api_key=API_KEY)

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. You are an expert translator and evaluator."
)

def back_translate(text: str) -> str:
    source_lang = "en"
    target_lang = "tl"
    url = "http://127.0.0.1:5000/translate"
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
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    system_instruction=system_instruction,
                    response_mime_type="text/plain"
                )
            )
            text = response.text or ""
            return text.strip()
        except Exception as e:
            print(f"[call_gemini] Attempt {attempt+1} failed: {e}")
            time.sleep(RATE_LIMIT_SLEEP * (attempt + 1))
    raise RuntimeError("call_gemini failed after retries")

def paraphrase_pair(english: str, filipino: str) -> Dict[str, str]:
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

    return {"paraphrase_src": paraphrase_src, "paraphrase_tr": paraphrase_tr}


def score_translation(english: str, filipino: str, back_translated: str = None) -> Dict:
    back_text_display = back_translated if back_translated else "(none)"
    scoring_prompt = f"""
        Given the English source, Filipino translation, and various paraphrases, score the Filipino translation from 1 to 5 on these criteria:

        Accuracy, Fluency, Coherence, Cultural Appropriateness, Completeness.

        English source: {english}
        Filipino translation: {filipino}
        Back-translation: {back_text_display}

        Respond clearly with each criterion’s score and justification in this format:
        Accuracy: 4.5 |  [your explanation here]
        Fluency: 4.0 |  [your explanation here]
        Coherence: 4.2 |  [your explanation here]
        Cultural Appropriateness: 5.0 |  [your explanation here]
        Completeness: 4.8 |  [your explanation here]
        """ #Also provide an integer overall score (1–5) at the end, labeled 'Score:'.

    scoring_response = call_gemini(
        system_instruction=SYSTEM_INSTRUCTION,
        user_prompt=scoring_prompt
    )

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

    return {
        "scores_and_justifications": scores_and_justifications,
        "raw_text": scoring_response
    }

def summarize_scores(scoring_text: str) -> str:
    summary_prompt = f"""
        Based on these scores and justifications:

        {scoring_text}

        Provide an integer score from 1-5 of the Filipino translation, and a brief summary of the overall translation quality.
        Score:
        Summary:
        """
    return call_gemini(
        system_instruction="You are a helpful assistant.",
        user_prompt=summary_prompt
    )


if __name__ == "__main__":
    all_results = []
    print("=== Interactive English–Filipino Translation Evaluation (with Paraphrasing) ===")

    while True:
        english = input("\nEnter English sentence (or 'quit'): ").strip()
        if english.lower() == "quit":
            break
        filipino = input("Enter Filipino translation: ").strip()

        print("\n--- Processing Evaluation ---")
        print(f"Input text is: {english}")
        print(f"Input translation is: {filipino}")
        
        paraphrases = paraphrase_pair(english, filipino)
        print("\n--- Paraphrases ---")
        print(f"English paraphrase: {paraphrases['paraphrase_src']}")
        print(f"Filipino paraphrase (in English): {paraphrases['paraphrase_tr']}")

        first_scores = score_translation(english, filipino, back_translated=None)
        print("\n--- Initial Scores & Justifications ---")
        print(first_scores["raw_text"])

        back_translated = None
        reevaluate_bt = input("\nWould you like to reevaluate using back translation? (y/n): ").strip().lower()
        if reevaluate_bt == "y":
            back_translated = back_translate(filipino).strip()
            print(f"\nBack translation generated: {back_translated}")

            reevaluated_scores = score_translation(english, filipino, back_translated)
            active_scores_text = reevaluated_scores["raw_text"]
            active_scores_struct = reevaluated_scores["scores_and_justifications"]

            print("\n--- Reevaluated Scores & Justifications ---")
            print(active_scores_text)
        else:
            active_scores_text = first_scores["raw_text"]
            active_scores_struct = first_scores["scores_and_justifications"]

        reevaluate_fb = input("\nWould you like to provide feedback for reevaluation? (y/n): ").strip().lower()
        if reevaluate_fb == "y":
            user_feedback = input("Enter your feedback: ").strip()
            adapt_prompt = f"""
            You previously gave these scores and justifications:
            {active_scores_text}

            User feedback:
            {user_feedback}

            Adjust weights or focus areas based on feedback, then re-evaluate.
            Provide updated scores and justifications.
            """
            adapted_scores = call_gemini(SYSTEM_INSTRUCTION, adapt_prompt)
            active_scores_text = adapted_scores
            active_scores_struct = {}
            for line in adapted_scores.splitlines():
                if ": " in line and "|" in line:
                    key_part, rest = line.split(":", 1)
                    score_part, justification_part = rest.split("|", 1)
                    key = key_part.strip()
                    try:
                        score = float(score_part.strip())
                    except ValueError:
                        score = None
                    justification = justification_part.strip()
                    active_scores_struct[key] = {
                        "score": score,
                        "justification": justification
                    }

            print("\n--- Adapted Scores & Justifications ---")
            print(active_scores_text)

        final_summary = summarize_scores(active_scores_text)
        print("\n--- Final Summary ---")
        print(final_summary)


        match = re.search(r"Score:\s*(\d+)", final_summary)
        overall_score = int(match.group(1)) if match else None
        result_entry = {
            "english": english,
            "filipino": filipino,
            "paraphrase_src": paraphrases["paraphrase_src"],
            "paraphrase_tr": paraphrases["paraphrase_tr"],
            "back_translated": back_translated if back_translated else "",
            "scores_and_justifications": active_scores_struct,
            "overall_score": overall_score,  
            "summary": final_summary,
            "human_score": None,
            "explanation1": "",
            "explanation2": ""
        }

        all_results.append(result_entry)

    out_file = "evaluation_results_structured.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_results)} evaluations to {out_file}")

