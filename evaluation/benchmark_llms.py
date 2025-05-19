import json
import random
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
import openai
from google import genai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set API keys
openai.api_key = "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"
gemini_client = genai.Client(api_key="AIzaSyDYxk8byOOJp0-ipnpAy5WVJWyN6B92vUE")

class SurveyBenchmark:
    def __init__(self, data_path: str, input_questions: List[str], target_questions: List[str]):
        """Initialize the benchmark with data and question selection."""
        self.data = self.load_data(data_path)
        self.input_questions = input_questions
        self.target_questions = target_questions
        print(f"Loaded {len(self.data)} respondents")
        print(f"Input questions: {self.input_questions}")
        print(f"Target questions: {self.target_questions}")
        
    def load_data(self, data_path: str) -> List[Dict]:
        """Load the merged dataset."""
        with open(data_path, 'r') as f:
            return json.load(f)
    
    def create_prompt(self, respondent: Dict, target_questions: list) -> str:
        """Create a prompt from input questions and a list of target questions."""
        prompt = (
            "You are to answer as if you are the following survey respondent. Use only the information provided below. Respond in the first person, and provide ONLY the answer value—no explanations.\n\n"
            "Given the following survey responses:\n\n"
        )
        # Add input questions (top 10 demographic attributes)
        for q in respondent['data']:
            if q['question_id'] in self.input_questions:
                prompt += f"Q: {q['question']}\nA: {q['answer_value']}\n\n"
        # Add all value questions (with non-null answers) as targets
        prompt += (
            "Please answer the following questions as this respondent. Respond with ONLY the answer value for each, nothing else:\n\n"
        )
        for q in respondent['data']:
            if q['question_id'] in target_questions:
                prompt += f"Q: {q['question']}\nA: "
        return prompt
    
    def evaluate_prediction(self, prediction: str, actual: Dict) -> float:
        """Evaluate prediction accuracy."""
        # Clean and normalize both prediction and actual value
        pred = str(prediction).strip().lower()
        act = str(actual['answer_value']).strip().lower()
        
        # Handle numeric values
        try:
            pred_num = float(pred)
            act_num = float(act)
            return 1.0 if abs(pred_num - act_num) < 0.1 else 0.0
        except ValueError:
            # Handle categorical values
            return 1.0 if pred == act else 0.0
    
    def run_benchmark(self, model_name: str, num_samples: int = 50) -> Dict[str, float]:
        """Run benchmark for a specific model."""
        # Sample respondents
        sampled_respondents = random.sample(self.data, num_samples)
        results = {
            'accuracy': 0.0,
            'predictions': []
        }
        total_questions = 0
        total_correct = 0
        for i, respondent in enumerate(sampled_respondents):
            print(f"\rProcessing respondent {i+1}/{num_samples}", end="")
            # Identify all value questions with non-null answers for this respondent
            target_questions = [q['question_id'] for q in respondent['data'] if q['use_case'] == 'value' and q['answer_value'] is not None]
            if not target_questions:
                continue
            prompt = self.create_prompt(respondent, target_questions)
            try:
                # Get model prediction
                prediction = self.get_huggingface_prediction(prompt, model_name)
                # Split model output into answers (one per target question)
                pred_answers = [a.strip() for a in prediction.strip().split('\n') if a.strip()]
                # Evaluate each answer
                for idx, qid in enumerate(target_questions):
                    actual = next(q for q in respondent['data'] if q['question_id'] == qid)
                    pred = pred_answers[idx] if idx < len(pred_answers) else ''
                    acc = self.evaluate_prediction(pred, actual)
                    total_correct += acc
                    total_questions += 1
                    results['predictions'].append({
                        'prompt': prompt,
                        'question_id': qid,
                        'prediction': pred,
                        'actual': actual['answer_value'],
                        'accuracy': acc
                    })
            except Exception as e:
                print(f"\nError processing respondent {i}: {str(e)}")
                continue
        results['accuracy'] = total_correct / total_questions if total_questions > 0 else 0.0
        return results
    
    def get_gpt4o_prediction(self, prompt: str) -> str:
        """Get prediction from GPT-4o."""
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3
        )
        return response.choices[0].message.content
    
    def get_gpt35_prediction(self, prompt: str) -> str:
        """Get prediction from GPT-3.5."""
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3
        )
        return response.choices[0].message.content
    
    def get_gemini_flash_prediction(self, prompt: str) -> str:
        """Get prediction from Gemini."""
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
    
    def get_huggingface_prediction(self, prompt: str, model_name: str) -> str:
        """Get prediction from a HuggingFace model."""
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs["input_ids"],
            max_length=100,
            num_return_sequences=1,
            temperature=0.3,
            do_sample=True,
            top_p=0.9
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    top10_demographic_ids = [
        'country',
        'age',
        'sex',
        'education',
        'marital_status',
        'employment_status',
        'urban_rural',
        'income',
        'religion',
        'ethnicity',
    ]
    benchmark = SurveyBenchmark(
        data_path='merged_wvs_gss.json',
        input_questions=top10_demographic_ids,
        target_questions=[]  # Not used in new logic
    )
    models = [
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'Qwen/Qwen2-7B-Instruct',
        'microsoft/phi-2',
        'mistralai/Mistral-7B-Instruct-v0.2',
        'deepseek-ai/deepseek-llm-7b-instruct'
    ]
    results = {}
    for model in models:
        print(f"\nBenchmarking {model}...")
        results[model] = benchmark.run_benchmark(model, num_samples=50)
        print(f"\nAccuracy: {results[model]['accuracy']:.2f}")
    output_path = Path('benchmark_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main() 