import json
import random
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
import openai
from google import genai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from openai import OpenAI
import anthropic

# Set API keys from environment variables
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from a .env file

openai.api_key = os.getenv("OPENAI_API_KEY")
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Set additional API keys from environment variables
grok_api_key = os.getenv("GROK_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize clients
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("benchmark_predictions.log"),
    logging.StreamHandler()
])

class SurveyBenchmark:
    def __init__(self, data_path: str, input_questions: List[str], target_questions: List[str]):
        """Initialize the benchmark with data and question selection."""
        self.data = self.load_data(data_path)
        self.input_questions = input_questions
        self.target_questions = target_questions
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
            logging.info(f"Processing respondent {i+1}/{num_samples}")
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
                    # Log each prediction
                    logging.info({
                        'prompt': prompt,
                        'question_id': qid,
                        'prediction': pred,
                        'actual': actual['answer_value'],
                        'accuracy': acc
                    })
            except Exception as e:
                logging.error(f"Error processing respondent {i}: {str(e)}")
                continue
        results['accuracy'] = total_correct / total_questions if total_questions > 0 else 0.0
        return results
    
    def get_gpt4o_prediction(self, prompt: str) -> str:
        """Get prediction from GPT-4o."""
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3
        )
        return completion.choices[0].message.content
    
    def get_gpt35_prediction(self, prompt: str) -> str:
        """Get prediction from GPT-3.5."""
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3
        )
        return completion.choices[0].message['content']
    
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
        
        # Check if CUDA is available and move model to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else None
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,  # Set attention mask
            max_new_tokens=100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def run_benchmark_api(self, prediction_function, num_samples: int = 50) -> Dict[str, float]:
        """Run benchmark for a specific model using an API-based prediction function."""
        sampled_respondents = random.sample(self.data, num_samples)
        results = {
            'accuracy': 0.0,
            'predictions': []
        }
        total_questions = 0
        total_correct = 0
        for i, respondent in enumerate(sampled_respondents):
            logging.info(f"Processing respondent {i+1}/{num_samples}")
            target_questions = [q['question_id'] for q in respondent['data'] if q['use_case'] == 'value' and q['answer_value'] is not None]
            if not target_questions:
                continue
            prompt = self.create_prompt(respondent, target_questions)
            try:
                prediction = prediction_function(prompt)
                pred_answers = [a.strip() for a in prediction.strip().split('\n') if a.strip()]
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
                    logging.info({
                        'prompt': prompt,
                        'question_id': qid,
                        'prediction': pred,
                        'actual': actual['answer_value'],
                        'accuracy': acc
                    })
            except Exception as e:
                logging.error(f"Error processing respondent {i}: {str(e)}")
                continue
        results['accuracy'] = total_correct / total_questions if total_questions > 0 else 0.0
        return results

    def get_grok_prediction(self, prompt: str) -> str:
        """Get prediction from Grok."""
        client = OpenAI(api_key=grok_api_key, base_url="https://api.x.ai/v1")
        completion = client.chat.completions.create(
            model="grok-3-latest",
            messages=[{"role": "system", "content": "You are a PhD-level mathematician."}, {"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3
        )
        return completion.choices[0].message.content

    def get_anthropic_prediction(self, prompt: str) -> str:
        """Get prediction from Anthropic."""
        message = anthropic_client.messages.create(
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            model="claude-3-5-sonnet-latest"
        )
        return message.content

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
        data_path='evaluation/merged_wvs_gss.json',
        input_questions=top10_demographic_ids,
        target_questions=[]  # Not used in new logic
    )

    results = {}

    # Run benchmarks for API-based models separately
    print("\nBenchmarking gpt-4o via API...")
    results['gpt-4o'] = benchmark.run_benchmark_api(benchmark.get_gpt4o_prediction, num_samples=50)
    print(f"\nAccuracy: {results['gpt-4o']['accuracy']:.2f}")

    print("\nBenchmarking gemini-2.0-flash via API...")
    results['gemini-2.0-flash'] = benchmark.run_benchmark_api(benchmark.get_gemini_flash_prediction, num_samples=50)
    print(f"\nAccuracy: {results['gemini-2.0-flash']['accuracy']:.2f}")

    print("\nBenchmarking grok-3-latest via API...")
    results['grok-3-latest'] = benchmark.run_benchmark_api(benchmark.get_grok_prediction, num_samples=50)
    print(f"\nAccuracy: {results['grok-3-latest']['accuracy']:.2f}")

    print("\nBenchmarking claude-3-5-sonnet-latest via API...")
    results['claude-3-5-sonnet-latest'] = benchmark.run_benchmark_api(benchmark.get_anthropic_prediction, num_samples=50)
    print(f"\nAccuracy: {results['claude-3-5-sonnet-latest']['accuracy']:.2f}")

    models = [
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'Qwen/Qwen2-7B-Instruct',
        'microsoft/phi-2',
        'mistralai/Mistral-7B-Instruct-v0.2',
    ]
    for model in models:
        print(f"\nBenchmarking {model}...")
        results[model] = benchmark.run_benchmark(model, num_samples=50)
        print(f"\nAccuracy: {results[model]['accuracy']:.2f}")

    # Save results
    output_path = Path('benchmark_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main() 