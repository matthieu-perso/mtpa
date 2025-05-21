import json
from tqdm import tqdm
import random
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Callable
import openai
from google import genai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from openai import OpenAI
import anthropic
import os
from dotenv import load_dotenv

# Setup
load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
grok_api_key = os.getenv("GROK_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
hf_api_key = os.getenv("HF_API_KEY")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("benchmark_predictions.log"),
    logging.StreamHandler()
])

class SurveyBenchmark:
    def __init__(self, data_path: str, input_questions: List[str], target_questions: List[str]):
        self.data = self.load_data(data_path)
        self.input_questions = input_questions
        self.target_questions = target_questions
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print(f"Loaded {len(self.data)} respondents")

    def load_data(self, data_path: str) -> List[Dict]:
        with open(data_path, 'r') as f:
            return json.load(f)

    def create_prompt(self, respondent: Dict, target_questions: list) -> str:
        prompt = (
            "You are to answer as if you are the following survey respondent. Use only the information provided below. "
            "Respond in the first person, and provide ONLY the answer value—no explanations.\n\n"
            "Given the following survey responses:\n\n"
        )
        for q in respondent['data']:
            if q['question_id'] in self.input_questions:
                prompt += f"Q: {q['question']}\nA: {q['answer_value']}\n\n"
        if target_questions:
            qid = target_questions[0]
            q = next((q for q in respondent['data'] if q['question_id'] == qid), None)
            if q:
                prompt += "Please answer the following question as this respondent. Respond with ONLY the answer value, nothing else:\n\n"
                prompt += f"Q: {q['question']}\nA: "
        return prompt

    def evaluate_prediction(self, prediction: str, actual: Dict) -> float:
        pred = str(prediction).strip().lower()
        act = str(actual['answer_value']).strip().lower()
        try:
            return 1.0 if abs(float(pred) - float(act)) < 0.1 else 0.0
        except ValueError:
            return 1.0 if pred == act else 0.0

    def run_hf_benchmark_single_model(self, model_name: str, clean_function: Callable[[str], str], num_samples: int = 50) -> Dict[str, float]:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

        sampled_respondents = random.sample(self.data, num_samples)
        results = {'accuracy': 0.0, 'predictions': []}
        total_correct, total_questions = 0, 0

        for i, respondent in enumerate(tqdm(sampled_respondents, desc="Processing respondents", total=num_samples)):
            logging.info(f"Processing respondent {i+1}/{num_samples}")
            for target_question in respondent['data']:
                if target_question['use_case'] != 'value' or target_question['answer_value'] is None:
                    continue
                prompt = self.create_prompt(respondent, [target_question['question_id']])
                try:
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    outputs = model.generate(inputs["input_ids"], max_new_tokens=3)
                    text = tokenizer.batch_decode(outputs)[0]
                    pred = clean_function(text)
                    print("PRED: ", pred)

                    acc = self.evaluate_prediction(pred, target_question)
                    total_correct += acc
                    total_questions += 1
                    results['predictions'].append({
                        'prompt': prompt,
                        'question_id': target_question['question_id'],
                        'category': target_question.get('category', 'Unknown'),
                        'prediction': pred,
                        'actual': target_question['answer_value'],
                        'accuracy': acc
                    })
                    logging.info(results['predictions'][-1])
                except Exception as e:
                    logging.error(f"Error processing respondent {i}: {str(e)}")

        results['accuracy'] = total_correct / total_questions if total_questions > 0 else 0.0
        return results

    def run_api_benchmark(self, model_name: str, api_client: Any, clean_function: Callable[[str], str], num_samples: int = 50) -> Dict[str, float]:
        sampled_respondents = random.sample(self.data, num_samples)
        results = {'accuracy': 0.0, 'predictions': []}
        total_correct, total_questions = 0, 0

        for i, respondent in enumerate(tqdm(sampled_respondents, desc=f"Processing respondents for {model_name}", total=num_samples)):
            logging.info(f"Processing respondent {i+1}/{num_samples}")
            for target_question in respondent['data']:
                if target_question['use_case'] != 'value' or target_question['answer_value'] is None:
                    continue
                prompt = self.create_prompt(respondent, [target_question['question_id']])
                try:
                    response = api_client.generate(prompt=prompt, max_tokens=3)
                    text = response['choices'][0]['text']
                    pred = clean_function(text)
                    print("PRED: ", pred)

                    acc = self.evaluate_prediction(pred, target_question)
                    total_correct += acc
                    total_questions += 1
                    results['predictions'].append({
                        'prompt': prompt,
                        'question_id': target_question['question_id'],
                        'category': target_question.get('category', 'Unknown'),
                        'prediction': pred,
                        'actual': target_question['answer_value'],
                        'accuracy': acc
                    })
                    logging.info(results['predictions'][-1])
                except Exception as e:
                    logging.error(f"Error processing respondent {i}: {str(e)}")

        results['accuracy'] = total_correct / total_questions if total_questions > 0 else 0.0
        return results

# Cleaners

def clean_output(text: str) -> str:
    # Find the last occurrence of '\nA: ' and process the text
    parts = text.split('\nA: ')
    if len(parts) > 0:
        generated_text = parts[-1].replace('<end_of_turn>', '').strip()
        # Remove '\n', 'Q', and punctuation from the generated text
        cleaned_text = generated_text.replace('\n', '').replace('Q', '').replace('</s>', '')
        # Only keep alphanumeric tokens
        tokens = [token for token in cleaned_text.split() if token.isalnum()][:2]
        return ' '.join(tokens).strip()
    return ''


def clean_output_meta_llama(text: str) -> str:
    return text.split('<|start_header_id|>assistant|<end_header_id|>')[-1].split('<eot_id>')[0].strip()

# Add API clients for new models
class GPT4oClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, max_tokens: int):
        response = self.client.responses.create(
            model="gpt-4o",
            instructions="You are a coding assistant that talks like a pirate.",
            input=prompt,
        )
        return {'choices': [{'text': response.output_text}]}

class GPT4oMiniClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, max_tokens: int):
        response = self.client.responses.create(
            model="gpt-4o-mini",
            instructions="You are a coding assistant that talks like a pirate.",
            input=prompt,
        )
        return {'choices': [{'text': response.output_text}]}

class GeminiFlash2Client:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, max_tokens: int):
        # Implement the API call for Gemini Flash 2
        response = self.client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=prompt
        )
        return {'choices': [{'text': response.text}]}

class AnthropicClient:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, max_tokens: int):
        message = self.client.messages.create(
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": str(prompt),
                }
            ],
            model="claude-3-5-sonnet-latest",
        )
        print("MESSAGE: ", message.content[0].text)
        return {'choices': [{'text': message.content[0].text}]}

def main():
    input_questions = [
        'country', 'age', 'sex', 'education', 'marital_status',
        'employment_status', 'urban_rural', 'income', 'religion', 'ethnicity']

    # Load data once to get all possible target questions
    data_path = 'evaluation/merged_wvs_gss.json'
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Extract all unique target questions
    all_target_questions = set(q['question_id'] for respondent in data for q in respondent['data'] if q['use_case'] == 'value')

    results = {}

    # Initialize API clients
    gpt4o_client = GPT4oClient(api_key=openai.api_key)
    gpt4o_mini_client = GPT4oMiniClient(api_key=openai.api_key)
    gemini_flash2_client = GeminiFlash2Client(api_key=os.getenv('GEMINI_API_KEY'))
    anthropic_client = AnthropicClient(api_key=anthropic_api_key)

    # Run benchmarks for each target question
    for target_question in all_target_questions:
        print(f"\nRunning benchmarks for target question: {target_question}")
        benchmark_instance = SurveyBenchmark(
            data_path=data_path,
            input_questions=input_questions,
            target_questions=[target_question]
        )

        # Run benchmarks for API-based models

        print("\nBenchmarking Gemini Flash 2...")
        results[f'gemini-flash-2_{target_question}'] = benchmark_instance.run_api_benchmark("Gemini Flash 2", gemini_flash2_client, clean_output)
        print(f"Accuracy: {results[f'gemini-flash-2_{target_question}']['accuracy']:.2f}")

        print("\nBenchmarking Anthropic...")
        results[f'anthropic_{target_question}'] = benchmark_instance.run_api_benchmark("Anthropic", anthropic_client, clean_output)
        print(f"Accuracy: {results[f'anthropic_{target_question}']['accuracy']:.2f}")

        print("\nBenchmarking GPT-4o...")
        results[f'gpt-4o_{target_question}'] = benchmark_instance.run_api_benchmark("GPT-4o", gpt4o_client, clean_output)
        print(f"Accuracy: {results[f'gpt-4o_{target_question}']['accuracy']:.2f}")

        print("\nBenchmarking GPT-4o Mini...")
        results[f'gpt-4o-mini_{target_question}'] = benchmark_instance.run_api_benchmark("GPT-4o Mini", gpt4o_mini_client, clean_output)
        print(f"Accuracy: {results[f'gpt-4o-mini_{target_question}']['accuracy']:.2f}")

        # Run benchmarks for open-source models
        print("\nBenchmarking google/gemma-3-4b-it...")
        results[f'gemma-3-4b-it_{target_question}'] = benchmark_instance.run_hf_benchmark_single_model("google/gemma-3-4b-it", clean_output)
        print(f"Accuracy: {results[f'gemma-3-4b-it_{target_question}']['accuracy']:.2f}")

        print("\nBenchmarking Meta-Llama-3.1-8B-Instruct...")
        results[f'meta_llama_{target_question}'] = benchmark_instance.run_hf_benchmark_single_model("meta-llama/Llama-3.1-8B-Instruct", clean_output)
        print(f"Accuracy: {results[f'meta_llama_{target_question}']['accuracy']:.2f}")

        print("\nBenchmarking mistralai/Mistral-7B-Instruct-v0.2...")
        results[f'mistral_{target_question}'] = benchmark_instance.run_hf_benchmark_single_model("mistralai/Mistral-7B-Instruct-v0.2", clean_output)
        print(f"Accuracy: {results[f'mistral_{target_question}']['accuracy']:.2f}")

        # Save results for the current target question
        output_path = Path(f'benchmark_results_{target_question}.json')
        with open(output_path, 'w') as f:
            json.dump({k: v for k, v in results.items() if target_question in k}, f, indent=2)
        print(f"\nResults for {target_question} saved to {output_path}")


if __name__ == "__main__":
    main()
