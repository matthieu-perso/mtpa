import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Protocol

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
from google import genai


class ModelRunner(Protocol):
    def generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0) -> str: ...
    def score_mc(self, question: str, options: List[str], preface: str = "") -> Tuple[int, List[float]]: ...
    @property
    def name(self) -> str: ...


@dataclass
class HFConfig:
    model_id: str
    torch_dtype: Optional[str] = None  # 'bf16', 'fp16', 'fp32'
    device_map: str = "auto"
    load_in_8bit: bool = False

def _prepend_preface(preface: str, text: str) -> str:
    if not preface:
        return text
    p = str(preface).strip()
    if not p.endswith("\n"):
        p += "\n\n"
    return p +  "[PROMPT]" +text


class HFRunner:
    def __init__(self, cfg: HFConfig):
        self._name = cfg.model_id
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            None: None,
        }
        torch_dtype = dtype_map.get(cfg.torch_dtype, None)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
        model_kwargs = dict(trust_remote_code=True)
        if cfg.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = cfg.device_map
        else:
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
            model_kwargs["device_map"] = cfg.device_map
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)
        self.model.eval()

    @property
    def name(self) -> str:
        return self._name

    def generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=max(temperature, 1e-6),
                top_p=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(prompt):].strip()

    def _option_logprob(self, prompt: str, option_text: str) -> float:
        device = next(self.model.parameters()).device
        with torch.no_grad():
            full = prompt + option_text
            inputs = self.tokenizer(full, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            prompt_len = prompt_ids.shape[1]
            option_len = shift_labels.shape[1] - (prompt_len - 1)
            option_logits = shift_logits[:, prompt_len - 1 : prompt_len - 1 + option_len, :]
            option_labels = shift_labels[:, prompt_len - 1 : prompt_len - 1 + option_len]
            logprobs = torch.log_softmax(option_logits, dim=-1)
            token_logprobs = logprobs.gather(-1, option_labels.unsqueeze(-1)).squeeze(-1)
            return token_logprobs.mean().item()

    def score_mc(self, question: str, options: List[str], preface: str = "") -> Tuple[int, List[float]]:
        user = f"{question}\nAnswer: "
        prompt = _prepend_preface(preface, user)    
        scores = [self._option_logprob(prompt, o) for o in options]
        best_idx = int(torch.tensor(scores).argmax().item())
        return best_idx, scores



@dataclass
class OpenAIConfig:
    model: str


class OpenAIRunner:
    def __init__(self, cfg: OpenAIConfig):
        self._name = cfg.model
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = cfg.model

    @property
    def name(self) -> str:
        return self._name

    def generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0) -> str:
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    def score_mc(self, question: str, options: List[str], preface: str = "") -> Tuple[int, List[float]]:
        letters = [chr(ord('A') + i) for i in range(len(options))]
        forced_choice = "\n".join([f"{letters[i]}. {opt}" for i, opt in enumerate(options)])
        core = (
            f"{question}\n{forced_choice}\n"
            "Respond with a single letter (A, B, C, ...) only."
        )
        prompt = _prepend_preface(preface, core)    
        txt = self.generate(prompt, max_new_tokens=2, temperature=0.0)
        choice = txt.strip().upper()[:1]
        try:
            idx = letters.index(choice)
        except ValueError:
            idx = 0
        scores = [1.0 if i == idx else 0.0 for i in range(len(options))]
        return idx, scores



@dataclass
class GeminiConfig:
    model: str


class GeminiRunner:
    def __init__(self, cfg: GeminiConfig):
        self._name = cfg.model
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = cfg.model

    @property
    def name(self) -> str:
        return self._name

    def generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0) -> str:
        resp = self.client.models.generate_content(model=self.model, contents=prompt)
        return getattr(resp, "text", "").strip()

    def score_mc(self, question: str, options: List[str], preface: str = "") -> Tuple[int, List[float]]:
        letters = [chr(ord('A') + i) for i in range(len(options))]
        forced_choice = "\n".join([f"{letters[i]}. {opt}" for i, opt in enumerate(options)])
        core = (
            f"{question}\n{forced_choice}\n"
            "Respond with a single letter (A, B, C, ...) only."
        )
        prompt = _prepend_preface(preface, core)   
        txt = self.generate(prompt, max_new_tokens=2, temperature=0.0)
        choice = txt.strip().upper()[:1]
        try:
            idx = letters.index(choice)
        except ValueError:
            idx = 0
        scores = [1.0 if i == idx else 0.0 for i in range(len(options))]
        return idx, scores



class ModelRegistry:
    def __init__(self):
        self._factories: Dict[str, callable] = {}

    def register_hf(self, name: str, model_id: str, torch_dtype: Optional[str] = None, device_map: str = "auto", load_in_8bit: bool = False):
        def factory():
            return HFRunner(HFConfig(model_id=model_id, torch_dtype=torch_dtype, device_map=device_map, load_in_8bit=load_in_8bit))
        self._factories[name] = factory

    def register_openai(self, name: str, model: str):
        def factory():
            return OpenAIRunner(OpenAIConfig(model=model))
        self._factories[name] = factory

    def register_gemini(self, name: str, model: str):
        def factory():
            return GeminiRunner(GeminiConfig(model=model))
        self._factories[name] = factory

    def create(self, name: str) -> ModelRunner:
        if name in self._factories:
            return self._factories[name]()
        # If not registered but looks like HF model id, try direct HF
        if "/" in name:
            return HFRunner(HFConfig(model_id=name))
        raise ValueError(f"Unknown model: {name}")

    def list_models(self) -> List[str]:
        return list(self._factories.keys()) 