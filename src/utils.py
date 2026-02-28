from src import env
import os
import sys
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaTokenizerFast
from typing import Tuple, Callable, Any
from PIL import Image
from rapidfuzz import fuzz
import re

git_repo_path = Path(__file__).parent.parent / "GitRepo/DeepSeek-VL2"
sys.path.insert(0, str(git_repo_path))
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM  # type: ignore
from deepseek_vl2.utils.io import load_pil_images   # type: ignore

class JsonlVLDataset(Dataset):
    '''Load train/val datasets'''
    def __init__(self, jsonl_path: str):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["image"]).convert("RGB")
        return {
            "image_pil": img,
            "prompt": r["prompt"],
            "response": r["response"],
            "id": r.get("id", str(idx)),
        }

def _to_rgb(im: Image.Image) -> Image.Image:
    return im.convert("RGB") if isinstance(im, Image.Image) and im.mode != "RGB" else im

def load_model(MODEL_ID: str = "deepseek-ai/deepseek-vl2-tiny") -> Tuple[
    DeepseekVLV2Processor,      # Processor
    LlamaTokenizerFast,         # tokenizer
    DeepseekVLV2ForCausalLM,    # Model
    Callable[[list], list]      # load_pil_images Function
]:
    MODEL_ID = os.path.join(env.MODEL_DIR, MODEL_ID)
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(MODEL_ID)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    return vl_chat_processor, tokenizer, vl_gpt, load_pil_images

# =====================================================================================
#                               Evaluation Metric
# =====================================================================================
def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\.\-/%]", "", s)
    return s

def parse_letter(pred: str):
    # Pick the first standalone A/B/C/D
    pred = pred.strip().upper()
    for ch in ["A", "B", "C", "D"]:
        if pred == ch:
            return ch
    # fallback: search anywhere
    import re
    m = re.search(r"\b([ABCD])\b", pred)
    return m.group(1) if m else None

def em(pred: str, answer: str) -> int:
    return int(normalize(pred) == normalize(answer))

def f1_fuzzy(pred: str, answer: str) -> float:
    return fuzz.token_set_ratio(normalize(pred), normalize(answer)) / 100.0

def acc(pred: str, answer: str) -> int:
    pred_choice = parse_letter(pred)
    answer = answer.strip().upper()
    correct = int(pred_choice == answer)
    return correct

# =====================================================================================
#                               custom_processor
# =====================================================================================

class custom_processor:
    def __init__(self, vl_chat_processor):
        self.vl_chat_processor = vl_chat_processor
        self.tokenizer = vl_chat_processor.tokenizer

        self.pad_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        
    def __call__(self, images, text, vl_gpt):
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{text}",
                # "images": ["./images/visual_grounding_1.jpeg"],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=images,
            force_batchify=True,
            system_prompt=""
        ).to(vl_gpt.device)

        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        return inputs_embeds, prepare_inputs, prepare_inputs.attention_mask
    
def load_model_processor(MODEL_ID: str = "deepseek-ai/deepseek-vl2-tiny", dtype = torch.bfloat16):
    MODEL_ID = os.path.join(env.MODEL_DIR, MODEL_ID)
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(MODEL_ID)
    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    vl_gpt = vl_gpt.to(dtype).cuda()
    processor = custom_processor(vl_chat_processor)
    return vl_gpt, processor

# =====================================================================================
#                                   Wrapper
# =====================================================================================

class DeepSeekVL2Wrapper:
    def __init__(self, model_dir: str, dtype=torch.bfloat16, device="cuda"):
        """
        model_dir: local path like /path/to/deepseek-ai/deepseek-vl2-tiny
        """
        self.device = device

        self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_dir)
        self.tokenizer = self.processor.tokenizer

        self.model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True, 
            torch_dtype=dtype
        ).to(device).eval()

        # IDs
        self.pad_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    @torch.inference_mode()
    def infer_one(self, image, text: str, max_new_tokens: int = 64) -> str:
        """
        image: PIL.Image OR list[PIL.Image]
        text: question/prompt
        """
        # # Ensure list of PIL images
        # if image is None:
        #     images = []
        #     content = text
        # else:
        #     if not isinstance(image, (list, tuple)):
        #         images = [image]
        #     else:
        #         images = list(image)

        #     # Force RGB for all images (fixes 4-channel crash)
        #     images = [_to_rgb(im) for im in images]
        #     content = f"<image>\n{text}"

        # conversation = [
        #     {
        #         "role": "<|User|>",
        #         "content": content,
        #         # "images" is not required here because we pass images separately
        #     },
        #     {"role": "<|Assistant|>", "content": ""},
        # ]

        images = [_to_rgb(image)]

        conversation = [
            {"role": "<|User|>", "content": f"{text}"},
            # {"role": "<|User|>", "content": f"<image>\n{text}"},
            {"role": "<|Assistant|>", "content": ""},
        ]

        prepare_inputs = self.processor(
            conversations=conversation,
            images=images,
            force_batchify=True,
            system_prompt=""
        ).to(self.model.device)

        # This is the DeepSeek-VL2 special step
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.model.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        # Decode full output (includes prompt tokens)
        decoded = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)

        # Usually DeepSeek returns something like:
        # "<|Assistant|> ... <|end|>" depending on tokenizer config.
        # We'll extract assistant part robustly:
        if "<|Assistant|>" in decoded:
            decoded = decoded.split("<|Assistant|>", 1)[-1]
        # remove any trailing special tokens if present
        decoded = decoded.replace(self.tokenizer.eos_token, "").strip()

        return decoded

def load_model_wrapper(MODEL_ID="deepseek-ai/deepseek-vl2-tiny", dtype=torch.bfloat16, device="cuda"):
    model_dir = os.path.join(env.MODEL_DIR, MODEL_ID)  # keep your env logic
    return DeepSeekVL2Wrapper(model_dir=model_dir, dtype=dtype, device=device)

# =====================================================================================
#                                   LoRA Wrapper
# =====================================================================================

class DeepSeekVL2LoRAWrapper:
    def __init__(self, base_model_path: str, lora_path: str, dtype=torch.bfloat16, device="cuda"):
        self.processor = DeepseekVLV2Processor.from_pretrained(base_model_path)
        self.tokenizer = self.processor.tokenizer

        base: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            trust_remote_code=True, 
            torch_dtype=dtype, 
            device_map=None
        ).to(device).eval()

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base, lora_path).to(device).eval()

        self.pad_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    @torch.inference_mode()
    def infer_one(self, image, text: str, max_new_tokens: int = 64) -> str:
        images = [_to_rgb(image)]

        conversation = [
            {"role": "<|User|>", "content": f"{text}"},
            # {"role": "<|User|>", "content": f"<image>\n{text}"},
            {"role": "<|Assistant|>", "content": ""},
        ]
        prepare_inputs = self.processor(
            conversations=conversation, images=images, force_batchify=True, system_prompt=""
        ).to(self.model.device)

        inputs_embeds = self.model.base_model.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.model.base_model.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True
        )

        decoded = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
        if "<|Assistant|>" in decoded:
            decoded = decoded.split("<|Assistant|>", 1)[-1]
        return decoded.replace(self.tokenizer.eos_token, "").strip()

def load_model_loRAwrapper(MODEL_ID="deepseek-ai/deepseek-vl2-tiny", lora_path=None, dtype=torch.bfloat16, device="cuda"):
    assert lora_path is not None
    base_model_path = os.path.join(env.MODEL_DIR, MODEL_ID)
    return DeepSeekVL2LoRAWrapper(base_model_path=base_model_path, lora_path=lora_path, dtype=dtype, device=device)