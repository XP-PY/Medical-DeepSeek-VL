from src import env
import os
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizerFast
from typing import Tuple, Callable, Any

git_repo_path = Path(__file__).parent.parent / "GitRepo/DeepSeek-VL2"
sys.path.insert(0, str(git_repo_path))
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM  # type: ignore
from deepseek_vl2.utils.io import load_pil_images   # type: ignore

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