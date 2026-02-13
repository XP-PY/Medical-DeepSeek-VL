from src import env
import os
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizerFast
from typing import Tuple, Callable, Any
from PIL import Image

git_repo_path = Path(__file__).parent.parent / "GitRepo/DeepSeek-VL2"
sys.path.insert(0, str(git_repo_path))
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM  # type: ignore
from deepseek_vl2.utils.io import load_pil_images   # type: ignore

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
            trust_remote_code=True
        )
        self.model = self.model.to(dtype).to(device).eval()

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
        # Ensure list of PIL images
        if image is None:
            images = []
            content = text
        else:
            if not isinstance(image, (list, tuple)):
                images = [image]
            else:
                images = list(image)

            # Force RGB for all images (fixes 4-channel crash)
            images = [_to_rgb(im) for im in images]
            content = f"<image>\n{text}"

        conversation = [
            {
                "role": "<|User|>",
                "content": content,
                # "images" is not required here because we pass images separately
            },
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

def _to_rgb(pil_img: Image.Image) -> Image.Image:
    # Handles RGBA, LA, P (palette), etc.
    if pil_img.mode != "RGB":
        return pil_img.convert("RGB")
    return pil_img

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

def load_model_processor(MODEL_ID: str = "deepseek-ai/deepseek-vl2-tiny", dtype = torch.bfloat16):
    MODEL_ID = os.path.join(env.MODEL_DIR, MODEL_ID)
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(MODEL_ID)
    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    vl_gpt = vl_gpt.to(dtype).cuda()
    processor = custom_processor(vl_chat_processor)
    return vl_gpt, processor

def load_model_wrapper(MODEL_ID="deepseek-ai/deepseek-vl2-tiny", dtype=torch.bfloat16, device="cuda"):
    model_dir = os.path.join(env.MODEL_DIR, MODEL_ID)  # keep your env logic
    return DeepSeekVL2Wrapper(model_dir=model_dir, dtype=dtype, device=device)