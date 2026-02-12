import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import utils
import fitz  # type: ignore
import torch
from PIL import Image

MODEL_ID = "deepseek-ai/deepseek-vl2-tiny"

def pdf_page_to_image(pdf_path: str, page_idx: int = 0, dpi: int = 200) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

@torch.inference_mode()
def main():
    pdf_path = "demo/sample.pdf"   # put any medical paper pdf here
    img = pdf_page_to_image(pdf_path, 0)

    vl_chat_processor, tokenizer, vl_gpt, load_pil_images = utils.load_model(MODEL_ID)
    
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\nWhat is the title of this document?",
            # "images": ["./images/visual_grounding_1.jpeg"],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # load images and prepare for inputs
    # pil_images = load_pil_images(conversation)
    pil_images = [img]
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = vl_gpt.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
    print(f"{prepare_inputs['sft_format'][0]}", answer)

if __name__ == "__main__":
    main()
