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

    model, processor = utils.load_model_processor(MODEL_ID)
    prompts = "What is the title of this document?"
    inputs_embeds, prepare_inputs, attention_mask = processor([img], prompts, model)

    # run the model to get the response
    outputs = model.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        pad_token_id=processor.eos_token_id,
        bos_token_id=processor.bos_token_id,
        eos_token_id=processor.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )

    answer = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
    print(f"{prepare_inputs['sft_format'][0]}", answer)

if __name__ == "__main__":
    main()
