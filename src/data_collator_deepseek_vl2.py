import torch
from PIL import Image
from src import utils

class DeepSeekVL2DataCollator:
    """
    Builds DeepSeek-VL2 supervised fine-tuning batches from (image_path, prompt, response).
    Key idea:
      - Build two conversations:
        (A) prompt-only: user + empty assistant  -> to get boundary length
        (B) full:        user + assistant answer -> to get full input_ids
      - labels = full_input_ids with tokens before assistant answer masked to -100
    """

    def __init__(self, vl_processor, pad_token_id: int, device: str = "cuda"):
        self.p = vl_processor
        self.tok = vl_processor.tokenizer
        self.pad_token_id = pad_token_id
        self.device = device

    def _encode_one(self, image: Image.Image, prompt: str, response: str):
        # prompt MUST already contain "<image>\n..." for your doc QA
        conv_prompt = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]
        conv_full = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": response},
        ]

        # processor expects a list of PIL images (even for single image)
        images = [utils._to_rgb(image)]

        enc_prompt = self.p(conversations=conv_prompt, images=images, force_batchify=True, system_prompt="")
        enc_full = self.p(conversations=conv_full, images=images, force_batchify=True, system_prompt="")

        # boundary length: how many tokens belong to prompt-only sequence
        prompt_len = enc_prompt.input_ids.shape[1]

        input_ids = enc_full.input_ids[0]              # (L,)
        attention_mask = enc_full.attention_mask[0]    # (L,)

        # DeepSeek-VL2 extra vision fields
        images_t = enc_full.images[0]                  # (Nimg, C, H, W) or (C,H,W) depending impl
        if images_t.dim() == 3:
            images_t = images_t.unsqueeze(0)            # (1,3,H,W)
        images_seq_mask = enc_full.images_seq_mask[0]   # (L,) or (L, something)
        images_spatial_crop = enc_full.images_spatial_crop[0]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask user/prompt tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_t,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }

    def __call__(self, features):
        encoded = [self._encode_one(f["image_pil"], f["prompt"], f["response"]) for f in features]

        # ---- pad text tokens ----
        max_len = max(x["input_ids"].shape[0] for x in encoded)

        def pad_1d(x, pad_value):
            if x.shape[0] == max_len:
                return x
            pad = torch.full((max_len - x.shape[0],), pad_value, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        batch = {}
        batch["input_ids"] = torch.stack([pad_1d(x["input_ids"], self.pad_token_id) for x in encoded])
        batch["attention_mask"] = torch.stack([pad_1d(x["attention_mask"], 0) for x in encoded])
        batch["labels"] = torch.stack([pad_1d(x["labels"], -100) for x in encoded])

        # ---- pad num_views (V) for images and related fields ----
        # images: (V, 3, 384, 384)
        Vmax = max(x["images"].shape[0] for x in encoded)

        def pad_views_4d(imgs, Vmax):
            # imgs: (V, C, H, W)
            V, C, H, W = imgs.shape
            if V == Vmax:
                return imgs
            pad = torch.zeros((Vmax - V, C, H, W), dtype=imgs.dtype)
            return torch.cat([imgs, pad], dim=0)

        batch["images"] = torch.stack([pad_views_4d(x["images"], Vmax) for x in encoded])

        # images_spatial_crop: usually (V, 4) or (V, something)
        # We'll pad along V with zeros
        crops0 = encoded[0]["images_spatial_crop"]
        if crops0.dim() == 2:
            D = crops0.shape[1]
            def pad_views_2d(t, Vmax):
                V, D2 = t.shape
                assert D2 == D
                if V == Vmax:
                    return t
                pad = torch.zeros((Vmax - V, D), dtype=t.dtype)
                return torch.cat([t, pad], dim=0)
            batch["images_spatial_crop"] = torch.stack([pad_views_2d(x["images_spatial_crop"], Vmax) for x in encoded])
        else:
            # If it's not 2D, just stack (but this is rare)
            batch["images_spatial_crop"] = torch.stack([x["images_spatial_crop"] for x in encoded])

        # images_seq_mask: commonly (V, L) aligned with views and token length
        # Need to pad L (to max_len) and V (to Vmax)
        m0 = encoded[0]["images_seq_mask"]
        if m0.dim() == 2:
            # (V, L)
            def pad_view_token_mask(m, Vmax, Lmax):
                V, L = m.shape
                # pad tokens
                if L < Lmax:
                    pad_tok = torch.zeros((V, Lmax - L), dtype=m.dtype)
                    m = torch.cat([m, pad_tok], dim=1)
                # pad views
                if V < Vmax:
                    pad_view = torch.zeros((Vmax - V, Lmax), dtype=m.dtype)
                    m = torch.cat([m, pad_view], dim=0)
                return m

            batch["images_seq_mask"] = torch.stack([pad_view_token_mask(x["images_seq_mask"], Vmax, max_len) for x in encoded])

        elif m0.dim() == 1:
            # (L,) — pad tokens only
            batch["images_seq_mask"] = torch.stack([pad_1d(x["images_seq_mask"], 0) for x in encoded])
        else:
            # fallback: stack
            batch["images_seq_mask"] = torch.stack([x["images_seq_mask"] for x in encoded])

        return batch

    # def __call__(self, features):
    #     # features: list of dicts from dataset, each has 'image_pil', 'prompt', 'response'
    #     encoded = [self._encode_one(f["image_pil"], f["prompt"], f["response"]) for f in features]

    #     # pad text fields to max length
    #     max_len = max(x["input_ids"].shape[0] for x in encoded)

    #     def pad_1d(x, pad_value):
    #         if x.shape[0] == max_len:
    #             return x
    #         pad = torch.full((max_len - x.shape[0],), pad_value, dtype=x.dtype)
    #         return torch.cat([x, pad], dim=0)

    #     batch = {}
    #     batch["input_ids"] = torch.stack([pad_1d(x["input_ids"], self.pad_token_id) for x in encoded])
    #     batch["attention_mask"] = torch.stack([pad_1d(x["attention_mask"], 0) for x in encoded])
    #     batch["labels"] = torch.stack([pad_1d(x["labels"], -100) for x in encoded])

    #     # pad images_seq_mask if it is 1D aligned with tokens
    #     # (some implementations store it as (L,) aligned with input_ids)
    #     if encoded[0]["images_seq_mask"].dim() == 1:
    #         batch["images_seq_mask"] = torch.stack([pad_1d(x["images_seq_mask"], 0) for x in encoded])
    #     else:
    #         # if it's not 1D, stack directly (usually already same shape)
    #         batch["images_seq_mask"] = torch.stack([x["images_seq_mask"] for x in encoded])

    #     # images and spatial crops usually already uniform across samples
    #     batch["images"] = torch.stack([x["images"] for x in encoded])
    #     batch["images_spatial_crop"] = torch.stack([x["images_spatial_crop"] for x in encoded])

    #     # move to GPU later by Trainer; keep on CPU here
    #     return batch
