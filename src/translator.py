import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
LANGUAGE_MAP = {
    "ar": "arb_Arab",
    "ko": "kor_Hang",
    "te": "tel_Telu",
}


class NLLBTranslator:

    def __init__(self, model_name=MODEL_NAME):

        if torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
        )

        self.model.to(self.device)
        self.model.eval()

    def translate_batch(self, texts, source_lang, batch_size=8):
        self.tokenizer.src_lang = LANGUAGE_MAP[source_lang]
        translations = []

        for start in tqdm(
            range(0, len(texts), batch_size),
            desc=f"Translating {source_lang}",
        ):
            batch = texts[start : start + batch_size]

            inputs = self.tokenizer(
                [str(text).strip() for text in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(
                        "eng_Latn"
                    ),
                    do_sample=False,
                    num_beams=1,
                )

            translations.extend(
                self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True,
                )
            )

        return [text.strip() for text in translations]

    def translate(self, text, source_lang):
        return self.translate_batch([text], source_lang)[0]
