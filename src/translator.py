import torch
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

    def translate(self, text, source_lang):

        text = str(text).strip()
        self.tokenizer.src_lang = LANGUAGE_MAP[source_lang]

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_length=20,
            )

        translation = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
        )[0]

        return translation.strip()
