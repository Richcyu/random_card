#!/usr/bin/env python3
"""Non-production, two-phrase CPU comparison of commercially usable MT models."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer


PHRASES = ("abyssal glow", "abyssal trench map")
DIRECT = (
    ("ko_opus_baseline", "ko", "Helsinki-NLP/opus-mt-tc-big-en-ko", None),
    ("ko_hplt", "ko", "Neurora/opus-hplt-en-ko-v2.0", None),
    ("zh_hans_opus_baseline", "zh-Hans", "Helsinki-NLP/opus-mt-en-zh", ">>cmn_Hans<<"),
    ("zh_hant_opus_baseline", "zh-Hant", "Helsinki-NLP/opus-mt-en-zh", ">>cmn_Hant<<"),
    ("zh_hant_hplt", "zh-Hant", "HPLT/translate-en-zh_hant-v1.0-hplt_opus", None),
    ("ja_opus_baseline", "ja", "Helsinki-NLP/opus-mt-en-jap", None),
    ("ja_fugumt", "ja", "staka/fugumt-en-ja", None),
    ("ja_hplt", "ja", "Neurora/opus-hplt-en-ja-v2.0", None),
    ("ru_opus_baseline", "ru", "Helsinki-NLP/opus-mt-en-ru", None),
)


def translate_direct(model_id: str, prefix: str | None) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()
    texts = [f"{prefix} {phrase}" if prefix else phrase for phrase in PHRASES]
    with torch.inference_mode():
        output = model.generate(**tokenizer(texts, return_tensors="pt", padding=True, truncation=True), num_beams=4, max_new_tokens=64)
    return [text.strip() for text in tokenizer.batch_decode(output, skip_special_tokens=True)]


def translate_m2m_ru() -> list[str]:
    model_id = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_id)
    tokenizer.src_lang = "en"
    model = M2M100ForConditionalGeneration.from_pretrained(model_id)
    model.eval()
    with torch.inference_mode():
        output = model.generate(**tokenizer(list(PHRASES), return_tensors="pt", padding=True), forced_bos_token_id=tokenizer.get_lang_id("ru"), num_beams=4, max_new_tokens=64)
    return [text.strip() for text in tokenizer.batch_decode(output, skip_special_tokens=True)]


def main() -> None:
    results: list[dict[str, object]] = []
    for label, locale, model_id, prefix in DIRECT:
        try:
            values = translate_direct(model_id, prefix)
            results.append({"label": label, "locale": locale, "model_id": model_id, "phrases": list(PHRASES), "translations": values})
        except Exception as error:  # retain evidence without treating an unavailable candidate as success
            results.append({"label": label, "locale": locale, "model_id": model_id, "error": f"{type(error).__name__}: {error}"})
    try:
        results.append({"label": "ru_m2m100", "locale": "ru", "model_id": "facebook/m2m100_418M", "phrases": list(PHRASES), "translations": translate_m2m_ru()})
    except Exception as error:
        results.append({"label": "ru_m2m100", "locale": "ru", "model_id": "facebook/m2m100_418M", "error": f"{type(error).__name__}: {error}"})
    output = Path("artifacts/candidate-localization-probe/results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema_version": "candidate-localization-probe.v1", "production": False, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"production": False, "candidate_count": len(results)}))


if __name__ == "__main__":
    main()
