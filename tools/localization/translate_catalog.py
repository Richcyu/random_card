#!/usr/bin/env python3
"""Offline, CPU-only catalog localization for GitHub Actions standard runners.

The script reads the immutable English master and writes only under an output
directory. It resumes from valid saved localization rows and never sends text
to an API.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


EXPECTED_MASTER_SHA256 = "46036a854b5889132a4f9699f3918d9c577bcc1b4a95e5f0c5ecc8680c866952"
MASTER_SCHEMA = "phase7.english-master.v1"
TARGET_LOCALES = ("ko", "zh-Hans", "zh-Hant", "ja", "ru")
PLACEHOLDER_RE = re.compile(r"(?:\$\{|\bTODO\b|\bTRANSLATION\b)", re.IGNORECASE)
TARGET_SCRIPTS = {
    "ko": re.compile(r"[\uac00-\ud7a3]"),
    "zh-Hans": re.compile(r"[\u4e00-\u9fff]"),
    "zh-Hant": re.compile(r"[\u4e00-\u9fff]"),
    "ja": re.compile(r"[\u3040-\u30ff\u3400-\u9fff]"),
    "ru": re.compile(r"[\u0400-\u052f]"),
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    chinese_target_token: str | None = None


MODELS = {
    "ko": ModelSpec("Helsinki-NLP/opus-mt-tc-big-en-ko"),
    "zh-Hans": ModelSpec("Helsinki-NLP/opus-mt-en-zh", ">>cmn_Hans<<"),
    "zh-Hant": ModelSpec("Helsinki-NLP/opus-mt-en-zh", ">>cmn_Hant<<"),
    "ja": ModelSpec("Helsinki-NLP/opus-mt-en-jap"),
    "ru": ModelSpec("Helsinki-NLP/opus-mt-en-ru"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("probe", "test", "batch"), required=True)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--locales", nargs="+", choices=TARGET_LOCALES, default=TARGET_LOCALES)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_master(path: Path) -> list[dict]:
    actual_hash = sha256(path)
    if actual_hash != EXPECTED_MASTER_SHA256:
        raise ValueError(f"English Master SHA-256 mismatch: {actual_hash}")
    with path.open("r", encoding="utf-8") as source:
        master = json.load(source)
    if master.get("schema_version") != MASTER_SCHEMA:
        raise ValueError("Unexpected English Master schema")
    records = master.get("records")
    if not isinstance(records, list) or len(records) != 30000:
        raise ValueError("English Master must contain exactly 30,000 records")
    stable_ids = [record.get("stable_id") for record in records]
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("English Master contains duplicate stable IDs")
    if any(not isinstance(record.get("english_master"), str) or not record["english_master"].strip() for record in records):
        raise ValueError("English Master contains an empty English term")
    return records


def localization_path(output: Path, locale: str) -> Path:
    return output / "localizations" / f"localization_{locale}.json"


def load_rows(path: Path, locale: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    indexed: dict[str, dict] = {}
    for row in payload.get("records", []):
        stable_id = row.get("stable_id")
        text = row.get("localized_text")
        if row.get("locale") != locale or not isinstance(stable_id, str) or not isinstance(text, str):
            raise ValueError(f"Malformed existing {locale} localization row")
        if stable_id in indexed:
            raise ValueError(f"Duplicate existing {locale} stable ID: {stable_id}")
        validate_text(locale, text)
        indexed[stable_id] = row
    return indexed


def validate_text(locale: str, text: str) -> None:
    value = text.strip()
    if not value:
        raise ValueError(f"Empty {locale} translation")
    if PLACEHOLDER_RE.search(value):
        raise ValueError(f"Placeholder found in {locale} translation")
    if not TARGET_SCRIPTS[locale].search(value):
        raise ValueError(f"No target-language script found in {locale} translation: {value!r}")
    letters = sum(character.isalpha() for character in value)
    latin = sum("a" <= character.lower() <= "z" for character in value)
    if letters and latin / letters > 0.8:
        raise ValueError(f"Excessive English leakage in {locale} translation: {value!r}")


def selected_records(records: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.mode == "probe":
        return records[:2]
    if args.mode == "test":
        return records[:50]
    if args.batch_index < 0 or args.batch_size <= 0:
        raise ValueError("batch-index must be non-negative and batch-size must be positive")
    start = args.batch_index * args.batch_size
    return records[start : start + args.batch_size]


def chunks(values: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def translate(locale: str, records: list[dict]) -> list[str]:
    spec = MODELS[locale]
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(spec.model_id)
    model.eval()
    translations: list[str] = []
    with torch.inference_mode():
        for group in chunks(records, 8):
            texts = [record["english_master"].strip() for record in group]
            if spec.chinese_target_token:
                texts = [f"{spec.chinese_target_token} {text}" for text in texts]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            generated = model.generate(**inputs, num_beams=4, max_new_tokens=64)
            translations.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    del model
    del tokenizer
    gc.collect()
    return [value.strip() for value in translations]


def write_rows(path: Path, locale: str, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "card-localization.v1", "locale": locale, "records": [rows[stable_id] for stable_id in sorted(rows)]}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def write_checkpoint(output: Path, master_hash: str, args: argparse.Namespace, all_rows: dict[str, dict[str, dict]]) -> None:
    checkpoint = {
        "schema_version": "card-localization-checkpoint.v1",
        "master_sha256": master_hash,
        "completed_stable_ids": {locale: sorted(rows) for locale, rows in all_rows.items()},
        "last_successful_run": {"run_id": args.run_id, "mode": args.mode, "batch_index": args.batch_index, "batch_size": args.batch_size, "completed_at": datetime.now(timezone.utc).isoformat()},
    }
    target = output / "checkpoint" / "catalog_localization_checkpoint.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(checkpoint, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(target)


def main() -> None:
    args = parse_args()
    records = read_master(args.master)
    chosen = selected_records(records, args)
    if not chosen:
        raise ValueError("Selected batch is outside the English Master range")
    all_rows = {locale: load_rows(localization_path(args.output, locale), locale) for locale in args.locales}
    for locale in args.locales:
        pending = [record for record in chosen if record["stable_id"] not in all_rows[locale]]
        if pending:
            translated = translate(locale, pending)
            if len(translated) != len(pending):
                raise ValueError(f"Unexpected {locale} translation count")
            for record, text in zip(pending, translated, strict=True):
                validate_text(locale, text)
                all_rows[locale][record["stable_id"]] = {"stable_id": record["stable_id"], "locale": locale, "localized_text": text}
            write_rows(localization_path(args.output, locale), locale, all_rows[locale])
    write_checkpoint(args.output, sha256(args.master), args, all_rows)
    print(json.dumps({"mode": args.mode, "records_selected": len(chosen), "rows_written_or_reused": {locale: len(all_rows[locale]) for locale in args.locales}}))


if __name__ == "__main__":
    main()
