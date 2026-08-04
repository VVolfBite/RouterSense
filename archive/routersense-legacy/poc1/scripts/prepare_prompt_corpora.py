from __future__ import annotations

import json
import random
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw'
PROMPT_DIR = ROOT / 'data' / 'prompts'
PROMPT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = '/root/autodl-tmp/models/OLMoE-1B-7B-0924'
MIN_TOKENS = 300
MAX_RECORDS = 50
SEED = 42


def load_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def dump_jsonl(rows, path: Path) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    rng = random.Random(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    corpora = {}

    # code: HumanEval is already long enough for many entries
    code_rows = []
    for row in load_jsonl(RAW_DIR / 'humaneval_test.jsonl'):
        text = row['text']
        length = len(tok(text, add_special_tokens=False)['input_ids'])
        if length >= MIN_TOKENS:
            code_rows.append({'document_id': len(code_rows), 'text': text, 'source': 'humaneval', 'tokens': length})
    corpora['code_prompts_50.jsonl'] = code_rows[:MAX_RECORDS]

    # math: GSM8K alone is too short; concatenate neighboring problems to build long math contexts
    gsm_rows = list(load_jsonl(RAW_DIR / 'gsm8k_test.jsonl'))
    math_rows = []
    for idx in range(len(gsm_rows) - 2):
        texts = []
        for offset in range(3):
            row = gsm_rows[idx + offset]
            texts.append(f"Question {offset+1}:\n{row['question']}\n\nAnswer {offset+1}:\n{row['answer']}")
        text = '\n\n' + ('\n\n---\n\n'.join(texts))
        length = len(tok(text, add_special_tokens=False)['input_ids'])
        if length >= MIN_TOKENS:
            math_rows.append({'document_id': len(math_rows), 'text': text, 'source': 'gsm8k_triplet', 'tokens': length})
    corpora['math_prompts_50.jsonl'] = math_rows[:MAX_RECORDS]

    # dialog: choose long English assistant trees/messages only
    dialog_candidates = []
    for row in load_jsonl(RAW_DIR / 'oasst1_train.jsonl'):
        if row.get('lang') != 'en':
            continue
        text = str(row.get('text', '')).strip()
        if not text:
            continue
        length = len(tok(text, add_special_tokens=False)['input_ids'])
        if length >= MIN_TOKENS:
            dialog_candidates.append({'document_id': 0, 'text': text, 'source': 'oasst1', 'tokens': length})
    rng.shuffle(dialog_candidates)
    dialog_rows = []
    for row in dialog_candidates[:MAX_RECORDS]:
        payload = dict(row)
        payload['document_id'] = len(dialog_rows)
        dialog_rows.append(payload)
    corpora['dialog_prompts_50.jsonl'] = dialog_rows

    # wiki fallback: use long theory prompts + gsm8k / humaneval are not wiki; build a neutral expository corpus from local theory prompts
    theory_rows = list(load_jsonl(PROMPT_DIR / 'theory_prompts_50_long.jsonl'))
    wiki_rows = []
    for idx in range(len(theory_rows) - 2):
        text = '\n\n'.join(theory_rows[idx + offset]['text'] for offset in range(3))
        length = len(tok(text, add_special_tokens=False)['input_ids'])
        if length >= MIN_TOKENS:
            wiki_rows.append({'document_id': len(wiki_rows), 'text': text, 'source': 'local_theory_triplet', 'tokens': length})
    corpora['wiki_prompts_50.jsonl'] = wiki_rows[:MAX_RECORDS]

    summary = {}
    for filename, rows in corpora.items():
        path = PROMPT_DIR / filename
        dump_jsonl(rows, path)
        summary[filename] = {
            'path': str(path),
            'count': len(rows),
            'min_tokens': min((row['tokens'] for row in rows), default=None),
            'median_tokens': sorted(row['tokens'] for row in rows)[len(rows)//2] if rows else None,
            'max_tokens': max((row['tokens'] for row in rows), default=None),
        }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
