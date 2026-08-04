from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)


def dump_jsonl(rows, path: Path) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def download_gsm8k() -> Path:
    ds = load_dataset('openai/gsm8k', 'main', split='test')
    rows = []
    for idx, row in enumerate(ds):
        rows.append({
            'document_id': idx,
            'question': row['question'],
            'answer': row['answer'],
            'text': f"Question:\n{row['question']}\n\nAnswer:\n{row['answer']}",
            'source': 'gsm8k',
        })
    path = RAW_DIR / 'gsm8k_test.jsonl'
    dump_jsonl(rows, path)
    return path


def download_humaneval() -> Path:
    ds = load_dataset('openai/openai_humaneval', split='test')
    rows = []
    for idx, row in enumerate(ds):
        rows.append({
            'document_id': idx,
            'task_id': row['task_id'],
            'prompt': row['prompt'],
            'canonical_solution': row['canonical_solution'],
            'test': row['test'],
            'text': f"{row['prompt']}\n\n# Reference solution\n{row['canonical_solution']}\n\n# Tests\n{row['test']}",
            'source': 'humaneval',
        })
    path = RAW_DIR / 'humaneval_test.jsonl'
    dump_jsonl(rows, path)
    return path


def download_oasst1() -> Path:
    ds = load_dataset('OpenAssistant/oasst1', split='train')
    rows = []
    for idx, row in enumerate(ds):
        rows.append({
            'document_id': idx,
            'message_id': row['message_id'],
            'parent_id': row['parent_id'],
            'text': row['text'],
            'role': row['role'],
            'lang': row['lang'],
            'message_tree_id': row['message_tree_id'],
            'source': 'oasst1',
        })
    path = RAW_DIR / 'oasst1_train.jsonl'
    dump_jsonl(rows, path)
    return path


def main() -> None:
    outputs = {
        'gsm8k': str(download_gsm8k()),
        'humaneval': str(download_humaneval()),
        'oasst1': str(download_oasst1()),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == '__main__':
    main()
