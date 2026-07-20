from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenCountContract:
    actual_batch_rows: int
    actual_seq_len: int
    total_token_slots: int
    valid_token_count: int | None
    padding_token_count: int | None
    token_count_status: str
    deprecated_padded_token_count: int | None
    deprecated_padded_token_count_unit: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "actual_batch_rows": int(self.actual_batch_rows),
            "actual_seq_len": int(self.actual_seq_len),
            "total_token_slots": int(self.total_token_slots),
            "valid_token_count": self.valid_token_count,
            "padding_token_count": self.padding_token_count,
            "token_count_status": str(self.token_count_status),
            "padded_token_count": self.deprecated_padded_token_count,
            "padded_token_count_deprecated": True,
            "padded_token_count_unit": str(self.deprecated_padded_token_count_unit),
            "valid_non_padding_token_count": self.valid_token_count,
        }


def compute_token_count_contract(
    *,
    actual_batch_rows: int,
    actual_seq_len: int,
    attention_mask_sum: int | None,
) -> TokenCountContract:
    total = int(actual_batch_rows) * int(actual_seq_len)
    if attention_mask_sum is None:
        return TokenCountContract(
            actual_batch_rows=int(actual_batch_rows),
            actual_seq_len=int(actual_seq_len),
            total_token_slots=int(total),
            valid_token_count=None,
            padding_token_count=None,
            token_count_status="unavailable",
            deprecated_padded_token_count=None,
            deprecated_padded_token_count_unit="padding_token_count_when_attention_mask_available",
        )
    valid = int(attention_mask_sum)
    padding = int(total - valid)
    status = "measured" if valid >= 0 and padding >= 0 and valid + padding == total else "invalid"
    return TokenCountContract(
        actual_batch_rows=int(actual_batch_rows),
        actual_seq_len=int(actual_seq_len),
        total_token_slots=int(total),
        valid_token_count=int(valid),
        padding_token_count=int(padding),
        token_count_status=status,
        deprecated_padded_token_count=int(padding) if status == "measured" else None,
        deprecated_padded_token_count_unit="padding_token_count",
    )


__all__ = ["TokenCountContract", "compute_token_count_contract"]
