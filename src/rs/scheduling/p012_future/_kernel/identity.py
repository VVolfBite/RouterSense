from __future__ import annotations

import re

_SAFE = re.compile(r"[^a-z0-9_.-]+")


def normalize_model_slug(value: str) -> str:
    slug = _SAFE.sub("-", str(value).strip().lower()).strip("-.")
    if not slug:
        raise ValueError("model slug must be non-empty")
    return slug


def canonical_instance_id(model_slug: str, instance_id: str) -> str:
    slug = normalize_model_slug(model_slug)
    raw = str(instance_id).strip()
    if not raw:
        raise ValueError("instance_id must be non-empty")
    prefix = f"{slug}:"
    return raw if raw.startswith(prefix) else prefix + raw


def legacy_instance_id(model_slug: str, instance_id: str) -> str:
    slug = normalize_model_slug(model_slug)
    raw = str(instance_id).strip()
    prefix = f"{slug}:"
    return raw[len(prefix):] if raw.startswith(prefix) else raw


__all__ = ["canonical_instance_id", "legacy_instance_id", "normalize_model_slug"]
