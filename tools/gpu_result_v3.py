"""Strict, provider-neutral reader for exact GPU v3 result checkpoints.

Callers supply workload identity; campaign totals and benchmark comparisons
remain outside this module. This file can also travel with standalone gates.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
from typing import Any


class ValidationError(ValueError, RuntimeError):
    """Invalid input; also preserves the campaign reducer's RuntimeError API."""
    pass


SHA256 = re.compile(r"[0-9a-f]{64}")
COMMON_INTEGER_FIELDS = (
    "records",
    "labelled_weight",
    "kernels",
    "covered_weight",
    "verified",
    "direct_comparisons",
    "contribution",
    "minimum_free_bytes",
)
PROVENANCE_FIELDS = (
    "solver_binary_sha256",
    "solver_configuration_sha256",
    "canonical_cache_sha256",
    "orbit_corpus_sha256",
)



def parse_nonnegative(text: str, context: str) -> int:
    if not text.isdecimal():
        raise ValidationError(f"invalid nonnegative integer for {context}: {text!r}")
    return int(text)



def read_result(path: Path, *, magic: str, identity: dict[str, str],
                geometry: str, transpose_quotient: bool = False) -> dict[str, Any]:
    lines = path.read_bytes().splitlines(keepends=True)
    expected_header = f"{magic} 3\n".encode()
    if not lines or lines[0] != expected_header:
        raise ValidationError(f"invalid result header: {path}")
    fields: dict[str, Any] = {}
    payload = bytearray()
    claimed_checksum: str | None = None
    for line_number, raw_line in enumerate(lines[1:], 2):
        if not raw_line.endswith(b"\n"):
            raise ValidationError(f"unterminated result line {line_number}: {path}")
        try:
            words = raw_line[:-1].decode("ascii").split()
        except UnicodeDecodeError as error:
            raise ValidationError(f"non-ASCII result field: {path}") from error
        if len(words) != 2:
            raise ValidationError(f"invalid result field on line {line_number}: {path}")
        key, value = words
        if key == "result_payload_sha256":
            if claimed_checksum is not None or line_number != len(lines):
                raise ValidationError(f"misplaced or duplicate result checksum: {path}")
            claimed_checksum = value
            continue
        if key in fields:
            raise ValidationError(f"duplicate result field {key}: {path}")
        fields[key] = value
        payload.extend(raw_line)
    actual_checksum = hashlib.sha256(payload).hexdigest()
    if claimed_checksum != actual_checksum:
        raise ValidationError(f"result checksum mismatch: {path}")

    for key, expected in identity.items():
        if fields.get(key) != expected:
            raise ValidationError(f"result identity mismatch for {key}: {path}")
    if fields.get("geometry") != geometry or fields.get(
        "token_plane_quotient"
    ) != "1":
        raise ValidationError(f"unsupported exact representation: {path}")
    if transpose_quotient and fields.get("transpose_quotient") != "1":
        raise ValidationError(f"missing mandatory transpose quotient: {path}")
    for key in PROVENANCE_FIELDS:
        if not SHA256.fullmatch(fields.get(key, "")):
            raise ValidationError(f"invalid provenance field {key}: {path}")
    for key in COMMON_INTEGER_FIELDS:
        if key not in fields:
            raise ValidationError(f"missing result field {key}: {path}")
        fields[key] = parse_nonnegative(fields[key], f"{path}:{key}")
    if fields["kernels"] != fields["records"]:
        raise ValidationError(f"record/kernel mismatch: {path}")
    for key, value in tuple(fields.items()):
        if key.endswith("_seconds"):
            try:
                converted = float(value)
            except ValueError as error:
                raise ValidationError(f"invalid timing field {key}: {path}") from error
            if converted < 0:
                raise ValidationError(f"negative timing field {key}: {path}")
            if not math.isfinite(converted):
                raise ValidationError(f"non-finite timing field {key}: {path}")
            fields[key] = converted
    for key in ("gpu_seconds", "total_seconds"):
        if not isinstance(fields.get(key), float):
            raise ValidationError(f"missing timing field {key}: {path}")
    fields["result_payload_sha256"] = actual_checksum
    return fields
