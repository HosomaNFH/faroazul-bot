"""Deterministic parsers for the forecasters' final answers.

Order of attempts: JSON block -> regex on labelled lines. If both fail the
caller may ask a cheap LLM to rewrite the answer as JSON and parse that.
All parsers return ``None`` when they cannot find a complete answer, never a
guess.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

from .cdf import STANDARD_PERCENTILES

_NUMBER = r"[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def to_float(text: Any) -> float | None:
    """Parse a plain number ("1,234.5", "-3e6", "12%"). No word suffixes."""
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        value = float(text)
        return value if math.isfinite(value) else None
    if not isinstance(text, str):
        return None
    cleaned = text.strip().replace("−", "-").replace("%", "").strip()
    cleaned = cleaned.replace(" ", "").replace(" ", "")
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", cleaned):
        cleaned = cleaned.replace(",", "")
    if not re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", cleaned):
        return None
    value = float(cleaned)
    return value if math.isfinite(value) else None


def extract_json_objects(text: str) -> list[Any]:
    """All JSON objects found in ```json fences or as bare {...} spans."""
    found: list[Any] = []
    for block in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S):
        try:
            found.append(json.loads(block))
        except json.JSONDecodeError:
            continue
    if found:
        return found
    # Bare objects: scan for balanced braces.
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                chunk = text[start : i + 1]
                try:
                    found.append(json.loads(chunk))
                except json.JSONDecodeError:
                    pass
                start = None
    return found


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", str(name)).casefold()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[*_`\"]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .:;-")


# --------------------------------------------------------------------------- #
# binary
# --------------------------------------------------------------------------- #


def _last_probability(pattern: re.Pattern, text: str) -> float | None:
    for match in reversed(list(pattern.finditer(text))):
        value = to_float(match.group(1))
        if value is None:
            continue
        if match.group(2) or value > 1.0:
            value /= 100.0
        if 0.0 <= value <= 1.0:
            return value
    return None


def parse_binary(text: str) -> float | None:
    """Return a probability in [0, 1] or None."""
    if not text:
        return None
    # Preferred: a line that starts with "Probability:" (the requested format).
    strict = re.compile(
        r"(?im)^[\s*#>\-]*(?:final\s+)?probability\s*\**\s*[:=]\s*\**\s*("
        + _NUMBER
        + r")\s*(%|percent)?"
    )
    value = _last_probability(strict, text)
    if value is not None:
        return value
    relaxed = re.compile(
        r"probability[^0-9\n%]{0,25}?(" + _NUMBER + r")\s*(%|percent)?",
        flags=re.I,
    )
    value = _last_probability(relaxed, text)
    if value is not None:
        return value
    for obj in reversed(extract_json_objects(text)):
        if isinstance(obj, dict):
            for key in ("probability", "prob", "p", "forecast"):
                if key in obj:
                    value = to_float(obj[key])
                    if value is None:
                        continue
                    if value > 1.0:
                        value /= 100.0
                    if 0.0 <= value <= 1.0:
                        return value
    return None


# --------------------------------------------------------------------------- #
# multiple choice
# --------------------------------------------------------------------------- #


def _match_option(label: str, options: list[str]) -> str | None:
    target = normalize_name(label)
    normalized = {normalize_name(o): o for o in options}
    if target in normalized:
        return normalized[target]
    stripped = re.sub(r"^(option|choice)\s*", "", target).strip(" .:)")
    if stripped in normalized:
        return normalized[stripped]
    index_match = re.fullmatch(r"(?:option|choice)?\s*(\d{1,2})", target)
    if index_match:
        idx = int(index_match.group(1)) - 1
        if 0 <= idx < len(options):
            return options[idx]
    letter_match = re.fullmatch(r"(?:option|choice)\s*([a-z])", target)
    if letter_match:
        idx = ord(letter_match.group(1)) - ord("a")
        if 0 <= idx < len(options):
            return options[idx]
    candidates = [o for n, o in normalized.items() if n and (n in target or target in n)]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _finalize_mc(raw: dict[str, float], options: list[str]) -> dict[str, float] | None:
    if not raw:
        return None
    total = sum(raw.values())
    if total <= 0:
        return None
    if total > 1.5:  # percentages
        raw = {k: v / 100.0 for k, v in raw.items()}
        total = sum(raw.values())
    missing = [o for o in options if o not in raw]
    if missing:
        if total < 0.97:
            return None
        for option in missing:
            raw[option] = 0.0
    if not 0.85 <= total <= 1.15:
        return None
    return {o: max(0.0, raw[o]) / total for o in options}


def parse_multiple_choice(text: str, options: list[str]) -> dict[str, float] | None:
    if not text or not options:
        return None
    for obj in reversed(extract_json_objects(text)):
        if not isinstance(obj, dict):
            continue
        mapping = obj.get("probabilities", obj)
        if isinstance(mapping, list):
            mapping = {
                str(item.get("option", item.get("name", ""))): item.get(
                    "probability", item.get("p")
                )
                for item in mapping
                if isinstance(item, dict)
            }
        if not isinstance(mapping, dict):
            continue
        raw: dict[str, float] = {}
        for label, value in mapping.items():
            option = _match_option(str(label), options)
            number = to_float(value)
            if option is None or number is None:
                continue
            raw[option] = number
        result = _finalize_mc(raw, options)
        if result is not None:
            return result

    # Line-based fallback: "<option>: 23%".
    raw = {}
    for option in options:
        escaped = re.escape(option.strip())
        pattern = re.compile(
            r"(?:^|\n)[\s*\-#>]*(?:option\s*\w*\s*[:.)-]\s*)?[\"'*]*"
            + escaped
            + r"[\"'*]*\s*[:=\-–]\s*("
            + _NUMBER
            + r")\s*%?",
            flags=re.I,
        )
        matches = pattern.findall(text)
        if matches:
            value = to_float(matches[-1])
            if value is not None:
                raw[option] = value
    return _finalize_mc(raw, options)


# --------------------------------------------------------------------------- #
# percentiles (numeric, discrete, date)
# --------------------------------------------------------------------------- #


def parse_date_value(text: Any) -> float | None:
    """ISO date/datetime -> UNIX timestamp (UTC)."""
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text) if text > 10_000_000 else None
    if not isinstance(text, str):
        return None
    s = text.strip().strip("\"'")
    s = s.replace("Z", "+00:00")
    for candidate in (s, s[:10]):
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    m = re.fullmatch(r"(\d{4})-(\d{2})", s)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), 15, tzinfo=timezone.utc).timestamp()
    return None


def _percent_key(label: Any) -> float | None:
    text = str(label).strip().lower()
    text = re.sub(r"^(p|percentile|pct)\s*_?", "", text)
    text = re.sub(r"(st|nd|rd|th)$", "", text).strip(" %")
    value = to_float(text)
    if value is None:
        return None
    if 0.0 < value < 1.0 and "." in text:
        # "0.1" style keys are fractions; "2.5" or "10" style keys are percents.
        return value
    if 0.0 < value < 100.0:
        return value / 100.0
    return None


def _clean_percentiles(pairs: dict[float, float]) -> list[tuple[float, float]] | None:
    if len(pairs) < 5:
        return None
    probs = sorted(pairs)
    if probs[0] > 0.1 + 1e-9 or probs[-1] < 0.9 - 1e-9:
        return None
    return [(p, pairs[p]) for p in probs]


def parse_percentiles(text: str, is_date: bool = False) -> list[tuple[float, float]] | None:
    """Return sorted (probability, value) pairs or None."""
    if not text:
        return None
    value_parser = parse_date_value if is_date else to_float

    for obj in reversed(extract_json_objects(text)):
        mapping: Any = obj.get("percentiles", obj) if isinstance(obj, dict) else None
        pairs: dict[float, float] = {}
        if isinstance(mapping, dict):
            for key, raw_value in mapping.items():
                prob = _percent_key(key)
                value = value_parser(raw_value)
                if prob is not None and value is not None:
                    pairs[round(prob, 6)] = value
        elif isinstance(mapping, list):
            for item in mapping:
                if not isinstance(item, dict):
                    continue
                prob = _percent_key(item.get("percentile", item.get("p")))
                value = value_parser(item.get("value", item.get("v")))
                if prob is not None and value is not None:
                    pairs[round(prob, 6)] = value
        cleaned = _clean_percentiles(pairs)
        if cleaned:
            return cleaned

    value_pattern = (
        r"(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?Z?)?)" if is_date else "(" + _NUMBER + ")"
    )
    line_patterns = (
        # "Percentile 10: 123", "P10 = 123", "p2.5: 7"
        re.compile(
            r"\b(?:percentile|p)\s*(\d{1,2}(?:\.\d+)?)(?:st|nd|rd|th)?\s*\**\s*[:=]\s*[\"'*]*"
            + value_pattern,
            flags=re.I,
        ),
        # "10th percentile: 123"
        re.compile(
            r"\b(\d{1,2}(?:\.\d+)?)(?:st|nd|rd|th)\s+percentile\s*\**\s*[:=]\s*[\"'*]*"
            + value_pattern,
            flags=re.I,
        ),
    )
    for line_pattern in line_patterns:
        pairs = {}
        for match in line_pattern.finditer(text):
            prob = _percent_key(match.group(1))
            value = value_parser(match.group(2))
            if prob is not None and value is not None:
                pairs[round(prob, 6)] = value  # later lines override earlier drafts
        cleaned = _clean_percentiles(pairs)
        if cleaned:
            return cleaned
    return None


def standard_percentile_labels() -> list[str]:
    labels = []
    for p in STANDARD_PERCENTILES:
        pct = p * 100
        labels.append(f"{pct:g}")
    return labels
