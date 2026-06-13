"""
guardrails.py
=============
Centralised guardrail layer for all LLM calls in the dashboard.
 
Responsibilities
----------------
1. PRIVACY       — Strip / hash any PII before it leaves the process.
2. SECURITY      — Detect and block prompt-injection attempts.
3. HALLUCINATION — Validate LLM output against known dataset values;
                   flag or reject responses that invent numbers.
4. ACCESS CONTROL— No rate limit — unlimited LLM calls per session.
                   Admins can still disable AI insights entirely via a
                   Streamlit secret or environment variable.
5. DATA LEAKAGE  — Ensure the context blob does not contain raw review
                   text, credentials, or private fields.
6. BUSINESS RULES— Enforce output length, tone, and format constraints.
 
FORBIDDEN FIELD MATCHING — IMPORTANT
--------------------------------------
The forbidden-field check looks for data *field names used as dict/CSV keys*,
NOT arbitrary substrings.  This prevents false positives from legitimate prose
such as "Here is the underlying data summary for the chart" or section headers
like "DATA SUMMARY:" in context strings.
 
Short generic words ("text", "summary", "token", "secret") are matched only
when they appear as explicit JSON/dict keys (quoted, followed by a colon) —
never as bare substrings in prose.
"""
 
from __future__ import annotations
 
import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────────────────
 
# Fields that must NEVER appear as data keys in the context sent to the LLM.
# Matched as key names (JSON / CSV / colon-label), NOT free substrings.
_FORBIDDEN_KEY_NAMES: frozenset[str] = frozenset({
    "review_text", "reviewtext", "review_body",
    "reviewerName", "reviewer_name",
    "email", "phone",
    "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY",
    "api_key", "apikey", "password",
})
 
# Short ambiguous words — only matched as explicit quoted dict/JSON keys
# (not as colon-labels or CSV headers, to reduce false positives).
_AMBIGUOUS_KEY_NAMES: frozenset[str] = frozenset({
    "text", "summary", "secret", "token",
})
 
# Regex patterns for PII that must be scrubbed.
_PII_PATTERNS: list[tuple[str, str]] = [
    (r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", "<email_redacted>"),
    (r"\b(\+?\d[\s\-.]?){9,11}\d\b", "<phone_redacted>"),
    (r"\b(sk-ant|AIza|sk-|gsk_)[A-Za-z0-9_\-]{20,}", "<api_key_redacted>"),
    (r"\b\d{1,3}(\.\d{1,3}){3}\b", "<ip_redacted>"),
]
 
# Prompt-injection signatures.
_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(the\s+)?(above|previous|prior)",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+(if\s+you\s+are\s+)?a\s+",
    r"new\s+system\s+prompt",
    r"jailbreak",
    r"DAN\s+mode",
    r"<\s*script",
    r"\{\{.*\}\}",
    r"--\s*(drop|select|insert|update|delete)\s+",
]
 
MAX_CONTEXT_CHARS: int  = 8_000
MAX_RESPONSE_CHARS: int = 4_000
 
# Rate limiting is DISABLED — calls are unlimited.
# Set RATE_LIMIT_CALLS = None to make this explicit throughout.
RATE_LIMIT_CALLS: None = None
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. Result type
# ─────────────────────────────────────────────────────────────────────────────
 
@dataclass
class GuardrailResult:
    passed: bool
    reason: str = ""
    sanitised_value: Any = None
    warnings: list[str] = field(default_factory=list)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. Forbidden-key detection
# ─────────────────────────────────────────────────────────────────────────────
 
def _make_key_pattern(key: str, strict: bool = False) -> re.Pattern:
    """
    Return a compiled regex matching *key* only when used as a data key.
 
    Matches:
      "key": ...     or  'key': ...      (JSON / Python dict)
      key: value                          (colon-label at line level, non-strict only)
      key,col2,...                        (CSV header, non-strict only)
 
    strict=True: quoted keys only (used for ambiguous short words like "summary").
    All patterns compiled with IGNORECASE | MULTILINE; no inline flags.
    """
    k = re.escape(key)
    quoted = f'["\']' + k + f'["\']' + r'\s*:'
    parts = [quoted]
    if not strict:
        parts.append(rf'(?:^|[ \t]){k}[ \t]*:')
        parts.append(rf'(?:^|,){k}(?:,|$)')
    return re.compile("|".join(parts), re.IGNORECASE | re.MULTILINE)
 
 
# Pre-compile at import time.
_FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern]] = [
    (k, _make_key_pattern(k, strict=False)) for k in _FORBIDDEN_KEY_NAMES
] + [
    (k, _make_key_pattern(k, strict=True)) for k in _AMBIGUOUS_KEY_NAMES
]
 
 
def _find_forbidden_keys(context: str) -> list[str]:
    return [key for key, pat in _FORBIDDEN_PATTERNS if pat.search(context)]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. Privacy & PII scrubbing
# ─────────────────────────────────────────────────────────────────────────────
 
def hash_user_id(user_id: str) -> str:
    """One-way hash a raw user_id before including it in LLM context."""
    return "USR_" + hashlib.sha256(user_id.encode()).hexdigest()[:12].upper()
 
 
def scrub_pii(text: str) -> tuple[str, list[str]]:
    """Remove PII patterns. Returns (scrubbed_text, warnings)."""
    warnings: list[str] = []
    for pattern, replacement in _PII_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            warnings.append(f"PII scrubbed ({replacement}): {len(matches)} occurrence(s)")
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text, warnings
 
 
def check_context_for_leakage(context: str) -> GuardrailResult:
    """Reject context containing forbidden data keys, PII, or over-length content."""
    warnings: list[str] = []
 
    found_forbidden = _find_forbidden_keys(context)
    if found_forbidden:
        return GuardrailResult(
            passed=False,
            reason=(
                f"Context contains forbidden data fields: {found_forbidden}. "
                "Remove raw review text, credentials, or private fields before sending."
            ),
        )
 
    cleaned, pii_warnings = scrub_pii(context)
    warnings.extend(pii_warnings)
 
    if len(cleaned) > MAX_CONTEXT_CHARS:
        cleaned = cleaned[:MAX_CONTEXT_CHARS] + "\n\n[CONTEXT TRUNCATED BY GUARDRAIL]"
        warnings.append(
            f"Context truncated from {len(context):,} to {MAX_CONTEXT_CHARS:,} characters."
        )
 
    return GuardrailResult(passed=True, sanitised_value=cleaned, warnings=warnings)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. Prompt-injection detection
# ─────────────────────────────────────────────────────────────────────────────
 
def check_prompt_injection(text: str) -> GuardrailResult:
    """Reject text containing prompt-injection patterns."""
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            return GuardrailResult(
                passed=False,
                reason=(
                    "Potential prompt-injection detected. "
                    "The input contains patterns that could manipulate AI behaviour. "
                    "Request blocked."
                ),
            )
    return GuardrailResult(passed=True, sanitised_value=text)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. Grounding validation
# ─────────────────────────────────────────────────────────────────────────────
 
def extract_numbers_from_text(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"\b\d+(?:\.\d+)?\b", text)]
 
 
def check_response_grounding(
    response: str,
    known_values: dict[str, float],
    tolerance: float = 0.10,
) -> GuardrailResult:
    """
    Warn (non-blocking) when LLM response numbers don't match dataset values.
    """
    warnings: list[str] = []
    response_nums = extract_numbers_from_text(response)
    for label, true_val in known_values.items():
        if true_val == 0:
            continue
        close_enough = any(
            abs(n - true_val) / abs(true_val) <= tolerance
            for n in response_nums
        )
        if not close_enough and response_nums:
            warnings.append(
                f"Grounding warning — '{label}' (true={true_val:.2f}) not found "
                f"in LLM response. The model may have hallucinated this figure."
            )
    return GuardrailResult(passed=True, sanitised_value=response, warnings=warnings)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 7. Output validation
# ─────────────────────────────────────────────────────────────────────────────
 
def check_response_format(response: str) -> GuardrailResult:
    """Validate LLM response: non-empty, length cap, structure check."""
    warnings: list[str] = []
 
    if not response or not response.strip():
        return GuardrailResult(passed=False, reason="LLM returned an empty response.")
 
    if len(response) > MAX_RESPONSE_CHARS:
        response = response[:MAX_RESPONSE_CHARS] + "\n\n*[Response trimmed by output guardrail]*"
        warnings.append("LLM response exceeded maximum length and was trimmed.")
 
    expected = [
        "what the chart", "what this means", "what to do",
        "insights", "interpretation", "recommendations",
    ]
    if sum(1 for s in expected if s in response.lower()) < 2:
        warnings.append(
            "LLM response does not appear to follow the expected "
            "Insights / Interpretation / Recommendations structure."
        )
 
    return GuardrailResult(passed=True, sanitised_value=response, warnings=warnings)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 8. Rate limiting — DISABLED (unlimited calls)
# ─────────────────────────────────────────────────────────────────────────────
 
def check_rate_limit(session_state: dict) -> GuardrailResult:
    """
    Rate limiting is disabled — always passes.
    The session_state argument is kept for API compatibility so callers
    do not need to change.
    """
    return GuardrailResult(passed=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 9. Kill-switch
# ─────────────────────────────────────────────────────────────────────────────
 
def llm_is_enabled() -> bool:
    """Return False if AI_INSIGHTS_ENABLED is set to false in env or secrets."""
    import os
    if os.environ.get("AI_INSIGHTS_ENABLED", "").strip().lower() in ("false", "0", "no", "off"):
        return False
    try:
        import streamlit as st
        if str(st.secrets.get("AI_INSIGHTS_ENABLED", "true")).strip().lower() in ("false", "0", "no", "off"):
            return False
    except Exception:
        pass
    return True
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 10. Composite checks
# ─────────────────────────────────────────────────────────────────────────────
 
def pre_call_checks(
    context: str,
    session_state: dict,
    user_id: str | None = None,
) -> GuardrailResult:
    """Run all pre-call guardrails. Returns first failure or sanitised context."""
    if not llm_is_enabled():
        return GuardrailResult(passed=False, reason="AI Insights have been disabled by an administrator.")
 
    # Rate limit check is a no-op but kept for API compatibility.
    rl = check_rate_limit(session_state)
    if not rl.passed:
        return rl
 
    if user_id:
        context = context.replace(user_id, hash_user_id(user_id))
 
    inj = check_prompt_injection(context)
    if not inj.passed:
        return inj
 
    leak = check_context_for_leakage(context)
    if not leak.passed:
        return leak
 
    return GuardrailResult(passed=True, sanitised_value=leak.sanitised_value, warnings=leak.warnings)
 
 
def post_call_checks(
    response: str,
    known_values: dict[str, float] | None = None,
) -> GuardrailResult:
    """Run all post-call guardrails. Returns sanitised response with warnings."""
    inj = check_prompt_injection(response)
    if not inj.passed:
        return GuardrailResult(passed=False, reason="LLM response blocked: prompt-injection patterns detected.")
 
    fmt = check_response_format(response)
    if not fmt.passed:
        return fmt
    response = fmt.sanitised_value
    all_warnings = list(fmt.warnings)
 
    if known_values:
        grnd = check_response_grounding(response, known_values)
        all_warnings.extend(grnd.warnings)
 
    return GuardrailResult(passed=True, sanitised_value=response, warnings=all_warnings)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 11. Auto-extract known values from context strings
# ─────────────────────────────────────────────────────────────────────────────
 
_NUM_LINE_RE = re.compile(r":\s*([\d,]+(?:\.\d+)?)\s*(?:%|⭐)?")
 
 
def extract_known_values_from_context(context: str) -> dict[str, float]:
    """Extract {label: value} pairs from structured context for grounding checks."""
    known: dict[str, float] = {}
    for line in context.splitlines():
        m = _NUM_LINE_RE.search(line)
        if m:
            label = line.split(":")[0].strip().lower().replace(" ", "_")
            try:
                known[label] = float(m.group(1).replace(",", ""))
            except ValueError:
                pass
    return known