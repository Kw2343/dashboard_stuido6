"""
llm_insights.py
---------------
AI-powered Insights / Interpretation / Recommendations for the dashboard.
 
FREE PROVIDERS SUPPORTED
------------------------
Set PROVIDER below to one of:  "gemini"  |  "groq"  |  "anthropic"
 
GEMINI  (free, 1500 req/day) — https://aistudio.google.com  → Get API Key
  pip install google-generativeai
  secrets.toml → GEMINI_API_KEY = "AIza..."
 
GROQ  (free, generous limits) — https://console.groq.com  → API Keys
  pip install groq
  secrets.toml → GROQ_API_KEY = "gsk_..."
 
ANTHROPIC  (paid) — https://console.anthropic.com  → API Keys
  pip install anthropic
  secrets.toml → ANTHROPIC_API_KEY = "sk-ant-..."
 
GUARDRAILS
----------
All calls go through guardrails.py before and after the LLM call:
 
  Pre-call:
    • Kill-switch (AI_INSIGHTS_ENABLED secret/env var)
    • Per-session rate limiting (30 calls / hour)
    • Prompt-injection detection in context
    • PII scrubbing (e-mail, phone, API keys, IPs)
    • Data-leakage check (no raw review text / credentials)
    • Context length cap (8 000 chars)
    • User-ID hashing before transmission
 
  Post-call:
    • Injection scan on model response
    • Response length cap (4 000 chars)
    • Format / structure check
    • Grounding validation — key dataset numbers must appear in the response
"""
 
from __future__ import annotations
 
import hashlib
import os
 
import streamlit as st
 
from guardrails import (
    pre_call_checks,
    post_call_checks,
    extract_known_values_from_context,
    hash_user_id,
)
 
# ╔══════════════════════════════════════════════════════════╗
# ║  CHANGE THIS LINE to switch provider                     ║
PROVIDER = "groq"   # "gemini" | "groq" | "anthropic"
# ╚══════════════════════════════════════════════════════════╝
 
 
# ── System prompt (shared across all providers) ───────────────────────────────
 
_SYSTEM = """\
You are a sharp, experienced e-commerce analyst who has just looked at the chart \
and data summary on screen. Write like a real human colleague explaining what they \
see — direct, a little opinionated, and grounded in the exact numbers provided. \
Never invent figures. Never say things like "the data shows" or "it is evident that."
 
IMPORTANT — HALLUCINATION PREVENTION:
Only reference numbers that appear explicitly in the data summary provided to you. \
Do NOT invent, estimate, or extrapolate any figures. If a metric is absent from the \
data, state that it is unavailable rather than guessing. Every numerical claim you \
make must be traceable to the data summary.
 
PRIVACY:
User IDs in the data have been anonymised (USR_ prefix). Do not attempt to \
reconstruct or speculate about real user identities.
 
Structure your response with these three sections, written in natural prose (not \
bullet lists). Each section should read like a short paragraph spoken aloud:
 
**What the chart is telling us**
Narrate the most striking pattern visible in the numbers — the peak, the gap, \
the skew, the outlier. Name the actual values. Be specific enough that someone \
who hasn't seen the chart can picture it.
 
**What this means for the business**
Connect the pattern to a real business consequence: revenue risk, a growth \
opportunity, a customer behaviour insight, or a warning sign. Be honest if \
something looks concerning.
 
**What to do about it**
Give 2–3 concrete actions in priority order. Each action must be tied directly \
to a number from the data — no vague advice. If one action is clearly more urgent \
than the others, say so explicitly.
 
Keep the whole response under 220 words. Write in plain English — no jargon, \
no filler phrases, no generic preambles.
"""
 
 
# ── Secret / env key lookup ───────────────────────────────────────────────────
 
def _secret(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        try:
            val = st.secrets.get(name, "") or ""
        except Exception:
            pass
    return val
 
 
# ── Provider implementations ──────────────────────────────────────────────────
 
def _call_gemini(context: str) -> tuple[bool, str]:
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        return False, (
            "⚠️ **Package missing.** Run:\n```\npip install google-generativeai\n```"
        )
 
    key = _secret("GEMINI_API_KEY")
    if not key:
        return False, (
            "⚠️ **`GEMINI_API_KEY` not set.**\n\n"
            "1. Get a free key at [aistudio.google.com](https://aistudio.google.com)\n"
            "2. Add to `.streamlit/secrets.toml`:\n"
            "```toml\nGEMINI_API_KEY = \"AIza...\"\n```"
        )
 
    try:
        genai.configure(api_key=key)
        model  = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=_SYSTEM,
        )
        result = model.generate_content(context)
        return True, result.text
    except Exception as exc:
        return False, f"⚠️ **Gemini error:** {exc}"
 
 
def _call_groq(context: str) -> tuple[bool, str]:
    try:
        from groq import Groq  # type: ignore
    except ImportError:
        return False, (
            "⚠️ **Package missing.** Run:\n```\npip install groq\n```"
        )
 
    key = _secret("GROQ_API_KEY")
    if not key:
        return False, (
            "⚠️ **`GROQ_API_KEY` not set.**\n\n"
            "1. Get a free key at [console.groq.com](https://console.groq.com)\n"
            "2. Add to `.streamlit/secrets.toml`:\n"
            "```toml\nGROQ_API_KEY = \"gsk_...\"\n```"
        )
 
    try:
        client = Groq(api_key=key)
        chat   = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system",  "content": _SYSTEM},
                {"role": "user",    "content": context},
            ],
            max_tokens=1200,
        )
        return True, chat.choices[0].message.content
    except Exception as exc:
        return False, f"⚠️ **Groq error:** {exc}"
 
 
def _call_anthropic(context: str) -> tuple[bool, str]:
    try:
        import anthropic  # type: ignore
    except ImportError:
        return False, (
            "⚠️ **Package missing.** Run:\n```\npip install anthropic\n```"
        )
 
    key = _secret("ANTHROPIC_API_KEY")
    if not key:
        return False, (
            "⚠️ **`ANTHROPIC_API_KEY` not set.**\n\n"
            "Add to `.streamlit/secrets.toml`:\n"
            "```toml\nANTHROPIC_API_KEY = \"sk-ant-...\"\n```"
        )
 
    try:
        client = anthropic.Anthropic(api_key=key)
        msg    = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            system=_SYSTEM,
            messages=[{"role": "user", "content": context}],
        )
        return True, msg.content[0].text
    except Exception as exc:
        return False, f"⚠️ **Anthropic error:** {exc}"
 
 
# ── Router ────────────────────────────────────────────────────────────────────
 
_PROVIDERS = {
    "gemini":    _call_gemini,
    "groq":      _call_groq,
    "anthropic": _call_anthropic,
}
 
 
def _call_llm(context: str) -> tuple[bool, str]:
    fn = _PROVIDERS.get(PROVIDER.lower())
    if fn is None:
        return False, f"⚠️ Unknown PROVIDER `{PROVIDER}`. Choose: gemini | groq | anthropic"
    return fn(context)
 
 
# ── Public Streamlit component ────────────────────────────────────────────────
 
def show_llm_insights(
    context: str,
    cache_key: str,
    title: str = "AI-Powered Analysis",
    chart_type: str = "",
    user_id: str | None = None,
    known_values: dict[str, float] | None = None,
) -> None:
    """
    Render an AI insights block with Generate / Refresh controls.
 
    Parameters
    ----------
    context : str
        Structured data summary for the LLM. Must NOT contain raw review
        text, credentials, or unmasked PII — the guardrail layer will
        reject or scrub these, but it is better to never include them.
    cache_key : str
        Stable identifier for this insight block (used for session caching).
    title : str
        Section heading shown above the insight block.
    chart_type : str
        Optional plain-English description of the visualisation type, e.g.
        "horizontal bar chart showing the top 10 products by review count".
    user_id : str | None
        Raw user ID associated with this request.  It will be one-way hashed
        before being sent to the LLM — never transmitted in plain text.
    known_values : dict[str, float] | None
        Key dataset figures used for grounding validation, e.g.
        ``{"avg_rating": 4.23, "total_reviews": 15430}``.
        If None, they are auto-extracted from *context*.
    """
    # ── Build the full prompt ─────────────────────────────────────────────────
    chart_hint   = f"You are looking at a {chart_type}. " if chart_type else ""
    full_context = (
        f"{chart_hint}Here is the underlying data summary for the chart above:\n\n"
        f"{context}\n\n"
        "Interpret this as a human analyst who has just studied the chart. "
        "Reference specific numbers from the data in every section. "
        "Do not invent any figures that are not in the data summary above."
    )
 
    # ── Pre-call guardrails ───────────────────────────────────────────────────
    pre = pre_call_checks(
        context=full_context,
        session_state=st.session_state,
        user_id=user_id,
    )
 
    if not pre.passed:
        st.markdown("---")
        st.warning(f"🛡️ **Guardrail blocked this request:** {pre.reason}")
        return
 
    # Use the sanitised (PII-scrubbed, truncated) context from here on.
    sanitised_context: str = pre.sanitised_value
 
    # Surface non-fatal warnings (e.g. PII was scrubbed) to admins.
    if pre.warnings:
        with st.expander("⚠️ Guardrail warnings (pre-call)", expanded=False):
            for w in pre.warnings:
                st.caption(w)
 
    # ── Cache key includes context hash so filter changes auto-invalidate ─────
    ctx_hash  = hashlib.md5(sanitised_context.encode()).hexdigest()[:10]
    state_key = f"llm_{cache_key}_{ctx_hash}"
 
    st.markdown("---")
    hdr_col, btn_col = st.columns([5, 1])
    hdr_col.markdown(f"#### 🤖 {title}")
 
    if state_key in st.session_state:
        cached_entry = st.session_state[state_key]
 
        if btn_col.button(
            "🔄 Refresh",
            key=f"refresh_{cache_key}_{ctx_hash}",
            help="Regenerate AI analysis",
        ):
            del st.session_state[state_key]
            st.rerun()
 
        st.markdown(cached_entry["text"])
 
        # Show any post-call grounding warnings that were stored with the result.
        if cached_entry.get("warnings"):
            with st.expander("⚠️ Grounding & format warnings", expanded=False):
                for w in cached_entry["warnings"]:
                    st.caption(w)
 
    else:
        if btn_col.button(
            "✨ Generate",
            key=f"gen_{cache_key}_{ctx_hash}",
            type="primary",
            help="Generate AI insights for this section",
        ):
            with st.spinner(f"Analysing with {PROVIDER.title()}…"):
                success, text = _call_llm(sanitised_context)
 
            if not success:
                st.error(text)   # provider error — shown inline, never cached
                return
 
            # ── Post-call guardrails ──────────────────────────────────────────
            kv = known_values or extract_known_values_from_context(context)
            post = post_call_checks(response=text, known_values=kv)
 
            if not post.passed:
                st.error(f"🛡️ **Guardrail rejected the LLM response:** {post.reason}")
                return
 
            # Store sanitised response + any warnings in session state.
            st.session_state[state_key] = {
                "text":     post.sanitised_value,
                "warnings": post.warnings,
            }
            st.rerun()
 
        else:
            st.info(
                f"Click **✨ Generate** for AI-powered analysis "
                f"*(powered by {PROVIDER.title()})*"
            )