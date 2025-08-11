import re
from typing import List, Dict, Union

BreakToken = Dict[str, Union[str, int]]
TextToken = Dict[str, str]

# Standard tags
_BREAK_RE = re.compile(r'<\s*break\b[^>]*time\s*=\s*"([^"]+)"\s*/?>', re.I)
_SPEAK_TAGS_RE = re.compile(r'</?\s*speak[^>]*>', re.I)

# Weird variant some paths emit: <break time equals "800ms " slash >
_BREAK_WEIRD_RE = re.compile(r'<\s*break\b[^>]*(?:\bequals\b|\bslash\b)[^>]*>', re.I)

def is_ssml(s: str) -> bool:
    return bool(_SPEAK_TAGS_RE.search(s) or _BREAK_RE.search(s) or _BREAK_WEIRD_RE.search(s))

def _parse_time_ms(val: str) -> int:
    v = val.strip().lower()
    try:
        if v.endswith("ms"):
            ms = int(round(float(v[:-2])))
        elif v.endswith("s"):
            ms = int(round(float(v[:-1]) * 1000))
        else:
            ms = int(round(float(v)))
    except Exception:
        ms = 0
    return max(0, min(ms, 10_000))  # cap 10s

def _normalize_weird_breaks(s: str) -> str:
    def repl(m: re.Match) -> str:
        inner = m.group(0)
        inner = re.sub(r'(\btime)\s+equals\s+', r'\1=', inner, flags=re.I)            # time equals -> time=
        inner = re.sub(r'="([^"]*?)\s+"', r'="\1"', inner)                            # trim space in quotes
        inner = re.sub(r'\s+slash\s*>', r'/>', inner, flags=re.I)                     # slash > -> />
        return inner
    return _BREAK_WEIRD_RE.sub(repl, s) if _BREAK_WEIRD_RE.search(s) else s

def tokenize_ssml(s: str) -> List[Union[TextToken, BreakToken]]:
    s = _normalize_weird_breaks(s)
    s = _SPEAK_TAGS_RE.sub("", s)  # strip <speak>
    tokens: List[Union[TextToken, BreakToken]] = []
    pos = 0
    for m in _BREAK_RE.finditer(s):
        start, end = m.start(), m.end()
        if start > pos:
            txt = s[pos:start]
            if txt.strip():
                tokens.append({"type": "text", "text": txt})
        ms = _parse_time_ms(m.group(1))
        tokens.append({"type": "break", "ms": ms})
        pos = end
    if pos < len(s):
        tail = s[pos:]
        if tail.strip():
            tokens.append({"type": "text", "text": tail})
    return tokens

def to_pause_tags(text: str) -> str:
    """Convert SSML <break time> to engine-native [pause:Ns] and drop <speak>."""
    parts = []
    for t in tokenize_ssml(text):
        if t.get("type") == "text":
            parts.append(t.get("text", ""))
        else:
            ms = int(t.get("ms", 0))
            parts.append(f"[pause:{ms/1000:g}s]")
    return "".join(parts)
