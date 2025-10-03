import re
from dataclasses import dataclass
from typing import List, Optional
from xml.etree import ElementTree as ET

@dataclass
class Segment:
    text: Optional[str] = None     # None => break
    break_ms: int = 0
    tempo: float = 1.0             # 1.0 = unchanged
    pitch_cents: int = 0           # +/- cents
    gain_db: float = 0.0           # +/- dB

_PROSODY_RE = re.compile(r'<\s*prosody\b', re.I)
_EMPHASIS_RE = re.compile(r'<\s*emphasis\b', re.I)

def has_prosody_or_emphasis(text: str) -> bool:
    return bool(text and (_PROSODY_RE.search(text) or _EMPHASIS_RE.search(text)))

def _rate_to_tempo(rate: str) -> float:
    rate = rate.strip().lower()
    table = {"x-slow":0.8, "slow":0.9, "medium":1.0, "fast":1.1, "x-fast":1.25}
    if rate in table: return table[rate]
    m = re.match(r'^(\d+)%$', rate)
    if m:
        v = max(50, min(200, int(m.group(1))))
        return v/100.0
    try: return float(rate)
    except: return 1.0

def _pitch_to_cents(pitch: str) -> int:
    pitch = pitch.strip().lower()
    m = re.match(r'^([+-]?\d+)\s*st$', pitch)   # semitones
    if m: return int(m.group(1))*100
    m = re.match(r'^([+-]?\d+)\s*c$', pitch)    # cents
    if m: return int(m.group(1))
    try: return int(float(pitch)*100.0)
    except: return 0

def _emphasis_fx(level: str) -> tuple[float, float]:
    level = (level or "").strip().lower()
    if level.startswith("strong"):   return (+4.0, 0.94)  # gain, tempo
    if level.startswith("moderate"): return (+2.0, 0.97)
    if level.startswith("reduced"):  return (-2.0, 1.03)
    return (0.0, 1.0)

def _parse_time_to_ms(val: str) -> int:
    v = (val or "").strip().lower()
    try:
        if v.endswith("ms"): ms = float(v[:-2])
        elif v.endswith("s"): ms = float(v[:-1])*1000.0
        else: ms = float(v)*1000.0
        ms = int(round(ms))
    except: ms = 0
    return max(0, min(ms, 10000))  # cap 10s

def _walk(node: ET.Element, inherited: dict, out: List[Segment]) -> None:
    tag = re.sub(r'^{.*}', '', node.tag).lower()
    local = dict(inherited)
    if tag == 'break':
        out.append(Segment(text=None, break_ms=_parse_time_to_ms(node.attrib.get('time','300ms'))))
        if node.tail and node.tail.strip():
            out.append(Segment(text=node.tail.strip(), tempo=inherited['tempo'], pitch_cents=inherited['pitch_cents'], gain_db=inherited['gain_db']))
        return
    if tag == 'prosody':
        if 'rate' in node.attrib:  local['tempo'] *= _rate_to_tempo(node.attrib['rate'])
        if 'pitch' in node.attrib: local['pitch_cents'] += _pitch_to_cents(node.attrib['pitch'])
    if tag == 'emphasis':
        g,t = _emphasis_fx(node.attrib.get('level','moderate'))
        local['gain_db'] += g
        local['tempo']   *= t

    if node.text and node.text.strip():
        out.append(Segment(text=node.text.strip(), tempo=local['tempo'], pitch_cents=local['pitch_cents'], gain_db=local['gain_db']))
    for child in list(node):
        _walk(child, local, out)
    if node.tail and node.tail.strip():
        out.append(Segment(text=node.tail.strip(), tempo=inherited['tempo'], pitch_cents=inherited['pitch_cents'], gain_db=inherited['gain_db']))

def parse_segments(ssml_text: str) -> List[Segment]:
    try: root = ET.fromstring(ssml_text)
    except ET.ParseError: return [Segment(text=ssml_text)]
    segs: List[Segment] = []
    _walk(root, {'tempo':1.0,'pitch_cents':0,'gain_db':0.0}, segs)
    merged: List[Segment] = []
    for s in segs:
        if s.text is None:
            merged.append(s); continue
        if merged and merged[-1].text is not None:
            last = merged[-1]
            if last.tempo==s.tempo and last.pitch_cents==s.pitch_cents and abs(last.gain_db-s.gain_db)<1e-6:
                merged[-1] = Segment(text=(last.text+" "+s.text).strip(), tempo=last.tempo, pitch_cents=last.pitch_cents, gain_db=last.gain_db)
                continue
        merged.append(s)
    return merged
