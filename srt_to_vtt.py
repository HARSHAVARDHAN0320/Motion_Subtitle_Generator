# srt_to_vtt.py
from pathlib import Path
p = Path("sample.srt")
text = p.read_text(encoding="utf-8")
# WebVTT requires header "WEBVTT" then a blank line
if not text.startswith("WEBVTT"):
    text = "WEBVTT\n\n" + text
Path("sample.vtt").write_text(text, encoding="utf-8")
print("Wrote sample.vtt")
