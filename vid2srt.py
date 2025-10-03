#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
import sys
import time
import requests
from dotenv import load_dotenv

import subprocess, math, tempfile, os, re
from tqdm import tqdm



import whisper, torch, srt
from datetime import timedelta


DEEPL_DEFAULT_URL = "https://api-free.deepl.com"  # override via env

def human_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def probe_duration(path: str) -> float:
    """
    Returns duration in seconds using ffprobe (ffmpeg must be installed).
    """
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ])
    return float(out.decode().strip())

def soft_wrap(text: str, width: int) -> str:
    if width <= 0:
        return text
    words = text.split()
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        extra = (1 if cur else 0) + len(w)
        if cur_len + extra > width:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += extra
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)

def transcribe_to_srt(input_path: str, model_size: str, language: str,
                      *_args, chunk_seconds: int = 60) -> list:
    """
    Transcribe in fixed-size chunks so we can show a real progress bar.
    Keeps timestamps correct by offsetting segment times per chunk.
    """
    # Device preference with safe fallback

    device = "cpu"
    print(f"Whisper (chunked): model={model_size}, device={device}, chunk={chunk_seconds}s")

    # Load once (MPS might fail; fallback to CPU)
    try:
        model = whisper.load_model(model_size, device=device)
        fp16 = False  # keep FP32; stable and same quality
    except Exception as e:
        print(f"MPS init failed ({e}). Falling back to CPU.")
        device = "cpu"
        model = whisper.load_model(model_size, device=device)
        fp16 = False

    total = probe_duration(input_path)
    pbar = tqdm(total=total, unit="s", desc="Decoding", smoothing=0.1)

    subs = []
    index = 1
    pos = 0.0
    while pos < total:
        length = min(chunk_seconds, total - pos)

        # Make a temporary wav of this chunk
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            # Re-encode to 16k mono PCM for whisper
            # (fast and avoids loading the full file each time)
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{pos}",
                "-t", f"{length}",
                "-i", input_path,
                "-vn", "-ac", "1", "-ar", "16000",
                "-f", "wav", tmp_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            # Transcribe just this slice
            result = model.transcribe(tmp_path, language=language, fp16=fp16)

            # Collect segments with global offsets
            for seg in result.get("segments", []):
                start = seg["start"] + pos
                end = seg["end"] + pos
                text = seg["text"].strip()
                subs.append(
                    srt.Subtitle(
                        index=index,
                        start=timedelta(seconds=start),
                        end=timedelta(seconds=end),
                        content=text
                    )
                )
                index += 1
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        pos += length
        pbar.update(length)

    pbar.close()
    return subs

def estimate_chars(subs: list) -> int:
    return sum(len(s.content) for s in subs)

def chunk_indices_by_chars(texts, max_chars: int):
    """Yield slices (start,end) where each chunk total chars <= max_chars."""
    start = 0
    cur = 0
    for i, t in enumerate(texts, start=1):
        L = len(t)
        if L > max_chars and start == i-1:
            # single huge line: force split
            yield (start, i)
            start = i
            cur = 0
            continue
        if cur + L > max_chars and start < i-1:
            yield (start, i-1)
            start = i-1
            cur = L
        else:
            cur += L
    if start < len(texts):
        yield (start, len(texts))

def deepl_translate_batch(api_url: str, api_key: str, src_lang: str, tgt_lang: str, texts: list, formality: str):
    """
    Translate a list of strings using DeepL.
    Sends them as multiple 'text' params in a single request.
    """
    endpoint = api_url.rstrip("/") + "/v2/translate"
    data = [
        ("auth_key", api_key),
        ("target_lang", tgt_lang),
        ("source_lang", src_lang),
        ("preserve_formatting", "1"),
        ("split_sentences", "1"),
    ]
    if formality and formality.lower() in ("more", "less", "default"):
        data.append(("formality", formality.lower()))
    for t in texts:
        data.append(("text", t))

    for attempt in range(6):
        try:
            r = requests.post(endpoint, data=data, timeout=60)
            if r.status_code == 429 or r.status_code == 456:
                # Too Many Requests / Quota exceeded: backoff
                wait = min(60, 2 ** attempt)
                time.sleep(wait)
                continue
            r.raise_for_status()
            js = r.json()
            translations = [item["text"] for item in js.get("translations", [])]
            if len(translations) != len(texts):
                raise RuntimeError("DeepL returned mismatched translations count")
            return translations
        except Exception as e:
            if attempt == 5:
                raise
            time.sleep(1 + attempt)

def translate_srt_with_deepl(subs: list, src_lang: str, tgt_lang: str, api_key: str,
                             api_url: str, batch_chars: int, formality: str) -> list:
    texts = [s.content for s in subs]
    translated = [None] * len(texts)
    for (a, b) in tqdm(list(chunk_indices_by_chars(texts, batch_chars)), desc="Translating"):
        chunk = texts[a:b]
        out = deepl_translate_batch(api_url, api_key, src_lang, tgt_lang, chunk, formality)
        translated[a:b] = out
    new_subs = []
    for s, txt in zip(subs, translated):
        new_subs.append(srt.Subtitle(index=s.index, start=s.start, end=s.end, content=txt))
    return new_subs

def _split_words_even(words, parts):
    n = len(words)
    step = math.ceil(n/parts)
    return [" ".join(words[i:i+step]).strip() for i in range(0, n, step)]

def _normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text

def _wrap_lines(words, max_chars):
    lines, cur = [], ""
    for w in words:
        cand = (cur + " " + w).strip() if cur else w
        if len(cand) <= max_chars:
            cur = cand
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines

def _split_by_ratio(words, parts):
    """Split words into `parts` chunks, preferring punctuation near split."""
    total = sum(len(w) for w in words) + max(0, len(words)-1)
    targets = [round(total*(i+1)/parts) for i in range(parts-1)]
    res, acc, start = [], 0, 0
    for i, w in enumerate(words):
        acc += len(w) + (1 if i else 0)
        if targets and acc >= targets[0]:
            cut = i+1
            # look back a few words for punctuation
            for j in range(i, max(start, i-5), -1):
                if words[j-1][-1:] in ".?!,;:":
                    cut = j; break
            res.append(" ".join(words[start:cut]).strip())
            start = cut
            targets.pop(0)
    res.append(" ".join(words[start:]).strip())
    return res

def tidy_subs(subs, *, max_chars=42, max_lines=2, target_cps=17.0,
              min_dur=1.0, max_dur=6.0, gap=0.04):
    """
    Enforce readability: ≤ max_lines, ≤ max_chars/line, CPS limit,
    and clamp durations. Splits long captions into multiple timed ones.
    """
    fixed = []
    for s in subs:
        text = _normalize(s.content)
        t0, t1 = s.start, s.end
        dur = max((t1 - t0).total_seconds(), 1e-6)
        cps = len(text.replace(" ", "")) / dur

        words = text.split()
        # how many parts do we need to satisfy all limits?
        need_chars = math.ceil(len(text) / (max_chars * max_lines))
        need_speed = math.ceil(cps / target_cps) if cps > target_cps else 1
        need_dur = math.ceil(dur / max_dur) if dur > max_dur else 1
        parts = max(1, need_chars, need_speed, need_dur)

        # candidate without splitting
        lines = _wrap_lines(words, max_chars)
        if parts == 1 and len(lines) <= max_lines and dur >= min_dur:
            fixed.append(srt.Subtitle(0, t0, t1, "\n".join(lines)))
            continue

        # split and allocate time by character share
        chunks = _split_by_ratio(words, parts)
        total_chars = sum(len(c) for c in chunks) or 1
        cur = t0
        for ch in chunks:
            ch = _normalize(ch)
            ch_chars = len(ch)
            seg_dur = max(min_dur, dur * (ch_chars / total_chars))
            end = cur + timedelta(seconds=seg_dur)
            lines = _wrap_lines(ch.split(), max_chars)

            # if still too many lines, split sub-chunk again
            if len(lines) > max_lines:
                subparts = math.ceil(len(ch) / (max_chars * max_lines))
                subsplits = _split_by_ratio(ch.split(), subparts)
                inner_total = sum(len(x) for x in subsplits) or 1
                inner_cur = cur
                for sw in subsplits:
                    sw = _normalize(sw)
                    frac = len(sw) / inner_total
                    inner_end = inner_cur + timedelta(seconds=seg_dur * frac)
                    fixed.append(srt.Subtitle(0, inner_cur, inner_end,
                                              "\n".join(_wrap_lines(sw.split(), max_chars))))
                    inner_cur = inner_end
            else:
                fixed.append(srt.Subtitle(0, cur, end, "\n".join(lines)))

            cur = end + timedelta(seconds=gap)  # tiny gap to avoid overlaps

    # final renumber and clamp max_dur
    out = []
    for i, s in enumerate(fixed, 1):
        dur = (s.end - s.start).total_seconds()
        if dur > max_dur:
            s.end = s.start + timedelta(seconds=max_dur)
        out.append(srt.Subtitle(i, s.start, s.end, s.content))
    return out

def main():
    load_dotenv()  # load .env if present

    p = argparse.ArgumentParser(description="Transcribe Italian MP4 to SRT and translate to Spanish via DeepL.")
    p.add_argument("--input", required=True, help="Path to input media (mp4/mp3/wav etc.)")
    p.add_argument("--out-it", default=None, help="Path to write Italian SRT (default: input basename + _it.srt)")
    p.add_argument("--out-es", default=None, help="Path to write Spanish SRT (default: input basename + _es.srt)")
    p.add_argument("--model", default="medium", help="Whisper model size: tiny|base|small|medium|large")
    p.add_argument("--language", default="it", help="Source language code (default: it)")
    p.add_argument("--deepl", action="store_true", help="Enable translation to Spanish using DeepL")
    p.add_argument("--wrap", type=int, default=None, help="Soft-wrap lines at N chars (e.g., 42)")
    p.add_argument("--formality", default="default", help="DeepL formality: default|more|less")
    p.add_argument("--batch-chars", type=int, default=4500, help="Max characters per DeepL request (safety default 4500)")
    p.add_argument("--dry-run", action="store_true", help="Only estimate character count and exit")
    p.add_argument("--device", default="cpu", choices=["auto","cpu","cuda","metal"])
    p.add_argument("--compute-type", default="int8_float16",
                   choices=["int8","int8_float16","float16","float32"])
    p.add_argument("--max-lines", type=int, default=2, help="Max lines per subtitle.")
    p.add_argument("--cps", type=float, default=17.0, help="Max reading speed (chars/sec).")
    p.add_argument("--min-dur", type=float, default=1.0, help="Min subtitle duration (sec).")
    p.add_argument("--max-dur", type=float, default=6.0, help="Max subtitle duration (sec).")
    p.add_argument("--gap", type=float, default=0.04, help="Min gap between subtitles in seconds.")

    args = p.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    base = os.path.splitext(input_path)[0]
    out_it = args.out_it or f"{base}_it.srt"
    out_es = args.out_es or f"{base}_es.srt"
    wrap_width = args.wrap or 42  # ensure a number for tidy_subs

    print(f"[1/4] Transcribing with Whisper ({args.model})...")
    subs_it = transcribe_to_srt(
        input_path,
        args.model,
        args.language,
        args.device, args.compute_type,  # extra args are ignored by this function
        chunk_seconds=60                 # tweak to 30–90 for your taste
    )

    subs_it = tidy_subs(
        subs_it,
        max_chars=wrap_width,
        max_lines=args.max_lines,
        target_cps=args.cps,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        gap=args.gap,
    )

    it_chars = estimate_chars(subs_it)
    print(f"   Italian captions: {len(subs_it)} segments, ~{it_chars} characters")

    if args.dry_run:
        print("Dry run: exiting before writing files.")
        return

    with open(out_it, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs_it))
    print(f"[2/4] Wrote Italian SRT → {out_it}")


    if not args.deepl:
            print("[3/4] DeepL translation disabled. Done.")
            return

    api_key = os.getenv("DEEPL_API_KEY")
    if not api_key:
        print("ERROR: DEEPL_API_KEY not set. Put it in .env or the environment.", file=sys.stderr)
        sys.exit(2)
    api_url = os.getenv("DEEPL_API_URL", DEEPL_DEFAULT_URL)

    print(f"[3/4] Translating to Spanish via DeepL ({api_url})...")
    subs_es = translate_srt_with_deepl(subs_it, src_lang=args.language.upper(), tgt_lang="ES",
                                       api_key=api_key, api_url=api_url,
                                       batch_chars=args.batch_chars, formality=args.formality)

    subs_es = tidy_subs(
        subs_es,
        max_chars=wrap_width,
        max_lines=args.max_lines,
        target_cps=args.cps,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        gap=args.gap,
    )

    with open(out_es, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs_es))
    print(f"[4/4] Wrote Spanish SRT → {out_es}")

    print("Done ✅")

if __name__ == "__main__":
    main()
