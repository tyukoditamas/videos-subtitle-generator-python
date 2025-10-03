# Video → Spanish Subtitles (SRT) – Python CLI

This tool transcribes an Italian MP4 (or any ffmpeg-readable media) to **Italian SRT** using Whisper,
then translates the captions to **Spanish** with **DeepL API**, preserving timestamps.

## Quick start
1) **Install ffmpeg**
   - macOS: `brew install ffmpeg`
   - Ubuntu: `sudo apt-get install ffmpeg`
   - Windows: download ffmpeg and add to PATH

2) **Create venv + install**
```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3) **DeepL key**
- Sign up for DeepL API Free/Pro, create an API key.
- Copy `.env.example` to `.env` and set `DEEPL_API_KEY`.
- For Free plan keep `DEEPL_API_URL=https://api-free.deepl.com` (default).
- For Pro use `https://api.deepl.com`.

4) **Run**
```bash
python vid2srt.py --input /path/to/video.mp4 --deepl --model medium --wrap 42 --formality default
```
Outputs: `..._it.srt` (Italian) and `..._es.srt` (Spanish).

## Options
- `--model` tiny|base|small|medium|large  (bigger = better)
- `--wrap`  wrap lines at N chars (e.g. 42)
- `--formality` DeepL: default|more|less
- `--dry-run` only estimate character count
