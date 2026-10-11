import os
import re
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="ClipSpark AI", page_icon="🎬", layout="wide")

def find_executable(name):
    found = shutil.which(name)
    if found:
        return found
    candidates = [
        Path(r"D:\bin") / f"{name}.exe",
        Path(r"C:\ffmpeg\bin") / f"{name}.exe",
        APP_DIR / "bin" / f"{name}.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None

FFMPEG = find_executable("ffmpeg")
FFPROBE = find_executable("ffprobe")

def run_command(command, cwd=None):
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "Unknown error")[-5000:]
        raise RuntimeError(details)
    return result.stdout

def valid_video_url(url):
    parsed = urlparse(url.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)

def download_video(url, folder):
    opts = {
        "outtmpl": str(folder / "source.%(ext)s"),
        "format": "bestvideo*+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        candidate = folder / "source.mp4"
        if candidate.exists():
            return candidate, info.get("title", "Video")
        requested = info.get("requested_downloads") or []
        for item in requested:
            p = item.get("filepath")
            if p and Path(p).exists():
                return Path(p), info.get("title", "Video")
        files = [p for p in folder.glob("source.*") if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
        if files:
            return files[0], info.get("title", "Video")
    raise RuntimeError("Download finished, but the video file could not be located.")

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    # CPU-friendly setting for 16 GB RAM / integrated graphics.
    return WhisperModel("small", device="cpu", compute_type="int8")

def transcribe_video(video_path, progress=None):
    model = load_whisper_model()
    segments, info = model.transcribe(
        str(video_path), beam_size=3, word_timestamps=True, vad_filter=True
    )
    all_segments = []
    for seg in segments:
        words = []
        if seg.words:
            for word in seg.words:
                words.append({
                    "start": float(word.start),
                    "end": float(word.end),
                    "word": word.word,
                })
        all_segments.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
            "words": words,
        })
    return all_segments, info.language

HOOK_TERMS = {
    "why": 2.0, "secret": 2.4, "mistake": 2.3, "never": 1.8,
    "always": 1.5, "best": 1.5, "worst": 1.7, "truth": 2.1,
    "actually": 1.4, "important": 1.4, "amazing": 1.8, "crazy": 1.8,
    "surprising": 2.0, "learn": 1.2, "money": 1.3, "problem": 1.4,
    "solution": 1.4, "because": 0.8, "here's": 1.2, "finally": 1.3,
    "imagine": 1.5, "remember": 1.1, "number one": 1.5, "the reason": 1.5,
}

def score_text(text):
    lower = text.lower()
    words = re.findall(r"\b[\w']+\b", lower)
    score = min(len(words) / 18.0, 2.5)
    for term, weight in HOOK_TERMS.items():
        if term in lower:
            score += weight
    if "?" in text:
        score += 1.0
    if "!" in text:
        score += 0.5
    if re.search(r"\b(i|we|you) (found|learned|realized|discovered)\b", lower):
        score += 1.0
    return score

def choose_clips(segments, duration, max_clips):
    if not segments:
        return []
    total_end = max(s["end"] for s in segments)
    step = max(3.0, duration / 5.0)
    starts = set()
    for seg in segments:
        starts.add(max(0.0, seg["start"] - 1.5))
        starts.add(max(0.0, seg["start"] - duration * 0.35))
        starts.add(max(0.0, seg["start"]))
    starts.add(0.0)
    candidates = []
    for start in starts:
        end = min(start + duration, total_end)
        if end - start < min(12.0, duration * 0.65):
            continue
        selected = [s for s in segments if s["end"] > start and s["start"] < end]
        text = " ".join(s["text"] for s in selected).strip()
        if len(text.split()) < 8:
            continue
        score = score_text(text)
        # Prefer windows that start near spoken content, and penalize nearly empty openings.
        if selected:
            first_start = min(s["start"] for s in selected)
            if first_start - start > 4:
                score -= 1.5
        candidates.append({"start": start, "end": end, "text": text, "score": score})
    candidates.sort(key=lambda c: c["score"], reverse=True)
    chosen = []
    for cand in candidates:
        if all(cand["end"] <= old["start"] - 1.0 or cand["start"] >= old["end"] + 1.0 for old in chosen):
            chosen.append(cand)
        if len(chosen) >= max_clips:
            break
    chosen.sort(key=lambda c: c["start"])
    return chosen

def srt_time(seconds):
    milliseconds = int(round(max(0, seconds) * 1000))
    hours = milliseconds // 3600000
    milliseconds %= 3600000
    minutes = milliseconds // 60000
    milliseconds %= 60000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def make_srt(segments, clip_start, clip_end, path):
    entries = []
    for seg in segments:
        words = seg.get("words") or []
        if words:
            # Group words into readable caption chunks of up to 7 words.
            chunk = []
            for w in words:
                if w["end"] <= clip_start or w["start"] >= clip_end:
                    continue
                chunk.append(w)
                if len(chunk) >= 7 or w["word"].strip().endswith((".", "?", "!", ",")):
                    start = max(clip_start, chunk[0]["start"]) - clip_start
                    end = min(clip_end, chunk[-1]["end"]) - clip_start
                    text = "".join(x["word"] for x in chunk).strip()
                    if text and end > start:
                        entries.append((start, end, text))
                    chunk = []
            if chunk:
                start = max(clip_start, chunk[0]["start"]) - clip_start
                end = min(clip_end, chunk[-1]["end"]) - clip_start
                text = "".join(x["word"] for x in chunk).strip()
                if text and end > start:
                    entries.append((start, end, text))
        else:
            start = max(clip_start, seg["start"]) - clip_start
            end = min(clip_end, seg["end"]) - clip_start
            if end > start and seg["text"]:
                entries.append((start, end, seg["text"]))
    entries.sort(key=lambda x: x[0])
    with open(path, "w", encoding="utf-8-sig") as f:
        for i, (start, end, text) in enumerate(entries, 1):
            safe = text.replace("-->", "→")
            f.write(f"{i}\n{srt_time(start)} --> {srt_time(end)}\n{safe}\n\n")

def render_clip(video_path, clip, segments, output_path, index, vertical=True):
    folder = output_path.parent
    srt_path = folder / f"clip_{index:02d}.srt"
    make_srt(segments, clip["start"], clip["end"], srt_path)
    duration = clip["end"] - clip["start"]
    if vertical:
        vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    else:
        vf = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
    # Use the SRT filter with a relative filename in the output folder.
    subtitle_filter = f"subtitles=filename='{srt_path.name}':force_style='FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=100'"
    cmd = [
        FFMPEG, "-y", "-ss", f"{clip['start']:.3f}", "-i", str(video_path),
        "-t", f"{duration:.3f}", "-vf", f"{vf},{subtitle_filter}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(output_path)
    ]
    try:
        run_command(cmd, cwd=str(folder))
    except Exception:
        # Fallback still creates a clip if this FFmpeg build cannot burn subtitles.
        cmd = [
            FFMPEG, "-y", "-ss", f"{clip['start']:.3f}", "-i", str(video_path),
            "-t", f"{duration:.3f}", "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(output_path)
        ]
        run_command(cmd, cwd=str(folder))
    return srt_path

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#21164a,#4931a8); color: white; margin-bottom: 1.2rem;}
.hero h1 {color:white; margin:0;}
.small-muted {color:#9aa0aa;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🎬 ClipSpark AI</h1><p>Paste a video URL. Get AI-assisted vertical shorts with captions.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Clip settings")
    clip_duration = st.slider("Clip duration (seconds)", 15, 30, 22)
    clip_count = st.selectbox("Number of clips", [3, 5, 10], index=2)
    aspect = st.selectbox("Output format", ["9:16 Vertical", "16:9 Landscape"])
    model_name = st.selectbox("Transcription model", ["small", "base"], index=0)
    st.caption("Small is more accurate but slower. Base is faster on CPU.")
    st.divider()
    st.caption("Processing runs locally on your PC. First run downloads the speech model.")

url = st.text_input("Video URL", placeholder="Paste a public video URL here (for example, a YouTube link)")
st.caption("Use public videos you own or have permission to process. Some sites restrict downloading, and availability can change.")
start_button = st.button("✨ Generate my clips", type="primary", use_container_width=True)

if start_button:
    if not valid_video_url(url):
        st.error("Please paste a valid URL beginning with http:// or https://.")
        st.stop()
    if not FFMPEG or not FFPROBE:
        st.error("FFmpeg or FFprobe was not found. Check that D:\\bin is in PATH, then restart VS Code.")
        st.stop()
    job_dir = Path(tempfile.mkdtemp(prefix="clipspark_", dir=str(OUTPUT_DIR)))
    progress = st.progress(0, text="Starting…")
    status = st.empty()
    try:
        status.info("1/4 — Downloading the video from the URL…")
        video_path, title = download_video(url.strip(), job_dir)
        progress.progress(20, text="Video downloaded")
        st.write(f"**Source:** {title}")
        status.info("2/4 — Transcribing speech with the local AI model. The first run may take longer.")
        # Honor the model selected in the sidebar.
        @st.cache_resource(show_spinner=False)
        def get_model(name):
            return WhisperModel(name, device="cpu", compute_type="int8")
        model = get_model(model_name)
        raw_segments, language = model.transcribe(str(video_path), beam_size=3, word_timestamps=True, vad_filter=True)
        segments = []
        for seg in raw_segments:
            words = [{"start": float(w.start), "end": float(w.end), "word": w.word} for w in (seg.words or [])]
            segments.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip(), "words": words})
        if not segments:
            raise RuntimeError("No speech was detected. This version needs spoken audio to find moments.")
        progress.progress(55, text="Transcription complete")
        status.info("3/4 — Ranking transcript moments and selecting clips…")
        clips = choose_clips(segments, int(clip_duration), int(clip_count))
        if not clips:
            raise RuntimeError("Not enough spoken content was found to create clips. Try a longer video with clear speech.")
        progress.progress(65, text=f"Selected {len(clips)} clip(s)")
        status.info("4/4 — Rendering vertical clips and captions. This can take a while on CPU.")
        rendered = []
        vertical = aspect.startswith("9:16")
        for i, clip in enumerate(clips, 1):
            out_path = job_dir / f"ClipSpark_clip_{i:02d}.mp4"
            render_clip(video_path, clip, segments, out_path, i, vertical=vertical)
            rendered.append((out_path, clip))
            progress.progress(65 + int(35 * i / len(clips)), text=f"Rendered {i}/{len(clips)}")
        status.success(f"Done! Created {len(rendered)} clip(s). Transcription language detected: {language}.")
        st.subheader("Your clips")
        for i, (path, clip) in enumerate(rendered, 1):
            with st.container(border=True):
                st.markdown(f"**Clip {i}** · {clip['start']:.1f}s–{clip['end']:.1f}s · Relevance score {clip['score']:.1f}")
                st.write(clip["text"][:500] + ("…" if len(clip["text"]) > 500 else ""))
                st.video(str(path))
                with open(path, "rb") as f:
                    st.download_button(
                        f"⬇️ Download clip {i}", data=f.read(),
                        file_name=path.name, mime="video/mp4",
                        key=f"download_{job_dir.name}_{i}"
                    )
                srt = job_dir / f"clip_{i:02d}.srt"
                if srt.exists():
                    with open(srt, "rb") as f:
                        st.download_button(
                            f"Download captions (.srt) for clip {i}", data=f.read(),
                            file_name=srt.name, mime="application/x-subrip",
                            key=f"srt_{job_dir.name}_{i}"
                        )
        st.info("Clip selection is AI-assisted and transcript-based; it does not guarantee virality. Review clips before posting.")
    except Exception as exc:
        status.error("The job could not be completed.")
        st.error(str(exc))
        st.markdown("**Troubleshooting:** Check your internet connection, confirm the URL is publicly accessible, and make sure FFmpeg works in a fresh terminal. Some platforms block downloading or require a different supported URL.")
