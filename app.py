import os, uuid, json, subprocess, shutil, re
from pathlib import Path
from urllib.parse import urlparse
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
UPLOADS, OUTPUTS = ROOT / "uploads", ROOT / "outputs"
UPLOADS.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True)
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
ALLOWED = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

def run(cmd, timeout=3600):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout or "Command failed")[-2000:])
    return p.stdout

def duration_of(path):
    out = run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(path)])
    return max(1.0, float(out.strip()))

def fetch_youtube(url, dest):
    # Download only content the user owns or is authorized to use.
    if not re.match(r"^https?://", url, re.I):
        raise ValueError("Please enter a valid http/https video URL.")
    host = (urlparse(url).hostname or "").lower()
    if not (host == "youtube.com" or host.endswith(".youtube.com") or host == "youtu.be" or host.endswith(".youtu.be")):
        raise ValueError("For URL import, enter a YouTube video URL.")
    if not shutil.which("yt-dlp"):
        raise RuntimeError("yt-dlp is not installed on this server. See README_HI.md.")
    # yt-dlp uses a best compatible MP4 output where possible.
    run(["yt-dlp", "--no-playlist", "--max-filesize", "1G", "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
         "--merge-output-format", "mp4", "-o", str(dest), url], timeout=7200)
    if not dest.exists():
        # yt-dlp may append a container extension depending on selected format
        candidates = list(dest.parent.glob(dest.stem + "*"))
        candidates = [p for p in candidates if p.is_file() and p.suffix.lower() in ALLOWED]
        if candidates:
            candidates.sort(key=lambda p:p.stat().st_size, reverse=True)
            return candidates[0]
        raise RuntimeError("The YouTube download did not produce a supported video file.")
    return dest

def ai_clips(path, duration, count, clip_len):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        count = min(count, max(1, int(duration // max(clip_len, 1))))
        starts = ([max(0, (duration-clip_len)/2)] if count == 1 else
                  [max(0, (duration-clip_len)*i/(count-1)) for i in range(count)])
        return [{"start":round(s,2),"end":round(min(duration,s+clip_len),2),"title":f"Clip {i+1}",
                 "reason":"Demo selection: evenly spaced clip (AI key not configured)."} for i,s in enumerate(starts)]
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    # Extract audio to a temporary compressed file for transcription.
    audio = path.with_name(path.stem + "_audio.mp3")
    try:
        run(["ffmpeg","-y","-i",str(path),"-vn","-ac","1","-ar","16000","-b:a","48k",str(audio)],timeout=7200)
        with open(audio,"rb") as f:
            transcript = client.audio.transcriptions.create(model="whisper-1",file=f,response_format="verbose_json")
        raw = transcript.model_dump() if hasattr(transcript,"model_dump") else {}
        segs = raw.get("segments", [])
        if not segs:
            segs = [{"start":0,"end":min(duration,clip_len),"text":getattr(transcript,"text","")}]
        compact = [{"start":s.get("start",0),"end":s.get("end",0),"text":s.get("text","")} for s in segs[:2500]]
        prompt = {"duration_seconds":duration,"requested_clips":count,"target_clip_seconds":clip_len,
          "transcript_segments":compact,
          "instruction":"Select distinct engaging self-contained short-video moments from transcript. Prefer useful, surprising, funny, emotional or curiosity-inducing moments with a strong hook. Return JSON only {\"clips\":[{\"start\":number,\"end\":number,\"title\":\"short title\",\"reason\":\"brief reason\"}]}. Keep within duration and target length, no overlap where possible. Do not promise virality."}
        response = client.chat.completions.create(model=os.getenv("OPENAI_TEXT_MODEL","gpt-4o-mini"),
          messages=[{"role":"user","content":json.dumps(prompt,ensure_ascii=False)}],
          response_format={"type":"json_object"},temperature=0.3)
        items = json.loads(response.choices[0].message.content).get("clips",[])
        out=[]
        for c in items[:count]:
            try:
                s=max(0,float(c["start"])); e=min(duration,float(c["end"]))
                if e-s<5: continue
                e=min(e,s+clip_len)
                out.append({"start":round(s,2),"end":round(e,2),"title":str(c.get("title","Highlight"))[:80],
                            "reason":str(c.get("reason","AI-selected highlight"))[:180]})
            except (ValueError,TypeError,KeyError): pass
        return out or [{"start":0,"end":min(duration,clip_len),"title":"Opening clip","reason":"Fallback clip"}]
    finally:
        audio.unlink(missing_ok=True)

def render_clip(src, clip, outpath, captions=False):
    start=clip["start"]; length=max(1,clip["end"]-start)
    # Center crop to 9:16; subtitles are not burned in this MVP.
    vf="scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    run(["ffmpeg","-y","-ss",str(start),"-i",str(src),"-t",str(length),"-vf",vf,
         "-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac","-b:a","128k",
         "-movflags","+faststart",str(outpath)],timeout=7200)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/create")
def create():
    source=None
    try:
        count=max(1,min(10,int(request.form.get("count",5))))
        clip_len=max(10,min(90,int(request.form.get("length",30))))
        url=(request.form.get("youtube_url") or "").strip()
        job=uuid.uuid4().hex[:12]
        if url:
            source=UPLOADS/f"{job}.mp4"
            source=fetch_youtube(url,source)
        else:
            file=request.files.get("video")
            if not file or not file.filename:
                return jsonify(error="Upload a video or paste a YouTube URL."),400
            ext=Path(secure_filename(file.filename)).suffix.lower()
            if ext not in ALLOWED: return jsonify(error="Supported: MP4, MOV, MKV, WEBM, AVI, M4V."),400
            source=UPLOADS/f"{job}{ext}"; file.save(source)
        duration=duration_of(source)
        clips=ai_clips(source,duration,count,clip_len)
        results=[]
        for i,c in enumerate(clips,1):
            name=f"{job}_short_{i}.mp4"; out=OUTPUTS/name
            render_clip(source,c,out)
            results.append({"title":c["title"],"reason":c["reason"],"start":c["start"],"end":c["end"],"url":f"/outputs/{name}"})
        return jsonify(clips=results,ai_enabled=bool(os.getenv("OPENAI_API_KEY")),
          note=("AI transcript-based highlight selection enabled." if os.getenv("OPENAI_API_KEY")
                else "Demo mode: clips are sampled across the video. Configure OPENAI_API_KEY for transcript-based highlight selection."))
    except Exception as e:
        return jsonify(error=str(e)),500
    finally:
        if source:
            try: source.unlink(missing_ok=True)
            except Exception: pass

@app.get("/outputs/<path:filename>")
def output_file(filename):
    # Basic filename allow-list: only generated MP4 outputs are served.
    if not filename.endswith(".mp4"): return "Not found",404
    return send_from_directory(OUTPUTS,filename,as_attachment=True)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT","5000")),debug=False)
