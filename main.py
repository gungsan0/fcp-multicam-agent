"""
FCP MultiCam Agent — FastAPI backend
멀티캠 음악 공연 자동 편집기 + 유튜브 썸네일 생성기
"""
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import AsyncGenerator

import httpx
import librosa
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ──────────────────────────────────────────────
# 경로 상수
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
_IS_VERCEL = os.environ.get("VERCEL") or os.environ.get("NOW_REGION")
_TMP_ROOT = Path("/tmp") if _IS_VERCEL else BASE_DIR
ASSETS_DIR = _TMP_ROOT / "assets"
FONTS_DIR = _TMP_ROOT / "fonts"
OUTPUT_DIR = _TMP_ROOT / "output"
CONFIG_PATH = _TMP_ROOT / "config.json"

for d in [ASSETS_DIR, FONTS_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ASR 직렬화 락 (numba/mlx 동시 호출 방지)
_asr_lock = asyncio.Lock()

# ASR 캐시: content_hash → segments
_asr_cache: dict[str, list] = {}

# 로그 브로드캐스트 큐 (SSE /logs 용)
_log_subscribers: list[asyncio.Queue] = []

def log(msg: str, level: str = "info"):
    """서버 콘솔 + SSE 로그 동시 출력"""
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] [{level.upper()}] {msg}"
    print(line, flush=True)
    payload = json.dumps({"ts": ts, "level": level, "msg": msg})
    for q in list(_log_subscribers):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass

# ──────────────────────────────────────────────
# 기본 config
# ──────────────────────────────────────────────
DEFAULT_CONFIG = {
    "channel_name": "",
    "channel_tag": "",
    "default_performer": "",
    "logo_path": "",
    "font_title": "NanumSquareExtraBold",
    "font_sub": "NanumGothic",
    "accent_color": "#22c55e",
    "layout": "B",
    "overlay_alpha": 0.45,
    "text_color": "#FFFFFF",
    "anthropic_api_key": "",
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return {**DEFAULT_CONFIG, **json.loads(CONFIG_PATH.read_text())}
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(data: dict):
    CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ──────────────────────────────────────────────
# FastAPI 앱
# ──────────────────────────────────────────────
app = FastAPI(title="FCP MultiCam Agent")
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


# ── config 엔드포인트 ──────────────────────────
@app.get("/config")
def get_config():
    return load_config()


@app.post("/config")
async def post_config(request_body: dict):
    cfg = {**DEFAULT_CONFIG, **request_body}
    save_config(cfg)
    return cfg


@app.get("/config/reset")
def reset_config():
    if CONFIG_PATH.exists():
        CONFIG_PATH.unlink()
    return DEFAULT_CONFIG


# ── 실시간 로그 스트림 (SSE) ──────────────────
@app.get("/logs")
async def stream_logs():
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _log_subscribers.append(q)

    async def generate():
        try:
            yield "data: {\"ts\":\"\",\"level\":\"info\",\"msg\":\"로그 연결됨\"}\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield "data: {\"ts\":\"\",\"level\":\"ping\",\"msg\":\"…\"}\n\n"
        finally:
            _log_subscribers.remove(q)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── 로고 업로드 ────────────────────────────────
@app.post("/upload/logo")
async def upload_logo(file: UploadFile = File(...)):
    dest = ASSETS_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)
    return {"logo_path": f"assets/{file.filename}"}


# ── 메인 편집 파이프라인 (SSE) ─────────────────
@app.post("/process")
async def process_videos(
    cameras: list[UploadFile] = File(...),
    master_audio: UploadFile = File(None),
    song_title: str = Form(""),
    performer: str = Form(""),
    date_venue: str = Form(""),
    cam_roles: str = Form("{}"),
    min_cut_sec: float = Form(3.0),
    max_cut_sec: float = Form(12.0),
    transition_type: str = Form("Dissolve"),
    transition_dur: float = Form(0.5),
    layout: str = Form(""),
    accent_color: str = Form(""),
    overlay_alpha: float = Form(-1.0),
    channel_tag: str = Form(""),
    logo_path: str = Form(""),
    subtitle_font: str = Form("Helvetica Neue"),
    subtitle_size: int = Form(48),
    subtitle_color: str = Form("#FFFFFF"),
    subtitle_bold: bool = Form(True),
    subtitle_alignment: str = Form("center"),
    subtitle_position: str = Form("bottom"),
):
    cfg = load_config()

    # config 기본값 병합
    _performer = performer or cfg.get("default_performer", "")
    _layout = layout or cfg.get("layout", "B")
    _accent = accent_color or cfg.get("accent_color", "#22c55e")
    _alpha = overlay_alpha if overlay_alpha >= 0 else cfg.get("overlay_alpha", 0.45)
    _channel_tag = channel_tag or cfg.get("channel_tag", "")
    _logo = logo_path or cfg.get("logo_path", "")
    _font_title = cfg.get("font_title", "NanumSquareExtraBold")
    _font_sub = cfg.get("font_sub", "NanumGothic")
    _text_color = cfg.get("text_color", "#FFFFFF")

    try:
        roles = json.loads(cam_roles)
    except Exception:
        roles = {}

    # 임시 작업 디렉토리
    work_dir = OUTPUT_DIR / f"job_{int(time.time())}"
    work_dir.mkdir(parents=True)

    async def stream() -> AsyncGenerator[str, None]:
        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            # ─── 파일 저장 (청크 스트리밍, 진행 표시) ────────
            log(f"파일 저장 시작: 카메라 {len(cameras)}개")
            yield sse("progress", {"step": "upload", "msg": f"파일 저장 중… (카메라 {len(cameras)}개)"})

            cam_paths = []
            for i, cam_file in enumerate(cameras):
                ext = Path(cam_file.filename).suffix or ".mp4"
                dest = work_dir / f"cam{i+1}{ext}"
                log(f"  cam{i+1} 저장 중: {cam_file.filename}")
                yield sse("progress", {"step": "upload", "msg": f"cam{i+1} 저장 중… ({cam_file.filename})"})
                # 청크 단위로 저장 (메모리 절약)
                with open(dest, "wb") as f:
                    while True:
                        chunk = await cam_file.read(1024 * 1024)  # 1MB씩
                        if not chunk:
                            break
                        f.write(chunk)
                size_mb = dest.stat().st_size / 1024 / 1024
                log(f"  cam{i+1} 저장 완료: {size_mb:.1f}MB → {dest}")
                cam_paths.append(dest)

            master_path = None
            if master_audio:
                ext = Path(master_audio.filename).suffix or ".wav"
                master_path = work_dir / f"master{ext}"
                log(f"마스터 오디오 저장 중: {master_audio.filename}")
                yield sse("progress", {"step": "upload", "msg": f"마스터 오디오 저장 중… ({master_audio.filename})"})
                with open(master_path, "wb") as f:
                    while True:
                        chunk = await master_audio.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                log(f"마스터 오디오 저장 완료: {master_path.stat().st_size/1024/1024:.1f}MB")

            if not master_path:
                master_path = cam_paths[0]
                log(f"마스터 오디오 없음 → cam1 사용: {master_path}")

            yield sse("progress", {"step": "upload", "msg": f"파일 저장 완료 ✓ ({len(cam_paths)}개)", "done": True})

            # ─── Step 1: SYNC ─────────────────────────────
            log("SYNC 시작: 오디오 교차상관 계산")
            yield sse("progress", {"step": "sync", "msg": "오디오 파형 교차상관 → offset 계산 중…"})
            await asyncio.sleep(0.5)

            offsets = await asyncio.get_event_loop().run_in_executor(
                None, compute_offsets, master_path, cam_paths
            )
            log(f"SYNC 완료: offsets={[round(o,3) for o in offsets]}")
            yield sse("progress", {"step": "sync", "msg": f"싱크 완료: offsets={[round(o,3) for o in offsets]}s", "done": True})

            # ─── Step 2: ASR ──────────────────────────────
            log("ASR 시작: mlx-whisper 로드 중 (첫 실행 시 모델 다운로드 ~수분)")
            yield sse("progress", {"step": "asr", "msg": "mlx-whisper 가사 인식 중… (첫 실행 시 모델 다운로드)"})

            async with _asr_lock:
                segments = await asyncio.get_event_loop().run_in_executor(
                    None, run_asr_cached, master_path
                )
            log(f"ASR 완료: {len(segments)}개 세그먼트")
            yield sse("progress", {"step": "asr", "msg": f"가사 인식 완료: {len(segments)}개 세그먼트", "done": True})

            # SRT/VTT 즉시 생성
            srt_path = work_dir / "subtitles.srt"
            vtt_path = work_dir / "subtitles.vtt"
            srt_path.write_text(generate_srt(segments), encoding="utf-8")
            vtt_path.write_text(generate_vtt(segments), encoding="utf-8")

            # ─── Step 3: CAM SELECT ───────────────────────
            log("CAM SELECT 시작")
            yield sse("progress", {"step": "cam_select", "msg": "에너지 분석 + 카메라 선택 중…"})

            duration = get_video_duration(master_path)
            log(f"영상 길이: {duration:.1f}s")
            cam_role_list = [roles.get(f"cam{i+1}", assign_default_role(i, len(cam_paths))) for i in range(len(cam_paths))]
            cuts, climax_start, climax_end = await asyncio.get_event_loop().run_in_executor(
                None, compute_cuts, master_path, len(cam_paths), cam_role_list, min_cut_sec, max_cut_sec, duration
            )
            log(f"CAM SELECT 완료: {len(cuts)}컷, 클라이맥스 {climax_start:.1f}~{climax_end:.1f}s")
            yield sse("progress", {"step": "cam_select", "msg": f"컷 배치 완료: {len(cuts)}컷, 클라이맥스 {climax_start:.1f}s~{climax_end:.1f}s", "done": True, "cuts": cuts})

            # ─── Step 4: SUBTITLE ────────────────────────
            log("SUBTITLE 삽입 중")
            yield sse("progress", {"step": "subtitle", "msg": "FCPXML 자막 타이틀 삽입 중…"})
            await asyncio.sleep(0.3)

            # ─── Step 5: TRANSITION ──────────────────────
            log(f"TRANSITION 삽입: {transition_type} {transition_dur}s")
            yield sse("progress", {"step": "transition", "msg": f"전환 효과 삽입: {transition_type} {transition_dur}s"})
            await asyncio.sleep(0.3)

            # ─── FCPXML 생성 ─────────────────────────────
            log("FCPXML 생성 중")
            fcpxml_path = work_dir / "multicam_draft.fcpxml"
            frame_rate = get_video_framerate(cam_paths[0])
            build_fcpxml(
                cam_paths=cam_paths,
                offsets=offsets,
                cuts=cuts,
                segments=segments,
                duration=duration,
                frame_rate=frame_rate,
                song_title=song_title,
                transition_type=transition_type,
                transition_dur=transition_dur,
                output_path=fcpxml_path,
                subtitle_font=subtitle_font,
                subtitle_size=subtitle_size,
                subtitle_color=subtitle_color,
                subtitle_bold=subtitle_bold,
                subtitle_alignment=subtitle_alignment,
                subtitle_position=subtitle_position,
            )
            log(f"FCPXML 저장 완료: {fcpxml_path}")
            yield sse("progress", {"step": "transition", "msg": "FCPXML 생성 완료", "done": True})

            # ─── Step 6: THUMB EXTRACT ───────────────────
            log(f"THUMB EXTRACT 시작: 클라이맥스 {climax_start:.1f}~{climax_end:.1f}s")
            yield sse("progress", {"step": "thumb_extract", "msg": "베스트 프레임 후보 추출 중 (ffmpeg)…"})
            await asyncio.sleep(0.5)

            frames_dir = work_dir / "frames"
            frames_dir.mkdir()
            frame_paths = await asyncio.get_event_loop().run_in_executor(
                None, extract_candidate_frames, cam_paths[0], climax_start, climax_end, frames_dir
            )
            log(f"THUMB EXTRACT 완료: {len(frame_paths)}장")
            yield sse("progress", {"step": "thumb_extract", "msg": f"프레임 {len(frame_paths)}장 추출 완료", "done": True})

            # ─── Step 7: THUMB SCORE ─────────────────────
            log("THUMB SCORE 시작: CLIP 채점 (첫 실행 시 모델 다운로드 ~1GB)")
            yield sse("progress", {"step": "thumb_score", "msg": "CLIP 로컬 채점 중… (첫 실행 시 모델 다운로드 ~1GB)"})
            await asyncio.sleep(0.5)

            scored = await score_frames_parallel(frame_paths)
            scored.sort(key=lambda x: x["total"], reverse=True)
            top3 = scored[:3]
            log(f"THUMB SCORE 완료: 상위 3위 선택")
            yield sse("progress", {
                "step": "thumb_score",
                "msg": f"채점 완료, 상위 3위 선택",
                "done": True,
                "top_frames": [{"path": t["path"], "total": t["total"]} for t in top3]
            })

            # ─── Step 8: THUMB COMPOSE ───────────────────
            log("THUMB COMPOSE 시작: Pillow 합성")
            yield sse("progress", {"step": "thumb_compose", "msg": "Pillow 썸네일 합성 중…"})
            await asyncio.sleep(0.5)

            best_frame = top3[0]["path"] if top3 else (str(frame_paths[0]) if frame_paths else None)
            thumb_params = {
                "song_title": song_title,
                "performer": _performer,
                "date_venue": date_venue,
                "channel_tag": _channel_tag,
                "logo_path": _logo,
                "font_title": _font_title,
                "font_sub": _font_sub,
                "text_color": _text_color,
                "accent_color": _accent,
                "overlay_alpha": _alpha,
                "layout": _layout,
            }
            thumb_files = await asyncio.get_event_loop().run_in_executor(
                None, compose_all_thumbnails, best_frame, thumb_params, work_dir
            )

            # ZIP 묶음
            zip_path = work_dir / "thumbnails.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for tf in thumb_files:
                    if Path(tf).exists():
                        zf.write(tf, Path(tf).name)

            yield sse("progress", {"step": "thumb_compose", "msg": f"썸네일 {len(thumb_files)}개 합성 완료", "done": True})

            # ─── Step 9: FCPXML EXPORT ───────────────────
            yield sse("progress", {"step": "fcpxml_export", "msg": "FCPXML 파일 저장 완료"})
            await asyncio.sleep(0.5)

            rel_fcpxml = str(fcpxml_path.relative_to(BASE_DIR))
            rel_thumbs = [str(Path(t).relative_to(BASE_DIR)) for t in thumb_files if Path(t).exists()]
            rel_zip = str(zip_path.relative_to(BASE_DIR))

            yield sse("done", {
                "fcpxml": rel_fcpxml,
                "thumbnails": rel_thumbs,
                "zip": rel_zip,
                "cuts": cuts,
                "top_frames": [{"path": str(Path(t["path"]).relative_to(BASE_DIR)), "total": t["total"]} for t in top3],
                "srt": str(srt_path.relative_to(BASE_DIR)),
                "vtt": str(vtt_path.relative_to(BASE_DIR)),
            })

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log(f"파이프라인 오류: {e}\n{tb}", level="error")
            yield sse("error", {"msg": str(e), "trace": tb})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 썸네일 재합성 (라이브 에디터) ─────────────────
@app.post("/thumbnail/compose")
async def thumbnail_compose(params: dict):
    frame_path = params.get("frame_path", "")
    if not frame_path or not Path(BASE_DIR / frame_path).exists():
        raise HTTPException(400, "프레임 경로 없음")

    out_dir = OUTPUT_DIR / "live_thumb"
    out_dir.mkdir(exist_ok=True)

    result = await asyncio.get_event_loop().run_in_executor(
        None, compose_all_thumbnails, str(BASE_DIR / frame_path), params, out_dir
    )
    rel = [str(Path(r).relative_to(BASE_DIR)) for r in result if Path(r).exists()]
    return {"thumbnails": rel}


# ── 자막 생성기 (URL 또는 파일) ──────────────────
@app.post("/subtitle/generate")
async def subtitle_generate_endpoint(
    url: str = Form(""),
    file: UploadFile = File(None),
):
    work_dir = OUTPUT_DIR / f"sub_{int(time.time())}"
    work_dir.mkdir(parents=True)

    async def stream() -> AsyncGenerator[str, None]:
        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        try:
            audio_path = None
            if url.strip():
                log(f"URL 다운로드: {url[:80]}")
                yield sse("progress", {"msg": f"오디오 다운로드 중… ({url[:60]})"})
                audio_path = await asyncio.get_event_loop().run_in_executor(
                    None, download_url_audio, url.strip(), work_dir
                )
                yield sse("progress", {"msg": "다운로드 완료 ✓"})
            elif file and file.filename:
                ext = Path(file.filename).suffix or ".mp4"
                audio_path = work_dir / f"upload{ext}"
                yield sse("progress", {"msg": f"파일 저장 중… ({file.filename})"})
                with open(audio_path, "wb") as f:
                    while True:
                        chunk = await file.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                yield sse("progress", {"msg": "업로드 완료 ✓"})
            else:
                yield sse("error", {"msg": "URL 또는 파일을 제공해 주세요"})
                return

            log("자막 ASR 시작")
            yield sse("progress", {"msg": "음성 인식 중… (mlx-whisper, 첫 실행 시 모델 다운로드)"})
            async with _asr_lock:
                segments = await asyncio.get_event_loop().run_in_executor(
                    None, run_asr_cached, audio_path
                )
            yield sse("progress", {"msg": f"인식 완료: {len(segments)}개 세그먼트"})

            srt_path = work_dir / "subtitles.srt"
            vtt_path = work_dir / "subtitles.vtt"
            txt_path = work_dir / "subtitles.txt"
            srt_path.write_text(generate_srt(segments), encoding="utf-8")
            vtt_path.write_text(generate_vtt(segments), encoding="utf-8")
            txt_path.write_text(
                "\n".join(s.get("text", "").strip() for s in segments if s.get("text", "").strip()),
                encoding="utf-8"
            )
            yield sse("done", {
                "srt": str(srt_path.relative_to(BASE_DIR)),
                "vtt": str(vtt_path.relative_to(BASE_DIR)),
                "txt": str(txt_path.relative_to(BASE_DIR)),
                "count": len(segments),
            })
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log(f"자막 생성 오류: {e}", level="error")
            yield sse("error", {"msg": str(e), "trace": tb})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── API 키 연결 테스트 ────────────────────────
@app.post("/test/api-key")
async def test_api_key(body: dict):
    key = body.get("api_key", "").strip()
    if not key:
        return {"ok": False, "error": "API 키 없음"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        if resp.status_code == 200:
            return {"ok": True, "model": "claude-haiku-4-5-20251001"}
        err = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
        return {"ok": False, "error": err}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── FCPXML 직접 열기 ──────────────────────────
@app.get("/open/fcpxml")
def open_fcpxml(path: str):
    full = BASE_DIR / path
    if not full.exists():
        raise HTTPException(404, "파일 없음")
    fcp_path = "/Volumes/Samsung T7/Applications/Final Cut Pro.app"
    subprocess.Popen(["open", "-a", fcp_path, str(full)])
    return {"ok": True}


# ── CapCut 내보내기 ──────────────────────────────────
@app.post("/process/capcut")
async def process_capcut(
    cameras: list[UploadFile] = File(...),
    master_audio: UploadFile = File(None),
    song_title: str = Form(""),
    performer: str = Form(""),
    date_venue: str = Form(""),
    cam_roles: str = Form("{}"),
    min_cut_sec: float = Form(3.0),
    max_cut_sec: float = Form(12.0),
):
    cfg = load_config()
    try:
        roles = json.loads(cam_roles)
    except Exception:
        roles = {}

    work_dir = OUTPUT_DIR / f"capcut_{int(time.time())}"
    work_dir.mkdir(parents=True)

    async def stream() -> AsyncGenerator[str, None]:
        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            # ── 파일 저장 ────────────────────────────────
            yield sse("progress", {"step": "upload", "msg": f"파일 저장 중… (카메라 {len(cameras)}개)"})
            cam_paths = []
            for i, cam_file in enumerate(cameras):
                ext = Path(cam_file.filename).suffix or ".mp4"
                dest = work_dir / f"cam{i+1}{ext}"
                log(f"  cam{i+1} 저장: {cam_file.filename}")
                yield sse("progress", {"step": "upload", "msg": f"cam{i+1} 저장 중… ({cam_file.filename})"})
                with open(dest, "wb") as f:
                    while True:
                        chunk = await cam_file.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                cam_paths.append(dest)

            master_path = None
            if master_audio:
                ext = Path(master_audio.filename).suffix or ".wav"
                master_path = work_dir / f"master{ext}"
                yield sse("progress", {"step": "upload", "msg": f"마스터 오디오 저장 중…"})
                with open(master_path, "wb") as f:
                    while True:
                        chunk = await master_audio.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            if not master_path:
                master_path = cam_paths[0]

            yield sse("progress", {"step": "upload", "msg": "저장 완료 ✓", "done": True})

            # ── SYNC ─────────────────────────────────────
            yield sse("progress", {"step": "sync", "msg": "오디오 싱크 계산 중…"})
            offsets = await asyncio.get_event_loop().run_in_executor(
                None, compute_offsets, master_path, cam_paths
            )
            log(f"CapCut SYNC: offsets={[round(o,3) for o in offsets]}")
            yield sse("progress", {"step": "sync", "msg": f"싱크 완료: {[round(o,3) for o in offsets]}s", "done": True})

            # ── ASR ──────────────────────────────────────
            yield sse("progress", {"step": "asr", "msg": "가사 인식 중… (mlx-whisper)"})
            async with _asr_lock:
                segments = await asyncio.get_event_loop().run_in_executor(
                    None, run_asr_cached, master_path
                )
            yield sse("progress", {"step": "asr", "msg": f"가사 인식 완료: {len(segments)}개", "done": True})

            # ── CAM SELECT ────────────────────────────────
            yield sse("progress", {"step": "cam_select", "msg": "카메라 배치 계산 중…"})
            duration = get_video_duration(master_path)
            cam_role_list = [roles.get(f"cam{i+1}", assign_default_role(i, len(cam_paths))) for i in range(len(cam_paths))]
            cuts, climax_start, climax_end = await asyncio.get_event_loop().run_in_executor(
                None, compute_cuts, master_path, len(cam_paths), cam_role_list, min_cut_sec, max_cut_sec, duration
            )
            yield sse("progress", {"step": "cam_select", "msg": f"컷 배치: {len(cuts)}컷", "done": True, "cuts": cuts})

            # ── CapCut 드래프트 생성 ──────────────────────
            yield sse("progress", {"step": "capcut_export", "msg": "CapCut 프로젝트 파일 생성 중…"})
            draft_dir, draft_json_path = build_capcut_draft(
                cam_paths=cam_paths,
                offsets=offsets,
                master_path=master_path,
                cuts=cuts,
                segments=segments,
                duration=duration,
                song_title=song_title or "멀티캠 프로젝트",
                work_dir=work_dir,
            )
            log(f"CapCut 드래프트 저장: {draft_json_path}")
            yield sse("progress", {"step": "capcut_export", "msg": "CapCut 프로젝트 생성 완료 ✓", "done": True})

            rel_draft = str(draft_json_path.relative_to(BASE_DIR))
            rel_dir = str(draft_dir.relative_to(BASE_DIR))

            yield sse("done", {
                "draft_json": rel_draft,
                "draft_dir": rel_dir,
                "draft_dir_abs": str(draft_dir),
                "cuts": cuts,
            })

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log(f"CapCut 파이프라인 오류: {e}\n{tb}", level="error")
            yield sse("error", {"msg": str(e), "trace": tb})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/open/capcut")
def open_capcut_folder(path: str):
    """생성된 CapCut 프로젝트 폴더를 Finder에서 열기"""
    full = BASE_DIR / path
    if not full.exists():
        raise HTTPException(404, "폴더 없음")
    subprocess.Popen(["open", str(full)])
    return {"ok": True}


@app.get("/install/capcut")
def install_capcut_project(path: str):
    """CapCut 프로젝트 폴더를 CapCut 기본 경로에 복사"""
    src = BASE_DIR / path
    if not src.exists():
        raise HTTPException(404, "폴더 없음")
    capcut_projects = Path.home() / "Movies" / "CapCut" / "User Data" / "Projects"
    capcut_projects.mkdir(parents=True, exist_ok=True)
    dest = capcut_projects / src.name
    if dest.exists():
        import shutil
        shutil.rmtree(dest)
    import shutil
    shutil.copytree(src, dest)
    log(f"CapCut 프로젝트 설치 완료: {dest}")
    subprocess.Popen(["open", str(capcut_projects)])
    return {"ok": True, "path": str(dest)}


# ── HTML 메인 페이지 ──────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = BASE_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>index.html 없음</h1>")


@app.get("/capcut", response_class=HTMLResponse)
def capcut_page():
    html_path = BASE_DIR / "capcut.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>capcut.html 없음</h1>")


# ══════════════════════════════════════════════
# 오디오 처리 함수들
# ══════════════════════════════════════════════

def get_audio_array(video_path: Path, sr: int = 22050) -> np.ndarray:
    """ffmpeg로 모노 오디오 추출 후 librosa 로드"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", str(sr), tmp_path],
            capture_output=True, check=True
        )
        y, _ = librosa.load(tmp_path, sr=sr, mono=True)
        return y
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def compute_offsets(master_path: Path, cam_paths: list[Path]) -> list[float]:
    """교차상관으로 각 카메라의 마스터 대비 오프셋(초) 계산"""
    sr = 22050
    master_y = get_audio_array(master_path, sr)
    offsets = []
    for cam_path in cam_paths:
        if cam_path == master_path:
            offsets.append(0.0)
            continue
        try:
            cam_y = get_audio_array(cam_path, sr)
            # 교차상관 (최대 30초 탐색)
            max_lag_samples = sr * 30
            min_len = min(len(master_y), len(cam_y), sr * 120)
            m = master_y[:min_len]
            c = cam_y[:min_len]
            corr = np.correlate(m, c[:min(len(c), min_len)], mode="full")
            lag = np.argmax(np.abs(corr)) - (len(c[:min(len(c), min_len)]) - 1)
            lag = max(-max_lag_samples, min(max_lag_samples, lag))
            offsets.append(lag / sr)
        except Exception:
            offsets.append(0.0)
    return offsets


def get_video_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True
    )
    try:
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return 120.0


def get_video_framerate(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True
    )
    try:
        streams = json.loads(result.stdout).get("streams", [])
        for s in streams:
            if s.get("codec_type") == "video":
                fr = s.get("r_frame_rate", "30/1")
                num, den = fr.split("/")
                return float(num) / float(den)
    except Exception:
        pass
    return 29.97


def assign_default_role(idx: int, total: int) -> str:
    if total == 1:
        return "wide"
    if idx == 0:
        return "wide"
    if idx == total - 1:
        return "close"
    return "medium"


def compute_cuts(
    master_path: Path,
    n_cams: int,
    cam_roles: list[str],
    min_cut: float,
    max_cut: float,
    duration: float,
) -> tuple[list[dict], float, float]:
    """에너지 기반 컷 포인트 생성"""
    sr = 22050
    try:
        y = get_audio_array(master_path, sr)
    except Exception:
        y = np.zeros(int(sr * duration))

    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    # 클라이맥스: RMS 상위 20% 구간
    threshold = np.percentile(rms, 80)
    climax_mask = rms >= threshold
    if climax_mask.any():
        climax_start = float(times[np.argmax(climax_mask)])
        last_climax = len(climax_mask) - 1 - np.argmax(climax_mask[::-1])
        climax_end = float(times[min(last_climax, len(times) - 1)])
        climax_end = max(climax_start + 5.0, climax_end)
    else:
        climax_start = duration * 0.6
        climax_end = duration * 0.9

    # 컷 생성
    cuts = []
    t = 0.0
    cam_idx = 0

    # 역할 우선순위 매핑
    def preferred_cam(section: str) -> int:
        role_pref = {
            "intro": "wide",
            "climax": "close",
            "ensemble": "wide",
            "normal": "medium",
        }
        target_role = role_pref.get(section, "medium")
        for i, r in enumerate(cam_roles):
            if r == target_role:
                return i
        return cam_idx % n_cams

    section_idx = 0
    while t < duration - 0.5:
        # 구간 결정
        if t < duration * 0.15:
            section = "intro"
        elif climax_start <= t <= climax_end:
            section = "climax"
        else:
            section = "normal"

        cam = preferred_cam(section)
        # 같은 카메라 연속 방지
        if cuts and cuts[-1]["cam"] == cam and n_cams > 1:
            cam = (cam + 1) % n_cams

        # 에너지 기반 컷 길이
        t_end_frame = min(int((t + max_cut) * sr / hop), len(rms) - 1)
        t_start_frame = int(t * sr / hop)
        local_rms = rms[t_start_frame:t_end_frame]

        if len(local_rms) > 0:
            # 에너지 변화가 큰 지점 찾기
            diff = np.abs(np.diff(local_rms))
            if len(diff) > 0:
                cut_frame = np.argmax(diff)
                cut_time = cut_frame * hop / sr
                cut_len = max(min_cut, min(cut_time + min_cut * 0.5, max_cut))
            else:
                cut_len = (min_cut + max_cut) / 2
        else:
            cut_len = (min_cut + max_cut) / 2

        end_t = min(t + cut_len, duration)
        cuts.append({"cam": cam, "start": round(t, 4), "end": round(end_t, 4)})
        t = end_t
        section_idx += 1

    return cuts, climax_start, climax_end


# ══════════════════════════════════════════════
# ASR
# ══════════════════════════════════════════════

def run_asr_cached(audio_path: Path) -> list[dict]:
    """content hash 기반 캐시로 mlx-whisper 실행"""
    content = audio_path.read_bytes()
    h = hashlib.sha256(content[:1024 * 1024]).hexdigest()[:16]

    if h in _asr_cache:
        return _asr_cache[h]

    # 오디오만 추출
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", "16000", tmp_path],
            capture_output=True, check=True
        )
        import mlx_whisper
        result = mlx_whisper.transcribe(tmp_path, path_or_hf_repo="mlx-community/whisper-small-mlx")
        segments = result.get("segments", [])
        _asr_cache[h] = segments
        return segments
    except Exception as e:
        print(f"ASR 오류: {e}")
        return []
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ══════════════════════════════════════════════
# FCPXML 생성
# ══════════════════════════════════════════════

def rational_time(seconds: float, frame_rate: float = 29.97) -> str:
    """초를 FCPXML rational time 문자열로 변환"""
    if abs(frame_rate - 29.97) < 0.01:
        # 30000/1001 timebase
        frames = round(seconds * 30000 / 1001)
        return f"{frames * 1001}/30000s"
    elif abs(frame_rate - 23.976) < 0.01:
        frames = round(seconds * 24000 / 1001)
        return f"{frames * 1001}/24000s"
    else:
        fr_int = round(frame_rate)
        frames = round(seconds * fr_int)
        return f"{frames}/{fr_int}s"


def build_fcpxml(
    cam_paths: list[Path],
    offsets: list[float],
    cuts: list[dict],
    segments: list[dict],
    duration: float,
    frame_rate: float,
    song_title: str,
    transition_type: str,
    transition_dur: float,
    output_path: Path,
    subtitle_font: str = "Helvetica Neue",
    subtitle_size: int = 48,
    subtitle_color: str = "#FFFFFF",
    subtitle_bold: bool = True,
    subtitle_alignment: str = "center",
    subtitle_position: str = "bottom",
):
    """완전한 FCPXML 멀티캠 드래프트 생성"""
    tb = "30000/1001s" if abs(frame_rate - 29.97) < 0.01 else f"{round(frame_rate)}/1s"

    total_dur_rt = rational_time(duration, frame_rate)
    trans_rt = rational_time(transition_dur, frame_rate)

    # ── asset 목록 ────────────────────────────
    # DTD 규칙: asset에 src 속성 없음 — media-rep 안에만 src 사용
    assets_xml = ""
    for i, cam_path in enumerate(cam_paths):
        uid = f"cam{i+1}_asset"
        abs_path = cam_path.resolve()
        cam_dur = get_video_duration(cam_path)
        cam_dur_rt = rational_time(cam_dur, frame_rate)
        assets_xml += f"""
        <asset id="{uid}" name="{cam_path.stem}" uid="{uid}"
               start="0s" duration="{cam_dur_rt}"
               hasVideo="1" hasAudio="1"
               audioSources="1" audioChannels="2">
            <media-rep kind="original-media" src="file://{abs_path}"/>
        </asset>"""

    # ── mc-angle 목록 ─────────────────────────
    angles_xml = ""
    for i, (cam_path, offset) in enumerate(zip(cam_paths, offsets)):
        uid = f"cam{i+1}_asset"
        cam_dur = get_video_duration(cam_path)

        # ─── 싱크 오프셋 처리 ────────────────────────────────────────────
        # offset > 0 : 이 카메라가 master보다 늦게 시작함
        #   → 멀티캠 타임라인 0부터 배치, 소스는 offset초부터 읽기
        #   → 실제로 읽을 수 있는 길이 = cam_dur - offset (초과하면 검은 화면)
        # offset < 0 : 이 카메라가 master보다 일찍 시작함
        #   → 멀티캠 타임라인 |offset|초부터 배치, 소스는 처음부터 읽기
        #   → 실제 길이는 cam_dur 그대로
        if offset >= 0:
            source_start_rt = rational_time(offset, frame_rate)
            clip_offset_rt  = "0s"
            effective_dur   = max(0.0, cam_dur - offset)
        else:
            source_start_rt = "0s"
            clip_offset_rt  = rational_time(-offset, frame_rate)
            effective_dur   = cam_dur

        effective_dur_rt = rational_time(effective_dur, frame_rate)

        # clip + video/audio 구조 사용
        # 중요: 중첩된 video/audio에는 start를 붙이지 않는다 (parent clip의 start가 in-point를 이미 지정함)
        angles_xml += f"""
            <mc-angle name="cam{i+1}" angleID="angle{i+1}">
                <clip name="{cam_path.stem}" offset="{clip_offset_rt}" duration="{effective_dur_rt}" start="{source_start_rt}">
                    <video ref="{uid}" offset="0s" duration="{effective_dur_rt}"/>
                    <audio ref="{uid}" offset="0s" duration="{effective_dur_rt}" role="dialogue"/>
                </clip>
            </mc-angle>"""

    # ── mc-clip 컷 배치 ───────────────────────
    spine_clips = ""
    for idx, cut in enumerate(cuts):
        start = cut["start"]
        end = cut["end"]
        cam_idx = cut["cam"]
        cut_dur = end - start
        if cut_dur < 0.1:
            continue

        offset_rt = rational_time(start, frame_rate)
        dur_rt = rational_time(cut_dur, frame_rate)

        # ── 전환 효과 ──────────────────────────────
        # FCPXML spine의 transition은 ref + 인접 클립 handle이 필요하여
        # DTD 오류 유발. Cut 방식으로 처리하고 FCP에서 후처리 권장.

        # DTD 규칙: mc-clip에 mcAngle 속성 없음
        # 활성 앵글은 mc-source 자식 요소로 지정
        angle_id = f"angle{cam_idx + 1}"
        # offset = 타임라인 배치 위치, start = 멀티캠 소스에서 읽을 시작점
        # 두 값이 같아야 연속 재생이 됨
        spine_clips += f"""
                <mc-clip ref="multicam1" name="cam{cam_idx+1}_cut{idx}" offset="{offset_rt}" duration="{dur_rt}" start="{offset_rt}">
                    <mc-source angleID="{angle_id}" srcEnable="all"/>
                </mc-clip>"""

    # Basic Title effect UID (FCP 내장 제너레이터)
    title_effect_id = "r_title"
    title_effect_uid = ".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti"

    safe_title = song_title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") or "Untitled"

    # ── 자막 타이틀 (ASR segments) ────────────
    fcp_color = hex_to_fcpxml_color(subtitle_color)
    bold_val = "1" if subtitle_bold else "0"
    pos_y = {"bottom": "-0.38", "center": "0", "top": "0.38"}.get(subtitle_position, "-0.38")

    titles_xml = ""
    for seg in segments[:50]:  # 최대 50개
        seg_start = float(seg.get("start", 0))
        seg_end = float(seg.get("end", seg_start + 2.0))
        text = seg.get("text", "").strip()
        if not text:
            continue
        seg_dur = max(0.5, seg_end - seg_start)
        seg_offset_rt = rational_time(seg_start, frame_rate)
        seg_dur_rt = rational_time(seg_dur, frame_rate)
        safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        titles_xml += f"""
                <title name="자막" ref="{title_effect_id}" lane="1" offset="{seg_offset_rt}" duration="{seg_dur_rt}" start="0s">
                    <adjust-transform position="0 {pos_y}"/>
                    <param name="Text" key="9999/10003/10003/2/352" value="{safe_text}"/>
                    <text>
                        <text-style font="{subtitle_font}" fontSize="{subtitle_size}" fontColor="{fcp_color}" bold="{bold_val}" alignment="{subtitle_alignment}">{safe_text}</text-style>
                    </text>
                </title>"""

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.11">
    <resources>
        <format id="r1" name="FFVideoFormat1080p2997" frameDuration="{tb}" width="1920" height="1080"/>
        <effect id="{title_effect_id}" name="Basic Title" uid="{title_effect_uid}"/>
{assets_xml}
        <media id="multicam1" name="{safe_title} 멀티캠">
            <multicam format="r1" tcStart="0s">
{angles_xml}
            </multicam>
        </media>
    </resources>
    <library>
        <event name="{safe_title}">
            <project name="{safe_title} Draft">
                <sequence format="r1" duration="{total_dur_rt}" tcStart="0s">
                    <spine>
{spine_clips}
{titles_xml}
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>"""

    output_path.write_text(xml, encoding="utf-8")


# ══════════════════════════════════════════════
# CapCut 드래프트 빌더
# ══════════════════════════════════════════════

def build_capcut_draft(
    cam_paths: list[Path],
    offsets: list[float],
    master_path: Path,
    cuts: list[dict],
    segments: list[dict],
    duration: float,
    song_title: str,
    work_dir: Path,
) -> tuple[Path, Path]:
    """
    CapCut draft_content.json 생성.
    반환: (project_folder, draft_json_path)

    time 단위: microseconds (1s = 1_000_000µs)
    """
    import uuid as _uuid

    def uid() -> str:
        return str(_uuid.uuid4()).upper().replace("-", "")[:32]

    def us(sec: float) -> int:
        return max(0, int(sec * 1_000_000))

    # ── 재료(materials) 준비 ──────────────────────
    cam_mat_ids = []
    video_materials = []
    for i, (cp, offset) in enumerate(zip(cam_paths, offsets)):
        cam_dur = get_video_duration(cp)
        mat_id = uid()
        cam_mat_ids.append(mat_id)
        video_materials.append({
            "audio_fade": None,
            "audio_track_indexes": [],
            "cartoon_path": "",
            "category_id": "",
            "category_name": "",
            "check_flag": 63,
            "crop": {
                "lower_left_x": 0.0, "lower_left_y": 1.0,
                "lower_right_x": 1.0, "lower_right_y": 1.0,
                "upper_left_x": 0.0, "upper_left_y": 0.0,
                "upper_right_x": 1.0, "upper_right_y": 0.0,
            },
            "crop_ratio": "free",
            "crop_scale": 1.0,
            "duration": us(cam_dur),
            "extra_type_option": 0,
            "file_Path": str(cp.resolve()),
            "formula_id": "",
            "freeze": None,
            "has_audio": True,
            "height": 1080,
            "id": mat_id,
            "import_time": int(time.time()),
            "import_time_ms": int(time.time() * 1000),
            "item_source": 1,
            "md5": "",
            "metetype": "natural",
            "roughcut_time": 0,
            "sub_time_range": {"duration": -1, "start": -1},
            "text_alpha": -1,
            "timeline_shape": "",
            "type": 0,
            "video_algorithm": {
                "motion_blur_config": None,
                "noise_reduction": None,
                "path": "",
                "time_range": None,
            },
            "width": 1920,
        })

    # 마스터 오디오 재료
    master_id = uid()
    master_dur = get_video_duration(master_path)
    audio_material = {
        "app_id": 0,
        "category_id": "",
        "category_name": "",
        "check_flag": 1,
        "duration": us(master_dur),
        "effect_id": "",
        "file_Path": str(master_path.resolve()),
        "formula_id": "",
        "id": master_id,
        "import_time": int(time.time()),
        "import_time_ms": int(time.time() * 1000),
        "item_source": 1,
        "md5": "",
        "metetype": "natural",
        "name": master_path.stem,
        "query": "",
        "request_id": "",
        "resource_id": "",
        "search_id": "",
        "source_platform": 0,
        "team_id": "",
        "text": "",
        "tone_folder_name": "",
        "type": "extract_music",
        "wave_points": [],
    }

    # 자막 재료 및 텍스트 세그먼트
    text_materials = []
    text_segments = []
    seg_timeline = 0.0
    for seg in segments:
        seg_start = float(seg.get("start", 0))
        seg_end = float(seg.get("end", seg_start + 2))
        seg_text = seg.get("text", "").strip()
        if not seg_text:
            continue
        txt_id = uid()
        text_materials.append({
            "add_type": 0,
            "alignment": 1,
            "background_alpha": 0.0,
            "background_color": "",
            "background_height": 0.1,
            "background_horizontal_offset": 0.0,
            "background_round_radius": 0.0,
            "background_style": 0,
            "background_vertical_offset": 0.0,
            "background_width": 0.1,
            "base_content": "",
            "bold_width": 0.0,
            "border_alpha": 1.0,
            "border_color": "",
            "border_width": 0.08,
            "caption_template_info": {"category_id": "", "category_name": "", "effect_id": "", "is_new": False, "path": "", "request_id": "", "resource_id": ""},
            "check_flag": 7,
            "combo_info": {"text_templates": []},
            "content": seg_text,
            "fixed_height": -1.0,
            "fixed_width": -1.0,
            "font_category_id": "",
            "font_category_name": "",
            "font_id": "",
            "font_name": "",
            "font_path": "",
            "font_resource_id": "",
            "font_size": 7.0,
            "font_source_platform": 0,
            "font_team_id": "",
            "font_title": "Default",
            "font_url": "",
            "global_alpha": 1.0,
            "group_id": "",
            "has_shadow": False,
            "id": txt_id,
            "initial_scale": 1.0,
            "inner_padding": -1.0,
            "is_rich_text": False,
            "italic": False,
            "italic_degree": 0,
            "kakao_effect_auth_required": False,
            "letter_spacing": 0.0,
            "line_feed": 1,
            "line_max_width": 0.82,
            "line_spacing": 0.02,
            "name": "",
            "original_size": [],
            "preset_id": "",
            "recognize_task_id": "",
            "recognize_type": 0,
            "relevance_segment": [],
            "shadow_alpha": 0.8,
            "shadow_angle": -45.0,
            "shadow_color": "",
            "shadow_distance": 0.08,
            "shadow_point": {"x": 0.6364, "y": -0.6364},
            "shadow_smoothing": 0.9,
            "shape_clip_x": False,
            "shape_clip_y": False,
            "style_name": "",
            "sub_type": 0,
            "text_alpha": 1.0,
            "text_color": "#FFFFFF",
            "text_curve": None,
            "text_preset_resource_id": "",
            "text_size": 30,
            "text_to_audio_ids": [],
            "tts_auto_update": False,
            "type": "text",
            "typesetting": 0,
            "underline": False,
            "underline_offset": 0.22,
            "underline_width": 0.05,
            "use_effect_default_color": True,
            "words": {"end_time": [], "start_time": [], "text": []},
        })
        seg_id = uid()
        text_segments.append({
            "cartoon": False,
            "clip": {
                "alpha": 1.0,
                "flip": {"horizontal": False, "vertical": False},
                "rotation": 0.0,
                "scale": {"x": 1.0, "y": 1.0},
                "transform": {"x": 0.0, "y": -0.4},
            },
            "common_keyframes": [],
            "enable_adjust": True,
            "enable_color_curves": True,
            "enable_color_wheels": True,
            "enable_lut": True,
            "enable_smart_color_adjust": False,
            "extra_material_refs": [],
            "group_id": "",
            "hdr_settings": {"intensity": 1.0, "mode": 1, "nits": 1000},
            "id": seg_id,
            "is_placeholder": False,
            "keyframe_refs": [],
            "last_nonzero_volume": 1.0,
            "material_id": txt_id,
            "render_index": 11000,
            "reverse": False,
            "source_timerange": {"duration": us(seg_end - seg_start), "start": 0},
            "target_timerange": {
                "duration": us(seg_end - seg_start),
                "start": us(seg_start),
            },
            "template_id": "",
            "template_scene": "default",
            "track_attribute": 0,
            "track_render_index": 0,
            "uniform_scale": {"on": True, "value": 1.0},
            "visible": True,
            "volume": 1.0,
        })

    # ── 비디오 트랙 세그먼트 (컷 배치) ──────────────
    video_segments = []
    for idx, cut in enumerate(cuts):
        seq_start = float(cut["start"])
        seq_end = float(cut["end"])
        cut_dur = seq_end - seq_start
        if cut_dur < 0.05:
            continue
        cam_idx = cut["cam"]

        # 소스 in-point: 시퀀스 시간 + 싱크 오프셋
        # offset > 0 → cam이 master보다 먼저 시작 → source = seq_time + offset
        source_in = seq_start + offsets[cam_idx]
        cam_dur = get_video_duration(cam_paths[cam_idx])
        # 범위 초과 방지
        source_in = max(0.0, min(source_in, cam_dur - 0.1))
        source_dur = min(cut_dur, cam_dur - source_in)
        if source_dur < 0.05:
            continue

        seg_id = uid()
        video_segments.append({
            "cartoon": False,
            "clip": {
                "alpha": 1.0,
                "flip": {"horizontal": False, "vertical": False},
                "rotation": 0.0,
                "scale": {"x": 1.0, "y": 1.0},
                "transform": {"x": 0.0, "y": 0.0},
            },
            "common_keyframes": [],
            "enable_adjust": True,
            "enable_color_curves": True,
            "enable_color_wheels": True,
            "enable_lut": True,
            "enable_smart_color_adjust": False,
            "extra_material_refs": [],
            "group_id": "",
            "hdr_settings": {"intensity": 1.0, "mode": 1, "nits": 1000},
            "id": seg_id,
            "intensifies_audio": False,
            "is_placeholder": False,
            "is_tone_modify": False,
            "keyframe_refs": [],
            "last_nonzero_volume": 1.0,
            "material_id": cam_mat_ids[cam_idx],
            "render_index": idx,
            "reverse": False,
            "source_timerange": {
                "duration": us(source_dur),
                "start": us(source_in),
            },
            "target_timerange": {
                "duration": us(source_dur),
                "start": us(seq_start),
            },
            "template_id": "",
            "template_scene": "default",
            "track_attribute": 0,
            "track_render_index": 0,
            "uniform_scale": {"on": True, "value": 1.0},
            "visible": True,
            "volume": 0.0,  # 카메라 오디오 묵음, 마스터 오디오 사용
        })

    # 마스터 오디오 세그먼트
    audio_seg_id = uid()
    audio_segment = {
        "cartoon": False,
        "clip": {"alpha": 1.0, "flip": {"horizontal": False, "vertical": False}, "rotation": 0.0, "scale": {"x": 1.0, "y": 1.0}, "transform": {"x": 0.0, "y": 0.0}},
        "common_keyframes": [],
        "enable_adjust": False,
        "enable_color_curves": False,
        "enable_color_wheels": False,
        "enable_lut": False,
        "enable_smart_color_adjust": False,
        "extra_material_refs": [],
        "group_id": "",
        "hdr_settings": {"intensity": 1.0, "mode": 1, "nits": 1000},
        "id": audio_seg_id,
        "intensifies_audio": False,
        "is_placeholder": False,
        "keyframe_refs": [],
        "last_nonzero_volume": 1.0,
        "material_id": master_id,
        "render_index": 0,
        "reverse": False,
        "source_timerange": {"duration": us(master_dur), "start": 0},
        "target_timerange": {"duration": us(master_dur), "start": 0},
        "template_id": "",
        "template_scene": "default",
        "track_attribute": 0,
        "track_render_index": 0,
        "uniform_scale": {"on": True, "value": 1.0},
        "visible": True,
        "volume": 1.0,
    }

    # ── 트랙 구성 ─────────────────────────────────
    tracks = [
        {
            "attribute": 0,
            "flag": 0,
            "id": uid(),
            "is_default_name": True,
            "name": "",
            "segments": video_segments,
            "type": "video",
        },
        {
            "attribute": 0,
            "flag": 0,
            "id": uid(),
            "is_default_name": True,
            "name": "",
            "segments": [audio_segment],
            "type": "audio",
        },
    ]
    if text_segments:
        tracks.append({
            "attribute": 0,
            "flag": 0,
            "id": uid(),
            "is_default_name": True,
            "name": "",
            "segments": text_segments,
            "type": "text",
        })

    # ── 최종 draft_content.json ───────────────────
    empty_lists = {k: [] for k in [
        "beats", "canvases", "chromas", "color_curves", "digital_humans",
        "drafts", "effects", "flowers", "green_screens", "handwrites",
        "log_color_wheels", "loudnesses", "manual_deformations", "placeholders",
        "plugin_effects", "primary_color_wheels", "realtime_denoises",
        "smart_crops", "smart_relights", "sound_channel_mappings", "speeds",
        "stickers", "tail_leaders", "text_templates", "transitions",
        "video_effects", "video_trackings", "vocal_beautifys", "vocal_separations",
    ]}

    draft = {
        "id": uid(),
        "name": song_title,
        "duration": us(duration),
        "fps": 30.0,
        "canvas_config": {"height": 1080, "ratio": "original", "width": 1920},
        "color_space": 0,
        "cover": "",
        "free_render_index_mode_on": False,
        "group_container": None,
        "keyframe_graph_list": [],
        "keyframes": {
            "adjusts": [], "audios": [], "effects": [], "filters": [],
            "handwrites": [], "stickers": [], "texts": [], "videos": [],
        },
        "lyrics_effects_enabled": True,
        "materials": {
            **empty_lists,
            "audios": [audio_material],
            "texts": text_materials,
            "videos": video_materials,
        },
        "mutable_config": None,
        "new_version": "119.0.0",
        "platform": "mac",
        "relationships": [],
        "render_index_track_mode_on": False,
        "retouch_cover": "",
        "source": "default",
        "static_cover_image_path": "",
        "time_marks": None,
        "tracks": tracks,
        "update_time": int(time.time()),
        "version": 360000,
    }

    # ── 프로젝트 폴더에 저장 ──────────────────────
    # CapCut 프로젝트 폴더 구조: {project_name}/draft_content.json
    safe_name = "".join(c for c in song_title if c.isalnum() or c in " _-()&")[:40].strip() or "capcut_project"
    project_folder = work_dir / safe_name
    project_folder.mkdir(exist_ok=True)
    draft_path = project_folder / "draft_content.json"
    draft_path.write_text(json.dumps(draft, ensure_ascii=False, indent=2))

    return project_folder, draft_path


# ══════════════════════════════════════════════
# 썸네일 처리
# ══════════════════════════════════════════════

def extract_candidate_frames(
    cam_path: Path,
    climax_start: float,
    climax_end: float,
    out_dir: Path,
    max_frames: int = 30,
) -> list[Path]:
    """클라이맥스 구간에서 1fps 프레임 추출"""
    duration = min(climax_end - climax_start, max_frames)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(climax_start),
        "-i", str(cam_path),
        "-t", str(duration),
        "-vf", "fps=1,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
        str(out_dir / "frame_%04d.jpg")
    ]
    subprocess.run(cmd, capture_output=True)
    return sorted(out_dir.glob("frame_*.jpg"))


async def score_frames_parallel(frame_paths: list[Path]) -> list[dict]:
    """CLIP 로컬 모델로 프레임 채점 (M칩 MPS 가속, 완전 무료)
    Anthropic API 키가 있으면 Vision API로 fallback.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "") or load_config().get("anthropic_api_key", "")
    if api_key:
        return await _score_frames_vision_api(frame_paths, api_key)
    return await asyncio.get_event_loop().run_in_executor(
        None, _score_frames_clip, frame_paths
    )


def _score_frames_clip(frame_paths: list[Path]) -> list[dict]:
    """CLIP ViT-B/32 로컬 추론 — MPS 가속, 모델 캐시 재사용"""
    import torch
    import open_clip
    from PIL import Image

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # 모델 캐시 (프로세스 수명 동안 재사용)
    global _clip_model_cache
    if "_clip_model_cache" not in globals():
        _clip_model_cache = {}

    cache_key = f"ViT-B-32_openai_{device}"
    if cache_key not in _clip_model_cache:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _clip_model_cache[cache_key] = (model, preprocess, tokenizer)

    model, preprocess, tokenizer = _clip_model_cache[cache_key]

    # 채점 기준 텍스트 프롬프트
    text_prompts = [
        "an exciting music performance youtube thumbnail",
        "a blurry dark boring concert photo",           # 네거티브
    ]
    with torch.no_grad():
        text_tokens = tokenizer(text_prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    results = []
    for frame in frame_paths[:30]:
        try:
            img = preprocess(Image.open(frame).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                img_features = model.encode_image(img)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                sims = (img_features @ text_features.T).squeeze(0).cpu().tolist()

            # 포지티브 유사도 - 네거티브 유사도 → 0~50 스케일
            score = (sims[0] - sims[1] + 0.5) * 50
            score = max(0.0, min(50.0, score))

            # 추가 OpenCV 보조 점수 (선명도 + 밝기)
            import cv2
            cv_img = cv2.imread(str(frame))
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            mean_brightness = gray.mean()

            sharp_score = min(10.0, sharpness / 500 * 10)
            bright_score = 10.0 - abs(mean_brightness - 128) / 128 * 10

            total = round(score + sharp_score + bright_score, 2)
            results.append({
                "path": str(frame),
                "total": total,
                "scores": {
                    "clip_appeal": round(score, 2),
                    "sharpness": round(sharp_score, 2),
                    "brightness": round(bright_score, 2),
                },
            })
        except Exception as e:
            results.append({"path": str(frame), "total": 0.0, "scores": {}})

    return results


async def _score_frames_vision_api(frame_paths: list[Path], api_key: str) -> list[dict]:
    """Anthropic Vision API 채점 (API 키 있을 때 사용)"""
    import base64

    system_prompt = (
        "당신은 유튜브 썸네일 전문가입니다. "
        "이 이미지를 음악 공연 유튜브 썸네일로 사용할 때의 매력도를 JSON으로 채점하세요. "
        '{"composition": 0-10, "brightness": 0-10, "emotion": 0-10, "clarity": 0-10, "thumbnail_appeal": 0-10} '
        "설명 없이 JSON만 출력."
    )

    async def score_one(frame: Path) -> dict:
        try:
            img_data = base64.b64encode(frame.read_bytes()).decode()
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-haiku-4-5-20251001",
                        "max_tokens": 100,
                        "system": system_prompt,
                        "messages": [{
                            "role": "user",
                            "content": [{
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}
                            }, {"type": "text", "text": "채점해주세요."}]
                        }]
                    }
                )
                text = resp.json()["content"][0]["text"]
                scores = json.loads(text)
                total = sum(scores.values())
                return {"path": str(frame), "total": total, "scores": scores}
        except Exception:
            return {"path": str(frame), "total": 30, "scores": {}}

    tasks = [score_one(f) for f in frame_paths[:20]]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]


# ══════════════════════════════════════════════
# 썸네일 합성 (Pillow)
# ══════════════════════════════════════════════

def get_font(name: str, size: int):
    """폰트 로드 (시스템 → Google Fonts 자동 다운로드)"""
    from PIL import ImageFont

    # 시스템 폰트 경로 목록
    system_font_dirs = [
        Path("/Library/Fonts"),
        Path("~/Library/Fonts").expanduser(),
        FONTS_DIR,
    ]

    font_map = {
        "NanumSquareExtraBold": ["NanumSquareEB.ttf", "NanumSquareExtraBold.ttf"],
        "NanumGothic": ["NanumGothic.ttf", "NanumGothicBold.ttf"],
        "BebasNeue": ["BebasNeue-Regular.ttf", "BebasNeue.ttf"],
        "Oswald": ["Oswald-Bold.ttf", "Oswald-Regular.ttf"],
    }

    candidates = font_map.get(name, [f"{name}.ttf"])

    for font_dir in system_font_dirs:
        for cand in candidates:
            fp = font_dir / cand
            if fp.exists():
                try:
                    return ImageFont.truetype(str(fp), size)
                except Exception:
                    pass

    # Google Fonts 다운로드 시도
    gf_urls = {
        "NanumSquareExtraBold": "https://fonts.gstatic.com/s/nanumsquare/v14/nanumsquareeb.woff2",
        "NanumGothic": "https://fonts.gstatic.com/s/nanumgothic/v23/PN_3Rfi-oW3hYwmKDpxS7F_LQv37zlEn14YEUQ.woff2",
        "BebasNeue": "https://fonts.gstatic.com/s/bebasneue/v14/JTUSjIg69CK48gW7PXoo9WdhyyTh89ZNpQ.woff2",
        "Oswald": "https://fonts.gstatic.com/s/oswald/v53/TK3_WkUHHAIjg75cFRf3bXL8LICs13NvgUFoZAaRliE.woff2",
    }

    # TTF fallback URLs
    gf_ttf = {
        "BebasNeue": "https://github.com/dharmatype/Bebas-Neue/raw/master/fonts/BebasNeue(2018)ByDhamraType/TTF/BebasNeue-Regular.ttf",
        "NanumGothic": "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf",
        "NanumSquareExtraBold": "https://github.com/google/fonts/raw/main/ofl/nanumsquare/NanumSquareEB.ttf",
    }

    if name in gf_ttf:
        dest = FONTS_DIR / f"{name}.ttf"
        if not dest.exists():
            try:
                import urllib.request
                urllib.request.urlretrieve(gf_ttf[name], str(dest))
            except Exception:
                pass
        if dest.exists():
            try:
                return ImageFont.truetype(str(dest), size)
            except Exception:
                pass

    return ImageFont.load_default()


def hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def hex_to_fcpxml_color(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16)/255, int(h[2:4], 16)/255, int(h[4:6], 16)/255
    return f"{r:.4f} {g:.4f} {b:.4f} 1"


def _srt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_srt(segments: list[dict]) -> str:
    lines = []
    idx = 1
    for seg in segments:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", start + 2))
        text = seg.get("text", "").strip()
        if not text:
            continue
        lines += [str(idx), f"{_srt_time(start)} --> {_srt_time(end)}", text, ""]
        idx += 1
    return "\n".join(lines)


def _vtt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def generate_vtt(segments: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", start + 2))
        text = seg.get("text", "").strip()
        if not text:
            continue
        lines += [f"{_vtt_time(start)} --> {_vtt_time(end)}", text, ""]
    return "\n".join(lines)


def download_url_audio(url: str, out_dir: Path) -> Path:
    out_tmpl = str(out_dir / "downloaded.%(ext)s")
    result = subprocess.run(
        ["yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0",
         "--no-playlist", "-o", out_tmpl, url],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        result2 = subprocess.run(
            ["yt-dlp", "-x", "--no-playlist", "-o", out_tmpl, url],
            capture_output=True, text=True, timeout=600
        )
        if result2.returncode != 0:
            raise RuntimeError(f"yt-dlp 오류: {result.stderr[:500]}")
    for f in sorted(out_dir.glob("downloaded.*")):
        return f
    raise RuntimeError("다운로드된 파일을 찾을 수 없음")


def compose_thumbnail(
    frame_path: str | None,
    params: dict,
    layout: str,
    output_path: Path,
    width: int = 1280,
    height: int = 720,
):
    """단일 레이아웃 썸네일 합성"""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

    song_title = params.get("song_title", "")
    performer = params.get("performer", "")
    date_venue = params.get("date_venue", "")
    channel_tag = params.get("channel_tag", "")
    logo_path_str = params.get("logo_path", "")
    font_title_name = params.get("font_title", "NanumSquareExtraBold")
    font_sub_name = params.get("font_sub", "NanumGothic")
    text_color = params.get("text_color", "#FFFFFF")
    accent_color = params.get("accent_color", "#22c55e")
    overlay_alpha = float(params.get("overlay_alpha", 0.45))

    tc = hex_to_rgb(text_color)
    ac = hex_to_rgb(accent_color)

    # 배경 이미지
    if frame_path and Path(frame_path).exists():
        bg = Image.open(frame_path).convert("RGB").resize((width, height), Image.LANCZOS)
    else:
        bg = Image.new("RGB", (width, height), (20, 20, 20))

    draw = ImageDraw.Draw(bg)

    # 로고 로드
    logo_img = None
    if logo_path_str:
        logo_full = BASE_DIR / logo_path_str
        if logo_full.exists():
            try:
                logo_img = Image.open(logo_full).convert("RGBA")
                logo_img.thumbnail((120, 60), Image.LANCZOS)
            except Exception:
                logo_img = None

    font_big = get_font(font_title_name, int(72 * width / 1280))
    font_med = get_font(font_sub_name, int(40 * width / 1280))
    font_sm = get_font(font_sub_name, int(28 * width / 1280))

    if layout == "A":
        # 좌측 사진 60% + 우측 텍스트 40%
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        rect = Image.new("RGBA", (int(width * 0.42), height), (10, 10, 10, 230))
        overlay.paste(rect, (int(width * 0.58), 0))
        bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(bg)

        tx = int(width * 0.60)
        ty = height // 4
        draw.text((tx, ty), song_title, font=font_big, fill=tc)
        if performer:
            draw.text((tx, ty + int(90 * height / 720)), performer, font=font_med, fill=ac)
        if channel_tag:
            draw.text((tx, ty + int(145 * height / 720)), channel_tag, font=font_sm, fill=(180, 180, 180))
        if logo_img:
            bg.paste(logo_img, (width - logo_img.width - 30, height - logo_img.height - 30), logo_img)

    elif layout == "B":
        # 풀스크린 + 하단 텍스트 바 ← 기본
        bar_h = int(height * 0.28)
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        # 그라데이션 바 (단색 반투명)
        bar = Image.new("RGBA", (width, bar_h), (0, 0, 0, int(255 * (overlay_alpha + 0.2))))
        overlay.paste(bar, (0, height - bar_h))
        # 상단 어두운 오버레이
        top_bar = Image.new("RGBA", (width, int(height * 0.08)), (0, 0, 0, int(255 * overlay_alpha * 0.5)))
        overlay.paste(top_bar, (0, 0))
        bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(bg)

        # 악센트 라인
        draw.rectangle([(0, height - bar_h), (width, height - bar_h + 4)], fill=ac)

        ty = height - bar_h + 12
        draw.text((40, ty), song_title, font=font_big, fill=tc)
        sub_y = ty + int(78 * height / 720)
        if performer:
            draw.text((40, sub_y), performer, font=font_med, fill=(220, 220, 220))
        if date_venue:
            draw.text((40, sub_y + int(48 * height / 720)), date_venue, font=font_sm, fill=(160, 160, 160))
        if channel_tag:
            bbox = draw.textbbox((0, 0), channel_tag, font=font_sm)
            tw = bbox[2] - bbox[0]
            draw.text((width - tw - 40, sub_y), channel_tag, font=font_sm, fill=ac)
        if logo_img:
            bg.paste(logo_img, (width - logo_img.width - 40, height - bar_h + 15), logo_img)

    elif layout == "C":
        # 텍스트 중심 + 반투명 배경
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, int(255 * overlay_alpha)))
        bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(bg)

        bbox = draw.textbbox((0, 0), song_title, font=font_big)
        tw = bbox[2] - bbox[0]
        tx = (width - tw) // 2
        ty = height // 2 - int(60 * height / 720)
        draw.text((tx, ty), song_title, font=font_big, fill=tc)
        if performer:
            bbox2 = draw.textbbox((0, 0), performer, font=font_med)
            pw = bbox2[2] - bbox2[0]
            draw.text(((width - pw) // 2, ty + int(90 * height / 720)), performer, font=font_med, fill=ac)
        if logo_img:
            bg.paste(logo_img, (width - logo_img.width - 30, height - logo_img.height - 30), logo_img)

    elif layout == "D":
        # 듀얼 프레임: 좌 와이드 | 우상 클로즈업 | 우하 텍스트
        if frame_path and Path(frame_path).exists():
            left = Image.open(frame_path).convert("RGB").resize((int(width * 0.55), height), Image.LANCZOS)
            bg.paste(left, (0, 0))
        # 우측 절반 배경 어둡게
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        right_bg = Image.new("RGBA", (int(width * 0.45), height), (15, 15, 15, 255))
        overlay.paste(right_bg, (int(width * 0.55), 0))
        bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(bg)

        # 구분선
        draw.rectangle([(int(width * 0.55), 0), (int(width * 0.55) + 4, height)], fill=ac)

        tx = int(width * 0.57)
        ty = height // 3
        draw.text((tx, ty), song_title, font=font_big, fill=tc)
        if performer:
            draw.text((tx, ty + int(90 * height / 720)), performer, font=font_med, fill=ac)
        if channel_tag:
            draw.text((tx, ty + int(145 * height / 720)), channel_tag, font=font_sm, fill=(180, 180, 180))
        if logo_img:
            bg.paste(logo_img, (width - logo_img.width - 20, height - logo_img.height - 20), logo_img)

    bg.save(str(output_path), "PNG", optimize=True)


def compose_all_thumbnails(
    frame_path: str | None,
    params: dict,
    out_dir: Path,
) -> list[str]:
    """모든 레이아웃 + 밝기/채도 변형 합성"""
    from PIL import Image, ImageEnhance

    results = []
    primary_layout = params.get("layout", "B")

    # 기본 레이아웃 1280x720
    main_path = out_dir / f"thumbnail_{primary_layout}.png"
    compose_thumbnail(frame_path, params, primary_layout, main_path)
    results.append(str(main_path))

    # 고해상도 2560x1440
    hq_path = out_dir / f"thumbnail_{primary_layout}_hq.png"
    compose_thumbnail(frame_path, params, primary_layout, hq_path, 2560, 1440)
    results.append(str(hq_path))

    # 나머지 레이아웃들
    for layout in ["A", "B", "C", "D"]:
        if layout == primary_layout:
            continue
        p = out_dir / f"thumbnail_{layout}.png"
        compose_thumbnail(frame_path, params, layout, p)
        results.append(str(p))

    # 밝기/채도 변형 (기본 레이아웃 기준)
    if main_path.exists():
        base_img = Image.open(main_path)
        for suffix, enhancer_cls, factor in [
            ("bright", ImageEnhance.Brightness, 1.1),
            ("dark", ImageEnhance.Brightness, 0.9),
            ("vivid", ImageEnhance.Color, 1.2),
        ]:
            var_path = out_dir / f"thumbnail_{primary_layout}_{suffix}.png"
            enhanced = enhancer_cls(base_img).enhance(factor)
            enhanced.save(str(var_path), "PNG")
            results.append(str(var_path))

    return results
