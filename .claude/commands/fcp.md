---
name: fcp
description: >
  FCP MultiCam Agent 실행 및 관리 스킬.
  "fcp 시작", "서버 켜줘", "편집 시작", "fcp 열어줘", "멀티캠 에이전트",
  "썸네일 만들어", "파이프라인 안 돼", "서버 죽었어", "fcp 재시작"
  같은 말이 나오면 반드시 이 스킬을 사용한다.
  서버 시작, 브라우저 오픈, 파이프라인 문제 진단, 의존성 설치까지 모두 처리한다.
---

# FCP MultiCam Agent 스킬

## 프로젝트 경로
```
/Volumes/Samsung T7/Agent/FinalCut AutoEdit
```

## 작업 흐름

### 1단계: 서버 상태 확인

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ 2>/dev/null
```

- `200` → 이미 실행 중. 브라우저만 열면 됨
- 그 외 / 실패 → 서버 시작 필요

### 2단계: 서버 시작 (필요 시)

```bash
pkill -f "uvicorn main:app" 2>/dev/null; sleep 1
cd "/Volumes/Samsung T7/Agent/FinalCut AutoEdit"
uvicorn main:app --port 8000 --host 0.0.0.0 > /tmp/fcp_server.log 2>&1 &
sleep 3
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/
```

`200`이 나오면 성공. 실패 시 → 3단계(진단)로.

### 3단계: 브라우저 열기

```bash
open http://localhost:8000
```

### 4단계: 상태 보고

사용자에게 다음을 알려준다:
- 서버 PID
- 로그 경로: `/tmp/fcp_server.log`
- 접속 URL: `http://localhost:8000`
- 간단한 사용 순서 (파일 드래그 → 노래 제목 → 편집 시작)

---

## 문제 진단 가이드

### "아무 일도 안 일어나" / "대기 중으로만 뜸"

1. 브라우저 콘솔 확인 (F12 → Console 탭)
2. 서버 로그 확인:
   ```bash
   tail -30 /tmp/fcp_server.log
   ```
3. 로그 창(🖥 로그 버튼)이 비어 있다면 → 카메라 파일과 노래 제목 입력 여부 확인

가장 흔한 원인:
- 카메라 파일 미업로드 → 드래그 앤 드롭 필요
- 노래 제목 미입력 → 입력란이 빨갛게 바뀌어야 함
- 서버가 죽어 있음 → 이 스킬로 재시작

### "서버 오류" / 500 에러

```bash
tail -50 /tmp/fcp_server.log | grep -E "ERROR|error|Traceback"
```

오류 내용을 사용자에게 보여주고 원인을 설명한다.

### 의존성 오류 (ModuleNotFoundError 등)

```bash
cd "/Volumes/Samsung T7/Agent/FinalCut AutoEdit"
pip3 install mlx-whisper librosa scipy Pillow fastapi uvicorn python-multipart aiofiles open-clip-torch opencv-python
```

### ffmpeg 오류

```bash
which ffmpeg || brew install ffmpeg
```

---

## 서버 재시작

```bash
pkill -f "uvicorn main:app" 2>/dev/null
sleep 1
cd "/Volumes/Samsung T7/Agent/FinalCut AutoEdit"
uvicorn main:app --port 8000 --host 0.0.0.0 > /tmp/fcp_server.log 2>&1 &
sleep 2
echo "재시작 완료"
open http://localhost:8000
```

---

## 로그 실시간 확인

```bash
tail -f /tmp/fcp_server.log
```

---

## 빠른 사용법 (사용자에게 안내)

```
1. 카메라 영상 파일(mp4/mov)을 드래그해서 올리기
2. 노래 제목 입력
3. ▶ 편집 시작 클릭
4. 🖥 로그 버튼으로 진행 상황 확인
5. 완료 후 "🎬 FCP에서 열기" 또는 썸네일 다운로드
```

채널 설정(⚙)은 처음 한 번만 해두면 이후 자동으로 적용된다.
