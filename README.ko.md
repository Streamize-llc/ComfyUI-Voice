<div align="center">

# ComfyUI-Voice 🎙️

**ComfyUI를 위한 통합되고 확장 가능한 오디오 & 음성 제품군.**
TTS · 음성 클로닝 · 음성-텍스트 변환 — 다양한 최신 오픈 모델을 하나의 깔끔한 노드 그래프로.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-8A2BE2.svg)](https://github.com/comfyanonymous/ComfyUI)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-contributing)

<br/>

[English](README.md) · **한국어** · [中文](README.zh.md) · [日本語](README.ja.md)

</div>

---

ComfyUI는 세계 최고 수준의 이미지 및 비디오 도구를 갖추고 있지만, 오디오는
저마다 충돌하는 의존성을 끌어들이고 종종 설치를 망가뜨리는 일회성 노드들의
누더기에 머물러 있었습니다. **ComfyUI-Voice는 바로 그 공백을 메웁니다**: 모든
음성 모델이 하나의 노드 집합 뒤에서 곧바로 꽂아 쓸 수 있는 어댑터로 동작하는
일관된 단일 제품군이며, **모든 것이 이미 사용 중인 PyTorch 위에서 네이티브로
실행됩니다** — 별도 환경도, 의존성 지옥도 없습니다.

```
text ─▶ [Voice TTS] ─▶ AUDIO ─▶ [Save Audio]            # synthesize
audio ─▶ [Voice ASR] ─▶ TRANSCRIPT + text               # transcribe
ref ──▶ [Voice TTS] ─▶ AUDIO                             # zero‑shot voice clone
```

## ✨ 주요 특징

- 🧩 **작업당 하나의 노드, 여러 엔진.** 드롭다운에서 엔진을 고르면 — 해당 엔진이
  할 수 있다고 선언한 기능에 맞게 노드의 입력이 적응합니다.
- 🛡️ ***당신의* torch 위에서 실행.** 엔진은 ComfyUI의 기존 PyTorch/Transformers
  위에서 네이티브로 로드됩니다. 설치와 충돌하는 버전 고정도, 지원 모델군을 위한
  별도 venv도 없습니다. (충돌하는 모델은 격리를 선택할 수 있지만 — 검증된 모델
  중에는 그럴 필요가 있는 것이 하나도 없습니다.)
- 🌍 **다국어 & SOTA.** 음성 합성, 제로샷 음성 클로닝, 수십 개 언어에 걸친 전사를
  위한 선도적 오픈 모델들을 래핑합니다 — 여기에 텍스트-음악 변환과 텍스트-SFX
  생성까지 더했습니다.
- 🔌 **한 파일로 모델 추가.** 새 엔진은 스스로 등록하는 단일 어댑터입니다 — 중앙
  디스패치도, 손댈 UI 배선도 없습니다.
- 🧵 **조합 가능.** 모든 것이 ComfyUI의 네이티브 `AUDIO` 타입으로 통신하므로,
  내장 Load/Save/Preview Audio 노드와 추가 작업 없이 체인으로 연결됩니다.
- ✅ **정직한 상태 표기.** 모델은 이 코드베이스에서 실제 왕복 테스트를 거친
  후에만 *검증됨*으로 표시됩니다.

## 🎧 지원 엔진

> **검증됨 (✅)** = 호스트 스택에서 실제 추론이 종단 간(합성 → 전사 왕복)으로
> 테스트됨. **실험적 (🧪)** = 어댑터는 포함되었으나 의존성이 아직
> 설치/검증되지 않음 — *Voice Engine Info* 노드의 설치 힌트를 통해 활성화하세요.

### Text‑to‑Speech

| 엔진 (`id`) | 기능 | 언어 | 라이선스 | 상태 |
|---|---|---|---|---|
| **MeloTTS** (`melotts_korean`) | 빠른 프리셋 TTS | Korean¹ | Apache‑2.0 / MIT | ✅ |
| **CosyVoice 3.0** (`cosyvoice3`) | **제로샷 음성 클로닝** + VC | zh · en · ja · ko · de · es · fr · it · ru | Apache‑2.0 | ✅ |
| **Supertonic 3** (`supertonic`) | 온디바이스 프리셋 TTS (ONNX) | 30+ incl. ko | code MIT / model OpenRAIL‑M | ✅ |
| **Higgs Audio v3 (4B)** (`higgs_audio_v3`) | 표현력 TTS + 제로샷 클로닝 | 100+ (ko·zh·ja 포함) | **연구/비상업** ⚠️ | ✅ (eval) |
| **Chatterbox** (`chatterbox`) | 클로닝 + 감정 제어 | 23 langs | MIT | 🧪 |
| **Qwen3‑TTS** (`qwen3_tts`) | 클로닝 + 음성 디자인 | 10 langs | Apache‑2.0 | 🧪 |
| **OuteTTS 1.0** (`oute_tts`) | 컴팩트 LLM‑TTS | 14 langs | Apache‑2.0 | 🧪 |

### Speech‑to‑Text

| 엔진 (`id`) | 기능 | 언어 | 라이선스 | 상태 |
|---|---|---|---|---|
| **faster‑whisper** (`faster_whisper`) | 빠른 ASR + 단어 타임스탬프 | 99 langs | MIT | ✅ |
| **Korean Whisper** (`korean_whisper`) | ASR (ko 파인튜닝) | Korean | Apache‑2.0 | 🧪 |
| **Qwen3‑ASR** (`qwen3_asr`) | ASR + 강제 정렬기 타임스탬프 | 30+ langs | Apache‑2.0 | 🧪 |
| **SenseVoice** (`sensevoice`) | 매우 빠른 ASR + 감정/이벤트 태그 | 5+ langs | custom ⚠️ | 🧪 |
| **WhisperX** (`whisperx`) | ASR + 단어 정렬 + 화자 분리 | whisper langs | BSD‑2 (+pyannote gated) | 🧪 |

<sub>¹ 포함된 체크포인트는 한국어용입니다. MeloTTS 계열은 다국어를 지원하며 — 추가 언어 어댑터는 손쉽게 꽂아 쓸 수 있습니다.</sub>

### 생성형 오디오 (음악 · SFX)

| 엔진 (`id`) | 기능 | 언어 | 라이선스 | 상태 |
|---|---|---|---|---|
| **ACE‑Step 1.5** (`ace_step`) | 텍스트-음악 변환 (연주곡 / 노래 + 가사) | 50+ incl. ko·zh·ja·en | Apache‑2.0 / MIT | ✅ |
| **MOSS‑SoundEffect v2.0** (`moss_soundeffect`) | 텍스트-효과음 변환 / 폴리 | en prompts | Apache‑2.0 | ✅ |

<sub>² 둘 다 호스트 torch 위에서 실행되며 48 kHz `AUDIO`를 출력합니다. ACE‑Step은 ComfyUI 코어의 네이티브 지원을 재사용하고, MOSS‑SoundEffect의 추론 코드는 벤더링되어 있습니다(`descript‑audiotools` 없음, 추가 torch 없음).</sub>

또한 의존성이 없는 네 개의 **레퍼런스 엔진**(`reference_tone`, `reference_asr`,
`reference_music`, `reference_sfx`)이 포함되어 있어, 깨끗한 설치 상태에서 전체
파이프라인을 빠르게 점검하고 어댑터 템플릿 역할을 합니다.

## 🚀 빠른 시작

```bash
# 1) Clone into your ComfyUI custom_nodes
cd ComfyUI/custom_nodes
git clone https://github.com/Streamize-llc/ComfyUI-Voice

# 2) Install an engine's deps (example: the workhorses)
pip install faster-whisper          # ASR
pip install librosa g2pkk jamo python-mecab-ko python-mecab-ko-dic num2words anyascii  # MeloTTS frontend
```

ComfyUI를 재시작한 뒤, 그래프에서:

1. **`Voice TTS 🎙️`** 추가 → `engine` 선택 → 텍스트 입력.
2. 해당 노드의 `AUDIO` 출력을 코어 **`Save Audio`**(또는 `Preview Audio`)에 연결.
3. STT의 경우, **`Voice ASR (STT) 🎙️`**를 추가하고 임의의 `AUDIO`를 넣은 뒤 전사 결과를 읽습니다.

무엇이 설치되어 있는지 모르겠다면? **`Voice Engine Info 🎙️`** 노드를 놓아 보세요 —
모든 엔진과 그 기능, 그리고 아직 활성화되지 않은 항목에 대한 `pip install …` 힌트를 나열해 줍니다.

> 일부 무거운 엔진(예: CosyVoice 3.0)은 일회성 모델/코드 설정이 필요합니다.
> 각 어댑터의 도큐먼트 문자열(`comfyui_voice/engines/`)에 정확한 단계가 문서화되어 있습니다.

## 🧩 노드

| 노드 | 카테고리 | 입력 → 출력 |
|---|---|---|
| **Voice TTS** | `audio/voice/tts` | text (+ optional `VOICE_REF`) → `AUDIO` |
| **Voice ASR (STT)** | `audio/asr` | `AUDIO` → `VOICE_TRANSCRIPT` + text |
| **Voice Music Gen** | `audio/generate/music` | text (+ duration/seed) → `AUDIO` |
| **Voice SFX Gen** | `audio/generate/sfx` | text (+ duration/seed) → `AUDIO` |
| **Voice Engine Info** | `audio/voice/util` | — → engine/capability report |

음성 클로닝은 단지 레퍼런스 클립을 (코어 **Load Audio**를 통해) TTS 노드의
`voice_ref` 입력에 배선하는 것일 뿐 — 별도의 업로드 단계가 필요 없습니다.

## 🏗️ 아키텍처

이 설계는 ComfyUI 코어 자체의 철학을 차용합니다 — 모델별로 스택을
pip 설치하는 대신 *모델 코드를 하나의 torch에서 실행되도록 재구현/포팅*하는 것입니다.

- **조합 가능한 타입드 소켓.** 코어 `AUDIO`(`{"waveform": [B,C,T], "sample_rate"}`)에
  더해, 네임스페이스가 지정된 소수의 `VOICE_*` 타입(`VOICE_REF`, `VOICE_TRANSCRIPT`,
  `VOICE_STEMS`, …)이 노드를 상호 운용 가능하고 미래에도 견고하게 만듭니다.
- **기능 기반 설계.** 각 엔진은 하나의 `EngineCapabilities` 데이터클래스(언어,
  클로닝, 샘플 레이트, 격리, 라이선스, 파라미터 스키마)를 선언합니다. 폼,
  검증, 도구 표면이 이로부터 **생성됩니다** — 어디에도 `if engine == …`
  분기가 없습니다.
- **두 가지 어댑터 계층.**
  1. **네이티브 (`inproc`, 권장):** 호스트 torch 위에서 실행됩니다 —
     Transformers(이미 존재하는 아키텍처 라이브러리)를 통하거나, 작은 순수 파이썬
     의존성 + 로드 시점 심(shim)과 함께 업스트림 추론 코드를 벤더링하는
     방식입니다. *검증된 모든 모델이 이 방식을 사용합니다.*
  2. **격리 (`subprocess`):** 고정된 의존성이 실제로 충돌하는 엔진을 위한
     것으로 — 균일한 워커 프로토콜 뒤에서 엔진별 venv에서 실행됩니다. 단일
     `isolation="subprocess"` 필드로 선택합니다.
- **제품군 자체 모델 매니저**가 (ComfyUI가 볼 수 없는) 원시 모델 VRAM을 추적하고
  노드 전반에 걸쳐 모델을 축출하며, 샘플 레이트 가드가 체이닝 시 조용한 손상을 방지합니다.

## ➕ 한 파일로 엔진 추가하기

`comfyui_voice/engines/tts/_reference_tone.py`를 복사하여 내용을 채우세요:

```python
@register_engine("my_tts")
class MyTTS(BaseEngine):
    CAPS = EngineCapabilities(
        id="my_tts", display_name="My TTS",
        tasks=("tts",), license="Apache-2.0", commercial_safe=True,
        supports_cloning=True, languages=("en", "ko"),
        sample_rate=24000, isolation="inproc",   # or "subprocess" if pins conflict
        pip_install=("my-model",), probe_import=("my_model",),
        param_schema={"params": {...}},
    )
    def load(self):
        import my_model                     # lazy-import heavy deps HERE
        self.m = my_model.load(...)
    def generate(self, task, req):
        wav, sr = self.m.tts(req["text"], ...)
        return {"waveform": wav, "sample_rate": sr}
    def unload(self):
        del self.m
```

이것이 계약의 전부입니다 — 드롭다운에 자동으로 나타나고, 폼은 해당 엔진의
`param_schema`에 맞춰 적응하며, 검증은 그 기능을 따릅니다. 수정할 코어 파일이 없습니다.

> **메인 환경에 의존성을 추가하기 전에 `pip install --dry-run`을 실행하여 ComfyUI의
> `torch` / `transformers` / `numpy`를 옮기지 않는지 확인하세요.** 만약 옮긴다면
> `isolation="subprocess"`를 선언하세요.

## 🗺️ 로드맵

- [x] 코어 프레임워크, 타입드 소켓, 기능 레지스트리, 격리 런타임
- [x] TTS + ASR 노드; 검증된 네이티브 TTS(프리셋 & 제로샷 클로닝) 및 ASR
- [ ] 더 많은 검증된 엔진 (Qwen3‑TTS, Chatterbox, …)
- [ ] `VOICE_REF` 클로닝 UX + 음성 라이브러리
- [ ] 음성 변환 (RVC / Seed‑VC)
- [ ] 소스 분리, 노이즈 제거/향상, 강제 정렬 & 자막
- [x] 음악 / SFX 생성 (ACE‑Step 1.5 · MOSS‑SoundEffect v2.0)
- [ ] 오디오 편집 / 인페인팅

## 🤝 기여하기

PR과 새로운 엔진 어댑터를 매우 환영합니다. 좋은 기여는:

1. `comfyui_voice/engines/<task>/` 아래의 **단일 어댑터 파일**입니다.
2. 순수 코어 테스트를 통과합니다: `python tests/test_core.py` (모델 불필요).
3. 호스트 스택을 온전히 유지합니다 (`pip install --dry-run` 깨끗하거나 `subprocess`).
4. 정확한 라이선스 + `commercial_safe` 플래그를 선언합니다.

## 📜 라이선스 & 크레딧

ComfyUI-Voice는 **Apache‑2.0** 라이선스로 배포됩니다. 래핑된 각 모델은 **자체**
라이선스를 유지합니다 — 위 표와 엔진별 어댑터를 참조하세요. 일부는 비상업용이거나
사용이 제한됩니다. 활성화하는 모델의 라이선스를 준수할 책임은 사용자에게 있습니다.

오픈소스 음성 & 오디오 커뮤니티의 어깨 위에서 만들어졌습니다 — MeloTTS, CosyVoice,
Supertonic, MMS, OpenAI Whisper / faster‑whisper, Kokoro, Chatterbox, Qwen, OuteTTS,
SenseVoice, WhisperX, ACE‑Step, MOSS‑SoundEffect (OpenMOSS), 그리고 ComfyUI 자체. 감사합니다. 🙏
