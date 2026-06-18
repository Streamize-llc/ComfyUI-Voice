<div align="center">

# ComfyUI-Voice 🎙️

**面向 ComfyUI 的统一、可扩展音频与语音套件。**
TTS · 声音克隆 · 语音转文字 —— 众多前沿开源模型，一套简洁的节点图。

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-8A2BE2.svg)](https://github.com/comfyanonymous/ComfyUI)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-contributing)

<br/>

[English](README.md) · [한국어](README.ko.md) · **中文** · [日本語](README.ja.md)

</div>

---

ComfyUI 拥有世界一流的图像与视频工具链 —— 但音频部分一直是各种零散节点的拼凑，
每个节点都拖入自己相互冲突的依赖，常常把你的安装环境搞坏。
**ComfyUI-Voice 填补了这一空白**：一个统一、协调的套件，
其中每个语音模型都是隐藏在同一组节点之后的即插即用适配器，
而且**一切都原生运行在你已有的 PyTorch 上** —— 无需第二套环境，没有依赖地狱。

```
text ─▶ [Voice TTS] ─▶ AUDIO ─▶ [Save Audio]            # synthesize
audio ─▶ [Voice ASR] ─▶ TRANSCRIPT + text               # transcribe
ref ──▶ [Voice TTS] ─▶ AUDIO                             # zero‑shot voice clone
```

## ✨ 亮点

- 🧩 **每项任务一个节点，背后多种引擎。** 从下拉菜单中选择一个引擎 —— 节点的输入会根据该引擎声明的能力自动适配。
- 🛡️ **运行在*你自己的* torch 上。** 引擎原生加载在 ComfyUI 现有的
  PyTorch/Transformers 之上。没有版本锁定与你的安装环境冲突，受支持的模型集也无需单独的虚拟环境。
  （存在冲突的模型可以选择隔离运行 —— 但所有已验证的模型都不需要。）
- 🌍 **多语言且达到 SOTA 水平。** 封装了领先的开源模型，覆盖数十种语言的语音合成、零样本声音克隆与转录。
- 🔌 **一个文件即可新增一个模型。** 新增引擎只需一个自注册的适配器 —— 无需中心化调度，无需改动任何 UI 管线。
- 🧵 **可组合。** 一切都使用 ComfyUI 原生的 `AUDIO` 类型，因此可以与内置的 Load/Save/Preview Audio 节点免费串联。
- ✅ **状态诚实可信。** 只有在本代码库上完成真实的往返测试后，模型才会被标记为*已验证*。

## 🎧 支持的引擎

> **已验证 (✅)** = 在宿主技术栈上完成端到端真实推理测试（合成 → 转录
> 往返）。**实验性 (🧪)** = 适配器已提供，但依赖尚未安装/验证 ——
> 可通过 *Voice Engine Info* 节点的安装提示来启用。

### 文本转语音（Text‑to‑Speech）

| 引擎 (`id`) | 能力 | 语言 | 许可证 | 状态 |
|---|---|---|---|---|
| **MeloTTS** (`melotts_korean`) | 快速预设 TTS | Korean¹ | Apache‑2.0 / MIT | ✅ |
| **CosyVoice 3.0** (`cosyvoice3`) | **零样本声音克隆** + VC | zh · en · ja · ko · de · es · fr · it · ru | Apache‑2.0 | ✅ |
| **Supertonic 3** (`supertonic`) | 设备端预设 TTS（ONNX） | 30+ 含 ko | code MIT / model OpenRAIL‑M | ✅ |
| **Higgs Audio v3 (4B)** (`higgs_audio_v3`) | 富表现力 TTS + 零样本克隆 | 100+ 含 ko·zh·ja | **研究/非商业** ⚠️ | ✅（评估） |
| **Chatterbox** (`chatterbox`) | 克隆 + 情感控制 | 23 种语言 | MIT | 🧪 |
| **Qwen3‑TTS** (`qwen3_tts`) | 克隆 + 声音设计 | 10 种语言 | Apache‑2.0 | 🧪 |
| **OuteTTS 1.0** (`oute_tts`) | 紧凑型 LLM‑TTS | 14 种语言 | Apache‑2.0 | 🧪 |

### 语音转文本（Speech‑to‑Text）

| 引擎 (`id`) | 能力 | 语言 | 许可证 | 状态 |
|---|---|---|---|---|
| **faster‑whisper** (`faster_whisper`) | 快速 ASR + 词级时间戳 | 99 种语言 | MIT | ✅ |
| **Korean Whisper** (`korean_whisper`) | ASR（ko 微调） | Korean | Apache‑2.0 | 🧪 |
| **Qwen3‑ASR** (`qwen3_asr`) | ASR + 强制对齐时间戳 | 30+ 种语言 | Apache‑2.0 | 🧪 |
| **SenseVoice** (`sensevoice`) | 极速 ASR + 情感/事件标签 | 5+ 种语言 | custom ⚠️ | 🧪 |
| **WhisperX** (`whisperx`) | ASR + 词级对齐 + 说话人分离 | whisper 支持的语言 | BSD‑2（+pyannote 需授权） | 🧪 |

<sub>¹ 随附的检查点为韩语；MeloTTS 系列本身是多语言的 —— 额外的语言适配器可以轻松即插即用。</sub>

此外还提供两个无依赖的**参考引擎**（`reference_tone`、`reference_asr`），
让你在干净的安装环境中即可冒烟测试整条流水线，同时它们也充当适配器模板。

## 🚀 快速开始

```bash
# 1) Clone into your ComfyUI custom_nodes
cd ComfyUI/custom_nodes
git clone https://github.com/Streamize-llc/ComfyUI-Voice

# 2) Install an engine's deps (example: the workhorses)
pip install faster-whisper          # ASR
pip install librosa g2pkk jamo python-mecab-ko python-mecab-ko-dic num2words anyascii  # MeloTTS frontend
```

重启 ComfyUI，然后在节点图中：

1. 添加 **`Voice TTS 🎙️`** → 选择一个 `engine` → 输入文本。
2. 将其 `AUDIO` 输出接入核心节点 **`Save Audio`**（或 `Preview Audio`）。
3. 若要进行 STT，添加 **`Voice ASR (STT) 🎙️`**，向它输入任意 `AUDIO`，即可读取转录结果。

不确定装了哪些引擎？放置一个 **`Voice Engine Info 🎙️`** 节点 —— 它会列出每个引擎、
其能力，以及对尚未启用项的 `pip install …` 提示。

> 部分重量级引擎（例如 CosyVoice 3.0）需要一次性的模型/代码配置。
> 每个适配器在 `comfyui_voice/engines/` 中的文档字符串都记录了其确切步骤。

## 🧩 节点

| 节点 | 类别 | 输入 → 输出 |
|---|---|---|
| **Voice TTS** | `audio/voice/tts` | text（+ 可选 `VOICE_REF`） → `AUDIO` |
| **Voice ASR (STT)** | `audio/asr` | `AUDIO` → `VOICE_TRANSCRIPT` + text |
| **Voice Engine Info** | `audio/voice/util` | — → 引擎/能力报告 |

声音克隆只需将一段参考片段（通过核心节点 **Load Audio**）接入
TTS 节点的 `voice_ref` 输入即可 —— 无需任何特殊的上传步骤。

## 🏗️ 架构

整体设计借鉴了 ComfyUI 核心自身的理念 —— *重新实现/移植模型代码使其
运行在同一个 torch 上*，而不是为每个模型各自 pip 安装一整套技术栈。

- **可组合的类型化插槽。** 核心 `AUDIO` 类型（`{"waveform": [B,C,T], "sample_rate"}`）
  外加一小组带命名空间的 `VOICE_*` 类型（`VOICE_REF`、`VOICE_TRANSCRIPT`、
  `VOICE_STEMS`、……），使节点之间可互操作并具备面向未来的扩展性。
- **能力驱动。** 每个引擎声明一个 `EngineCapabilities` 数据类
  （语言、克隆能力、采样率、隔离方式、许可证、参数 schema）。表单、
  校验与工具接口都由它**自动生成** —— 任何地方都没有 `if engine == …` 这样的分支。
- **两种适配器层级。**
  1. **原生（`inproc`，推荐）：** 运行在宿主 torch 上 —— 通过 Transformers
     （这个架构库已经存在），或者通过内置上游推理代码并搭配少量纯 Python 依赖
     与加载时垫片实现。*所有已验证的模型都使用这种方式。*
  2. **隔离（`subprocess`）：** 适用于其锁定依赖确实存在冲突的引擎
     —— 它们运行在每个引擎独立的虚拟环境中，背后采用统一的 worker 协议。
     只需设置一个 `isolation="subprocess"` 字段即可选用。
- **套件自有的模型管理器**会跟踪原始模型的 VRAM 占用（ComfyUI 自身无法看到）并
  跨节点进行驱逐回收；采样率守卫则可在串联时防止静默的数据损坏。

## ➕ 一个文件新增一个引擎

复制 `comfyui_voice/engines/tts/_reference_tone.py` 并填入内容：

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

这就是全部约定 —— 它会自动出现在下拉菜单中，表单会根据其
`param_schema` 自动适配，校验也会遵循其能力声明。无需编辑任何核心文件。

> **在向主环境添加依赖之前，请先运行 `pip install --dry-run` 并确认它不会改动
> ComfyUI 的 `torch` / `transformers` / `numpy`。** 如果会，请声明
> `isolation="subprocess"`。

## 🗺️ 路线图

- [x] 核心框架、类型化插槽、能力注册表、隔离运行时
- [x] TTS + ASR 节点；已验证的原生 TTS（预设与零样本克隆）及 ASR
- [ ] 更多已验证引擎（Qwen3‑TTS、Chatterbox、……）
- [ ] `VOICE_REF` 克隆交互体验 + 声音库
- [ ] 声音转换（RVC / Seed‑VC）
- [ ] 声源分离、降噪/增强、强制对齐与字幕
- [ ] 音乐 / 音效生成、音频编辑

## 🤝 贡献

非常欢迎提交 PR 和新的引擎适配器。一个好的贡献应当：

1. 是 `comfyui_voice/engines/<task>/` 下的**单个适配器文件**。
2. 通过纯核心测试：`python tests/test_core.py`（无需任何模型）。
3. 保持宿主技术栈完好（`pip install --dry-run` 干净，或使用 `subprocess`）。
4. 声明准确的许可证 + `commercial_safe` 标志。

## 📜 许可证与致谢

ComfyUI-Voice 以 **Apache‑2.0** 许可证发布。每个被封装的模型都保留其**自身的**
许可证 —— 详见上方表格及各引擎的适配器；其中部分是非商业用途或有使用限制的。
你需要自行负责遵守你所启用的任何模型的许可证。

本项目站在开源语音社区的肩膀上构建 —— MeloTTS、CosyVoice、
Supertonic、MMS、OpenAI Whisper / faster‑whisper、Kokoro、Chatterbox、Qwen、OuteTTS、
SenseVoice、WhisperX，以及 ComfyUI 本身。感谢你们。🙏
