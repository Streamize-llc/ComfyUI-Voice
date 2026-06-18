<div align="center">

# ComfyUI-Voice 🎙️

**ComfyUI のための、統合された拡張可能な音声・ボイススイート。**
TTS · ボイスクローン · 音声認識 — 最先端のオープンモデルを多数、ひとつのクリーンなノードグラフで。

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-8A2BE2.svg)](https://github.com/comfyanonymous/ComfyUI)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-contributing)

<br/>

[English](README.md) · [한국어](README.ko.md) · [中文](README.zh.md) · **日本語**

</div>

---

ComfyUI には世界最高クラスの画像・動画ツールがあります。しかし音声については、
それぞれが独自の競合する依存関係を引きずり込み、しばしばインストールを壊してしまう、
場当たり的なノードの寄せ集めにとどまっていました。**ComfyUI-Voice はそのギャップを埋めます**。
これは、あらゆる音声モデルがひとつのノード群の背後にあるドロップイン式アダプターとして動作する、
単一かつ一貫したスイートであり、**すべてが今お使いの PyTorch 上でネイティブに動作します** —
2 つ目の環境も、依存関係地獄もありません。

```
text ─▶ [Voice TTS] ─▶ AUDIO ─▶ [Save Audio]            # synthesize
audio ─▶ [Voice ASR] ─▶ TRANSCRIPT + text               # transcribe
ref ──▶ [Voice TTS] ─▶ AUDIO                             # zero‑shot voice clone
```

## ✨ ハイライト

- 🧩 **タスクごとに 1 ノード、エンジンは多数。** ドロップダウンからエンジンを選ぶと、
  そのエンジンが宣言する機能に応じてノードの入力が適応します。
- 🛡️ **あなたの torch 上で動作。** エンジンは ComfyUI の既存の
  PyTorch/Transformers 上でネイティブにロードされます。インストールと競合するバージョンピンも、
  サポート対象のために別の venv を用意する必要もありません。(競合するモデルは分離を選択できますが、
  検証済みモデルでそれを必要とするものはありません。)
- 🌍 **多言語かつ SOTA。** 数十の言語にわたる音声合成、ゼロショットボイスクローン、
  文字起こしのための主要なオープンモデルをラップします。
- 🔌 **1 ファイルでモデルを追加。** 新しいエンジンは、自己登録するアダプター 1 つで完結します —
  中央ディスパッチも、触るべき UI の配線もありません。
- 🧵 **組み合わせ可能。** すべてが ComfyUI ネイティブの `AUDIO` 型を話すため、
  組み込みの Load/Save/Preview Audio ノードとそのまま連結できます。
- ✅ **正直なステータス。** モデルは、このコードベース上で実際のラウンドトリップテストを
  通過してはじめて *検証済み* と表示されます。

## 🎧 サポート対象エンジン

> **検証済み (✅)** = ホストスタック上で実際の推論をエンドツーエンド(合成 → 文字起こしの
> ラウンドトリップ)でテスト済み。**実験的 (🧪)** = アダプターは同梱済みだが依存関係が
> まだインストール/検証されていない — *Voice Engine Info* ノードのインストールヒントから有効化してください。

### Text‑to‑Speech

| エンジン (`id`) | 機能 | 言語 | ライセンス | ステータス |
|---|---|---|---|---|
| **MeloTTS** (`melotts_korean`) | 高速プリセット TTS | Korean¹ | Apache‑2.0 / MIT | ✅ |
| **CosyVoice 3.0** (`cosyvoice3`) | **ゼロショットボイスクローン** + VC | zh · en · ja · ko · de · es · fr · it · ru | Apache‑2.0 | ✅ |
| **Supertonic 3** (`supertonic`) | オンデバイスプリセット TTS (ONNX) | 30+ incl. ko | code MIT / model OpenRAIL‑M | ✅ |
| **Higgs Audio v3 (4B)** (`higgs_audio_v3`) | 表現力豊かな TTS + ゼロショットクローン | 100+ (ko·zh·ja 含む) | **研究/非商用** ⚠️ | ✅ (eval) |
| **MMS‑TTS** (`mms_tts_korean`) | プリセット TTS (Transformers VITS) | Korean¹ | CC‑BY‑NC ⚠️ | ✅ (eval) |
| **Kokoro‑82M** (`kokoro`) | 超軽量プリセット TTS | en · es · fr · hi · it · pt · ja · zh | Apache‑2.0 | 🧪 |
| **Chatterbox** (`chatterbox`) | クローン + 感情制御 | 23 langs | MIT | 🧪 |
| **Qwen3‑TTS** (`qwen3_tts`) | クローン + ボイスデザイン | 10 langs | Apache‑2.0 | 🧪 |
| **OuteTTS 1.0** (`oute_tts`) | コンパクトな LLM‑TTS | 14 langs | Apache‑2.0 | 🧪 |

### Speech‑to‑Text

| エンジン (`id`) | 機能 | 言語 | ライセンス | ステータス |
|---|---|---|---|---|
| **faster‑whisper** (`faster_whisper`) | 高速 ASR + 単語タイムスタンプ | 99 langs | MIT | ✅ |
| **Whisper large‑v3** (`whisper_v3`) | ASR (Transformers) | 99 langs | Apache‑2.0 | ✅ |
| **Korean Whisper** (`korean_whisper`) | ASR (ko ファインチューン) | Korean | Apache‑2.0 | 🧪 |
| **Qwen3‑ASR** (`qwen3_asr`) | ASR + 強制アライナーによるタイムスタンプ | 30+ langs | Apache‑2.0 | 🧪 |
| **SenseVoice** (`sensevoice`) | 非常に高速な ASR + 感情/イベントタグ | 5+ langs | custom ⚠️ | 🧪 |
| **WhisperX** (`whisperx`) | ASR + 単語アライメント + 話者分離 | whisper langs | BSD‑2 (+pyannote gated) | 🧪 |

<sub>¹ 同梱のチェックポイントは Korean ですが、MeloTTS/MMS ファミリーは多言語対応です — 追加の言語アダプターは容易にドロップインできます。</sub>

さらに、依存関係不要の **リファレンスエンジン** が 2 つ (`reference_tone`、`reference_asr`) あり、
クリーンインストール上でパイプライン全体をスモークテストでき、アダプターのテンプレートとしても機能します。

## 🚀 クイックスタート

```bash
# 1) Clone into your ComfyUI custom_nodes
cd ComfyUI/custom_nodes
git clone https://github.com/Streamize-llc/ComfyUI-Voice

# 2) Install an engine's deps (example: the workhorses)
pip install faster-whisper          # ASR
pip install librosa g2pkk jamo python-mecab-ko python-mecab-ko-dic num2words anyascii  # MeloTTS frontend
```

ComfyUI を再起動し、グラフ内で:

1. **`Voice TTS 🎙️`** を追加 → `engine` を選択 → テキストを入力。
2. その `AUDIO` 出力をコアの **`Save Audio`**(または `Preview Audio`)へ配線。
3. STT の場合は **`Voice ASR (STT) 🎙️`** を追加し、任意の `AUDIO` を入力して文字起こしを読み取ります。

何がインストールされているか分からない? **`Voice Engine Info 🎙️`** ノードを置いてみてください —
すべてのエンジン、その機能、そしてまだ有効化されていないものについては `pip install …` のヒントを一覧表示します。

> 一部の重量級エンジン(例: CosyVoice 3.0)は、一度きりのモデル/コードのセットアップが必要です。
> `comfyui_voice/engines/` 内の各アダプターの docstring に、その正確な手順が記載されています。

## 🧩 ノード

| ノード | カテゴリ | In → Out |
|---|---|---|
| **Voice TTS** | `audio/voice/tts` | text (+ optional `VOICE_REF`) → `AUDIO` |
| **Voice ASR (STT)** | `audio/asr` | `AUDIO` → `VOICE_TRANSCRIPT` + text |
| **Voice Engine Info** | `audio/voice/util` | — → エンジン/機能レポート |

ボイスクローンは、参照クリップを(コアの **Load Audio** 経由で)TTS ノードの `voice_ref` 入力に
配線するだけです — 特別なアップロード手順はありません。

## 🏗️ アーキテクチャ

この設計は ComfyUI コア自身の哲学を踏襲しています — モデルごとにスタックを pip インストールするのではなく、
*ひとつの torch 上で動くようモデルコードを再実装/移植する* という考え方です。

- **組み合わせ可能な型付きソケット。** コアの `AUDIO` (`{"waveform": [B,C,T], "sample_rate"}`)
  に加え、名前空間付きの `VOICE_*` 型の小さなセット(`VOICE_REF`、`VOICE_TRANSCRIPT`、
  `VOICE_STEMS`、…)により、ノードは相互運用可能で将来にわたって安心です。
- **機能駆動。** 各エンジンは `EngineCapabilities` データクラスを 1 つ宣言します
  (言語、クローン、サンプルレート、分離、ライセンス、パラメータスキーマ)。フォーム、
  バリデーション、ツールサーフェスはそこから **生成** されます — どこにも `if engine == …`
  の分岐はありません。
- **2 つのアダプター階層。**
  1. **ネイティブ (`inproc`、推奨):** ホストの torch 上で動作します — Transformers
     (すでに存在するアーキテクチャライブラリ)経由か、もしくは上流の推論コードを
     小さな純 Python 依存 + ロード時のシムとともにベンダリングして動かします。*すべての検証済みモデルがこれを使用します。*
  2. **分離 (`subprocess`):** ピン留めされた依存関係が本当に競合するエンジン向けです —
     統一されたワーカープロトコルの背後で、エンジンごとの venv で実行されます。
     `isolation="subprocess"` というフィールド 1 つでオプトインします。
- **スイート所有のモデルマネージャー** が、生のモデルの VRAM(ComfyUI には見えない)を追跡し、
  ノードをまたいで退避させます。サンプルレートのガードが、連結時の無言の破損を防ぎます。

## ➕ 1 ファイルでエンジンを追加

`comfyui_voice/engines/tts/_reference_tone.py` をコピーして埋めるだけです:

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

これがすべての契約です — ドロップダウンに自動的に表示され、フォームはその `param_schema` に
適応し、バリデーションはその機能を尊重します。編集すべきコアファイルはありません。

> **メイン環境に依存関係を追加する前に、`pip install --dry-run` を実行し、ComfyUI の
> `torch` / `transformers` / `numpy` を動かさないことを確認してください。** 動かしてしまう場合は、
> `isolation="subprocess"` を宣言してください。

## 🗺️ ロードマップ

- [x] コアフレームワーク、型付きソケット、機能レジストリ、分離ランタイム
- [x] TTS + ASR ノード。検証済みのネイティブ TTS(プリセット & ゼロショットクローン)と ASR
- [ ] さらなる検証済みエンジン(Qwen3‑TTS、Chatterbox、…)
- [ ] `VOICE_REF` クローンの UX + ボイスライブラリ
- [ ] ボイス変換 (RVC / Seed‑VC)
- [ ] 音源分離、ノイズ除去/エンハンス、強制アライメント & 字幕
- [ ] 音楽 / SFX 生成、オーディオ編集

## 🤝 コントリビュート

PR や新しいエンジンアダプターを大歓迎します。良いコントリビューションは:

1. `comfyui_voice/engines/<task>/` 配下の **単一アダプターファイル** であること。
2. 純コアテストを通過すること: `python tests/test_core.py`(モデル不要)。
3. ホストスタックを壊さないこと(`pip install --dry-run` がクリーン、もしくは `subprocess`)。
4. 正確なライセンスと `commercial_safe` フラグを宣言すること。

## 📜 ライセンスとクレジット

ComfyUI-Voice は **Apache‑2.0** ライセンスの下で公開されています。ラップされた各モデルは
**それ自身の** ライセンスを保持します — 上記の表と各エンジンのアダプターを参照してください。
一部は非商用または利用制限付きです。有効化したモデルのライセンスを順守する責任はあなたにあります。

オープンソース音声コミュニティの肩の上に築かれています — MeloTTS、CosyVoice、
Supertonic、MMS、OpenAI Whisper / faster‑whisper、Kokoro、Chatterbox、Qwen、OuteTTS、
SenseVoice、WhisperX、そして ComfyUI 自身。ありがとう。🙏
