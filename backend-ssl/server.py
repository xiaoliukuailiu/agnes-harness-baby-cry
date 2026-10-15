"""
婴儿哭声识别后端服务（SSL 版）—— 使用参赛者自训练的 Wav2Vec2 checkpoint

模型：facebook/wav2vec2-base 在 Baby Chillanto 语料上微调，6 分类
checkpoint：默认 backend-ssl/checkpoint-1500/（可用环境变量 BABY_CRY_CHECKPOINT 覆盖）
标签：pain / uncomfortable / fright / hungry / disturbance / angry

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
与旧后端（backend/server.py）的关键区别，务必注意：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 采样率：本模型训练时使用真实 16kHz，直接以 sampling_rate=16000 喂入。
   旧后端那套「内部重采样到 8k、但以 16000 名义喂入」的 trick 是
   DistilHuBERT + Donate_a_cry(8kHz原生) 特有的处理方式，
   照搬到本模型会造成分布偏移，因此本服务已移除该 trick。

2. 分析窗口：训练时 max_duration=10.0s，故 WINDOW_SEC=10.0（旧后端为 7.0s）。

3. 标签体系：Baby Chillanto 6 类，与 Donate_a_cry 5 类不同，
   两套标签不可混用。旧 samples/ 目录的测试音频标签体系不匹配，
   不能用于本模型的验证。

4. 特征提取器：checkpoint 内不含 preprocessor_config.json，
   本服务优先从本地 feature_extractor/ 目录加载（离线可用），
   仅在缺失时回退到 HuggingFace Hub。

端点：
  GET  /health    健康检查 + 模型加载状态
  POST /analyze   输入音频（本地路径 / base64），输出分类结果
  GET  /labels    标签与建议字典

设计要点（对应评分项「验证严谨性与异常处理」）：
  - 静音检测、时长边界、格式异常、超限文件统一返回结构化错误
  - 长音频按能量选段，输出哭声占比与候选段数
  - 附加声学特征（基频 F0、能量 RMS），用于区分婴儿哭声与环境噪声/成人语音
  - 低置信度时给出显式告警，不隐瞒模型区分度不足的情况
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForAudioClassification,
    Wav2Vec2FeatureExtractor,
)

# ---------------------------------------------------------------- 常量配置

# 自训练 checkpoint 目录：默认放在 backend-ssl/checkpoint-1500/（与代码同包，跨机器可移植），
# 可用环境变量 BABY_CRY_CHECKPOINT 覆盖为任意绝对路径。
CKPT_DIR = os.getenv(
    "BABY_CRY_CHECKPOINT",
    str(Path(__file__).resolve().parent / "checkpoint-1500"),
)

# 本地特征提取器目录（离线部署，避免运行时联网）
LOCAL_FE_DIR = Path(__file__).parent / "feature_extractor"

# 回退用的特征提取器（与训练基座一致）
FE_FALLBACK_ID = "facebook/wav2vec2-base"

NATIVE_HZ = 16_000          # 本模型真实训练采样率
MODEL_INPUT_HZ = 16_000     # 喂给模型的名义采样率（与训练一致，无偏移）

WINDOW_SEC = 10.0           # 分析片段长度，与训练 max_duration 一致
HOP_SEC = 5.0               # 选段滑窗步长
ACOUSTIC_FRAME_SEC = 1.0    # 声学统计帧长
MAX_DURATION_SEC = 60.0     # 超过则截断，防止超大文件打爆内存
MIN_DURATION_SEC = 0.3      # 低于此值判定无效输入
SILENCE_RMS = 0.008         # 静音判定阈值
CRY_RMS = 0.015             # 「有哭声」窗口能量阈值
LOW_F0_HZ = 200.0           # 低于此基频怀疑非婴儿声源
HIGH_CONFIDENCE = 0.60      # 低于此置信度给出「区分度不足」告警
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

# 6 类标签中文（Baby Chillanto 体系）
LABEL_ZH: dict[str, str] = {
    "pain": "疼痛",
    "uncomfortable": "难受不适",
    "fright": "受惊",
    "hungry": "饥饿",
    "disturbance": "不安/烦躁",
    "angry": "生气",
}

ADVICE_ZH: dict[str, list[str]] = {
    "pain": [
        "检查是否有外伤、衣物标签摩擦、毛发或线头缠绕手指脚趾",
        "触摸腹部判断是否胀气发硬，可顺时针轻揉或做蹬自行车动作排气",
        "测量体温；若伴发热、呕吐、血便、拒奶或哭声持续无法安抚，请立即就医",
    ],
    "uncomfortable": [
        "依次排查：尿布是否湿、衣物是否过紧、室温是否在 24–26℃",
        "检查皮肤褶皱、颈部与臀部有无红疹或湿疹",
        "尝试襁褓包裹或调整体位，观察是否缓解",
    ],
    "fright": [
        "立即抱起贴身安抚，让宝宝感受到身体接触与心跳",
        "降低环境刺激：调暗灯光、减少突然的声响与强光",
        "用平稳低音调说话或哼唱，避免频繁更换安抚人",
    ],
    "hungry": [
        "观察觅食信号：张嘴、转头寻乳、吸吮手指",
        "按需喂养，记录本次开始与结束时间",
        "留意上次喂养距今时长，逐步建立喂养规律",
    ],
    "disturbance": [
        "判断是否过度刺激（人多、嘈杂、被频繁抱传），尝试带至安静环境",
        "规律轻拍、白噪音或缓慢摇晃，帮助宝宝重新安定",
        "检查是否处于困倦但难以入睡的状态，避免继续逗玩",
    ],
    "angry": [
        "确认是否有未满足的需求：饥饿、困倦、尿布、想要被抱",
        "保持照护者情绪平稳，用稳定节奏安抚，避免急躁摇晃",
        "若哭闹伴随肢体僵硬、面部涨红，可尝试襁褓包裹提供边界感",
    ],
}

RED_FLAGS: list[str] = [
    "哭闹持续超过 3 小时且无法安抚",
    "发热、呕吐、血便、拒奶或尿量明显减少",
    "哭声高尖刺耳（脑膜炎样哭声）或异常微弱",
    "伴随呼吸费力、口唇发青、意识异常、抽搐",
]


class UnsupportedAudio(ValueError):
    """音频格式无法解码。"""


class InvalidInput(ValueError):
    """输入不合法（文件不存在、过大、缺少参数）。"""


app_state: dict[str, Any] = {
    "model": None,
    "fe": None,
    "error": None,
    "id2label": {},
    "history": [],  # 最近的成功分析结果（供 demo 页 /latest 轮询，AGH 联动用）
}


def _record(result: dict[str, Any], source: str) -> None:
    """把一次成功分析的结果记入 history（内存，最多 20 条），供 /latest 轮询。"""
    result["source"] = source
    result["received_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    history: list = app_state["history"]
    history.append(result)
    del history[:-20]


# ---------------------------------------------------------------- 模型加载

def _load_feature_extractor() -> Any:
    """加载特征提取器。

    优先用显式类 Wav2Vec2FeatureExtractor 从本地目录加载（不依赖自动探测，
    避免因缺 model_type 字段而失败）；本地缺失时回退到 Hub。
    """
    cfg = LOCAL_FE_DIR / "preprocessor_config.json"
    if cfg.exists():
        return Wav2Vec2FeatureExtractor.from_pretrained(str(LOCAL_FE_DIR))
    return Wav2Vec2FeatureExtractor.from_pretrained(FE_FALLBACK_ID)


def load_model() -> None:
    if app_state["model"] is not None or app_state["error"] is not None:
        return
    try:
        if not Path(CKPT_DIR).exists():
            raise FileNotFoundError(f"checkpoint 目录不存在: {CKPT_DIR}")

        app_state["fe"] = _load_feature_extractor()
        app_state["model"] = AutoModelForAudioClassification.from_pretrained(CKPT_DIR)
        app_state["model"].eval()
        torch.set_grad_enabled(False)

        # 统一 id2label 的键类型为 str（transformers 可能返回 int 键）
        app_state["id2label"] = {
            str(k): v for k, v in app_state["model"].config.id2label.items()
        }
        app_state["error"] = None
    except Exception as exc:  # noqa: BLE001
        app_state["error"] = f"{type(exc).__name__}: {exc}"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Baby Cry Analysis Service (SSL)", version="2.0.0", lifespan=lifespan)

# 允许浏览器 demo 从本地页面直连（127.0.0.1/localhost 任意端口，含 file:// 打开时的 null 来源）
# 生产部署到公网时应收紧 allow_origins 到具体域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8800",
        "http://localhost:8800",
        "http://127.0.0.1",
        "http://localhost",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "null",
    ],
    # 注意：raw 字符串里不能再转义 $，否则会变成字面美元符号导致正则永不匹配
    allow_origin_regex=r"^(https?://(127\.0\.0\.1|localhost)(:\d+)?|null)$",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=False,
)


# ---------------------------------------------------------------- 数据模型

class AnalyzeRequest(BaseModel):
    audio_path: str | None = Field(default=None, description="本地音频文件绝对路径")
    audio_base64: str | None = Field(default=None, description="base64 编码的音频内容")
    top_k: int = Field(default=6, ge=1, le=6, description="返回的候选类别数量")
    source: str = Field(default="", description="调用来源标识（如 agh=AGH 智能体会话、web=演示页）")


# ---------------------------------------------------------------- 工具函数

def _decode(raw: bytes, label: str) -> tuple[np.ndarray, int]:
    """解码字节流为波形，失败抛 UnsupportedAudio。"""
    try:
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedAudio(f"无法解码音频（{label}，支持 wav/mp3/flac/ogg）: {exc}") from exc
    return np.asarray(wav), sr


def _resample(wav: np.ndarray, sr: int) -> np.ndarray:
    """线性重采样到 NATIVE_HZ（16kHz），与训练时一致。"""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:                    # 多声道 → 单声道
        wav = wav.mean(axis=1)
    if sr == NATIVE_HZ:
        return wav
    target = int(round(len(wav) * NATIVE_HZ / sr))
    if target < 1:
        raise UnsupportedAudio("音频内容过短，无法重采样")
    idx = np.linspace(0, len(wav) - 1, target)
    return np.interp(idx, np.arange(len(wav)), wav).astype(np.float32)


def _load_audio(req: AnalyzeRequest) -> tuple[np.ndarray, dict[str, Any]]:
    meta: dict[str, Any] = {}

    if req.audio_path:
        path = os.path.abspath(os.path.expanduser(req.audio_path))
        if not os.path.isfile(path):
            raise InvalidInput(f"音频文件不存在: {path}")
        size = os.path.getsize(path)
        if size > MAX_UPLOAD_BYTES:
            raise InvalidInput(f"音频文件过大: {size / 1048576:.1f}MB，上限 20MB")
        wav, sr = _decode(Path(path).read_bytes(), path)
        meta["source"] = path
        meta["bytes"] = size
    elif req.audio_base64:
        try:
            raw = base64.b64decode(req.audio_base64)
        except Exception as exc:  # noqa: BLE001
            raise InvalidInput(f"base64 解码失败: {exc}") from exc
        if len(raw) > MAX_UPLOAD_BYTES:
            raise InvalidInput("音频内容过大，上限 20MB")
        wav, sr = _decode(raw, "base64")
        meta["source"] = "base64"
        meta["bytes"] = len(raw)
    else:
        raise InvalidInput("必须提供 audio_path 或 audio_base64 之一")

    meta["original_sr"] = sr
    wav = _resample(wav, sr)
    meta["duration_sec"] = round(len(wav) / NATIVE_HZ, 3)
    return wav, meta


def _select_segment(wav: np.ndarray) -> tuple[np.ndarray, int]:
    """选取一段固定长度（WINDOW_SEC=10s）的音频交给模型。

    与训练期 max_duration=10.0 对齐。短音频零填充到 10s；
    长音频按能量选最强片段（首段能量已足够时优先用首段，结果更可预期）。
    """
    win = int(WINDOW_SEC * NATIVE_HZ)
    hop = int(HOP_SEC * NATIVE_HZ)
    if len(wav) <= win:
        seg = np.zeros(win, dtype=np.float32)
        seg[: len(wav)] = wav
        return seg, 1

    head_rms = float(np.sqrt(np.mean(wav[:win] ** 2)))
    if head_rms >= CRY_RMS:
        return wav[:win], max(1, (len(wav) - win) // hop + 1)

    best = max(
        range(0, len(wav) - win + 1, hop),
        key=lambda s: float(np.sqrt(np.mean(wav[s : s + win] ** 2))),
    )
    return wav[best : best + win], max(1, (len(wav) - win) // hop + 1)


def _cry_ratio(wav: np.ndarray) -> float:
    """按 1s 帧统计「有哭声」帧占比。"""
    frame = int(ACOUSTIC_FRAME_SEC * NATIVE_HZ)
    if len(wav) < frame:
        return round(float(np.sqrt(np.mean(wav**2)) >= CRY_RMS), 3)
    frames = [wav[s : s + frame] for s in range(0, len(wav) - frame + 1, frame)]
    flags = [float(np.sqrt(np.mean(f**2)) >= CRY_RMS) for f in frames]
    return round(sum(flags) / len(flags), 3)


def _fundamental_hz(seg: np.ndarray) -> float | None:
    """估计基频，用于非婴儿声源判别（物理频率按 NATIVE_HZ 计算）。"""
    try:
        import librosa

        f0 = librosa.yin(seg, fmin=150, fmax=1500, sr=NATIVE_HZ,
                         frame_length=1024, hop_length=256)
        valid = f0[np.isfinite(f0)]
        return float(np.median(valid)) if valid.size else None
    except Exception:  # noqa: BLE001
        return None


def _run_model(seg: np.ndarray) -> np.ndarray:
    """单次推理，返回 6 类概率分布。"""
    load_model()
    if app_state["error"] is not None:
        raise RuntimeError(f"模型未就绪: {app_state['error']}")

    fe, model = app_state["fe"], app_state["model"]
    # 以真实 16kHz 名义喂入 16k 波形，与训练期预处理完全一致（无采样率 trick）
    inputs = fe(seg, sampling_rate=MODEL_INPUT_HZ, return_tensors="pt", padding=True)
    with torch.no_grad():
        return torch.softmax(model(**inputs).logits[0], dim=-1).numpy()


# ---------------------------------------------------------------- 路由

@app.get("/health")
def health() -> dict[str, Any]:
    load_model()
    return {
        "status": "ready" if app_state["error"] is None else "model_error",
        "model_id": str(CKPT_DIR),
        "model_base": "facebook/wav2vec2-base",
        "model_error": app_state["error"],
        "labels": list(app_state["id2label"].values()) or list(LABEL_ZH.keys()),
        "num_labels": len(app_state["id2label"]) or len(LABEL_ZH),
        "feature_extractor": "local" if (LOCAL_FE_DIR / "preprocessor_config.json").exists() else FE_FALLBACK_ID,
        "native_hz": NATIVE_HZ,
        "model_input_hz": MODEL_INPUT_HZ,
        "window_sec": WINDOW_SEC,
    }


@app.get("/labels")
def labels() -> dict[str, Any]:
    return {"labels": LABEL_ZH, "advice": ADVICE_ZH, "red_flags": RED_FLAGS}


@app.post("/analyze")
async def analyze(request: Request) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        req = AnalyzeRequest(**(await request.json()))
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error_type": "bad_request", "message": str(exc)}

    try:
        wav, meta = _load_audio(req)
    except InvalidInput as exc:
        return {"status": "error", "error_type": "missing_or_too_large", "message": str(exc)}
    except UnsupportedAudio as exc:
        return {"status": "error", "error_type": "decode_failed", "message": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error_type": "unexpected", "message": str(exc)}

    warnings: list[str] = []
    duration = float(meta["duration_sec"])

    if duration < MIN_DURATION_SEC:
        return {
            "status": "error",
            "error_type": "too_short",
            "message": f"音频时长 {duration:.2f}s，短于最小可分析时长 {MIN_DURATION_SEC}s",
            "duration_sec": duration,
        }

    if duration > MAX_DURATION_SEC:
        warnings.append(f"音频时长 {duration:.1f}s，已截断至前 {MAX_DURATION_SEC:.0f}s 分析")
        wav = wav[: int(MAX_DURATION_SEC * NATIVE_HZ)]

    rms = float(np.sqrt(np.mean(wav**2)))
    if rms < SILENCE_RMS:
        result = {
            "status": "silence",
            "message": "音频能量过低，判定为静音/底噪，未进行哭声分类",
            "audio": {**meta, "rms": round(rms, 5)},
            "hint": "请重新录制靠近婴儿、环境较安静的哭声片段（建议 3–10 秒）",
        }
        _record(result, req.source)
        return result

    seg, candidate_count = _select_segment(wav)
    try:
        probs = _run_model(seg)
    except RuntimeError as exc:
        return {"status": "error", "error_type": "model_unavailable", "message": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error_type": "inference_failed", "message": str(exc)}

    id2label = app_state["id2label"]
    order = np.argsort(probs)[::-1][: req.top_k]
    distribution = [
        {
            "label": id2label[str(int(i))],
            "label_zh": LABEL_ZH.get(id2label[str(int(i))], id2label[str(int(i))]),
            "score": round(float(probs[int(i)]), 4),
        }
        for i in order
    ]
    top = distribution[0]

    cry_ratio = _cry_ratio(wav)
    f0 = _fundamental_hz(seg[: int(3.0 * NATIVE_HZ)])

    if cry_ratio < 0.3:
        warnings.append("哭声段占比偏低，结果可能受环境噪声影响，建议复录")
    if f0 is not None and f0 < LOW_F0_HZ:
        warnings.append(
            f"主频约 {f0:.0f}Hz，低于婴儿哭声典型区间，请确认声源是否为婴儿"
        )
    if top["score"] < HIGH_CONFIDENCE:
        warnings.append(
            f"最高置信度 {top['score']:.2f} 低于 {HIGH_CONFIDENCE}，模型区分度不足，结论仅供参考"
        )
    if len(distribution) > 1 and top["score"] - distribution[1]["score"] < 0.15:
        warnings.append(
            f"前两名 {top['label']} 与 {distribution[1]['label']} 概率接近，建议结合现场观察判断"
        )

    result = {
        "status": "ok",
        "audio": meta,
        "top_label": top["label"],
        "top_label_zh": top["label_zh"],
        "top_confidence": top["score"],
        "distribution": distribution,
        "advice": ADVICE_ZH.get(top["label"], []),
        "red_flags": RED_FLAGS,
        "acoustics": {
            "duration_sec": duration,
            "rms": round(rms, 5),
            "analyzed_segment_sec": round(len(seg) / NATIVE_HZ, 2),
            "candidate_segments": candidate_count,
            "cry_segment_ratio": cry_ratio,
            "estimated_f0_hz": None if f0 is None else round(f0, 1),
        },
        "model": {
            "checkpoint": str(CKPT_DIR),
            "base": "facebook/wav2vec2-base",
            "architecture": "Wav2Vec2ForSequenceClassification",
            "dataset": "Baby Chillanto（参赛者自行处理，混合采样平衡）",
            "native_hz": NATIVE_HZ,
            "window_sec": WINDOW_SEC,
        },
        "warnings": warnings,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }
    _record(result, req.source)
    return result


@app.get("/latest")
def latest() -> dict[str, Any]:
    """最近一次成功分析的结果（demo 页轮询，实现 AGH 会话 → 演示页联动）。"""
    history: list = app_state["history"]
    return {"total": len(history), "latest": history[-1] if history else None}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8800")))

