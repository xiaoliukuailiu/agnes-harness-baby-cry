# 工作流形式化描述（AGH 智能体复现依据）

本文件把「任务规划 → 能力调用 → 反馈处理与结果验证」写成 AGH 可复现的形式化步骤。
`run.mjs` 是这个描述的**可执行实现**（离线 mock 或 `--live` 真实推理）；
AGH 侧的智能体（LLM）读这里也能按同样的顺序发起工具调用。

## 0. 前置

- 后端 8800（`backend-ssl/server.py`，自训练 Wav2Vec2 checkpoint）已启动，CORS 已放行 127.0.0.1。
- AGH 已加载插件 `local-plugins/baby-cry-ssl/`，注册了 4 个工具。

## 1. 任务规划（Planning）

输入：一条婴儿哭声音频（`audio_path`，绝对路径）。

规划器先判定**能否开始**：

```
if 后端未就绪 (health.status != 'ready'):
    输出诊断，终止（不要继续 analyze）
else:
    计划 = [analyze(audio), verify(result), labels() → advice/top_label]
```

规划产出一份**有序工具调用计划**，并预判 3 种终态分支：
- 分支 A：`top_confidence ≥ 0.60` 且 `cry_segment_ratio ≥ 0.3` → 正常出护理建议
- 分支 B：置信度不足或哭声占比低 → 追加「建议人工复核 / 重新录制」
- 分支 C：命中任一 RED_FLAG 场景 → 升级为「立即就医」

## 2. 能力调用（Tool Invocation）

按规划依次调用，每步都记录 (工具名, 入参, 出参) 三元组，供后续验证与留痕：

| 序 | 工具 | 入参 | 出参关键字段 |
|---|---|---|---|
| ① | `baby_cry_ssl_health` | — | `status`, `model_base`, `num_labels`, `labels` |
| ② | `analyze_baby_cry_ssl` | `{ audio_path, top_k: 6 }` | `status`, `top_label`, `top_label_zh`, `top_confidence`, `distribution`, `acoustics`, `advice`, `warnings` |
| ③ | `verify_baby_cry_result` | `{ result, expected_label? }` | `passed`, `severity`, `checks[]`, `failed_names` |
| ④ | `baby_cry_ssl_labels` | — | `labels`(en→zh), `advice`(label→建议[]), `red_flags`[] |

> ① 未就绪时不执行 ②③④；④ 在 ② 之后执行，用于把 ② 的 `top_label` 映射到人类可读建议；③ 对 ② 的结果做客观校验。

## 3. 反馈处理 + 结果验证（Feedback & Verification）

### 3a. 反馈处理（消费 warnings）

读 ② 的 `warnings[]`，逐条映射为护理/就医动作：

```
for w in warnings:
    if w 含「区分度不足」:   severity = 'manual_review'
    if w 含「占比偏低」:      severity = 're_record'
    if w 含「主频…低于」:    severity = 'non_baby_source'
命中 red_flags 中任一临床信号 → severity = 'urgent_care'
```

### 3b. 确定性结果验证（verify_baby_cry_result）

这是**客观、可回归**的校验，不依赖 LLM 主观判断（实现见 `verify.mjs`）：

```
checks = [
  { name:'confidence', pass: top_confidence >= 0.60, value: top_confidence },
  { name:'cry_ratio',  pass: acoustics.cry_segment_ratio >= 0.30 },
  { name:'acoustic',   pass: acoustics.estimated_f0_hz in [150,1500] },  # 婴儿声源
  { name:'label_in_vocabulary', pass: top_label in labels.keys() },
]
if expected_label: checks += [{ name:'accuracy', pass: top_label == expected_label }]
passed = all(c.pass for c in checks)
```

### 3c. 生成结构化报告

把 ②③ + 验证结果合并成一份**机器可读报告**（见 `fixture/result/*.json`），
并落到 `fixture/result/`（离线模式）或 stdout（live 模式）：

```jsonc
{
  "audio": "...",
  "plan": ["health","analyze","verify","labels"],
  "analyze": { "top_label":"hungry", "top_confidence":0.82, "warnings":[...] },
  "verify":  { "passed": true, "checks":[...] },
  "advice":  ["观察觅食信号…", "..."],
  "red_flags": [...],
  "severity": "normal",
  "trace": [ ["baby_cry_ssl_health",{}, "..."], ["analyze_baby_cry_ssl",{"audio_path":"..."},"..."], ... ]
}
```

## 4. 退出条件

- 3 段 fixture 中，正常/边界段（01、03）`verify.passed == true` 且 `accuracy` 命中，
  反例段（02）按 `expect_fail` 命中预期失败原因 → 汇总 `PASS · 2/3 · 1 段预期反例`
- 出现**意外**失败（正常段未过、或反例未命中预期失败原因）或后端未就绪 → 退出码 1

## 5. 与 AGH 真实会话的对应

在 AGH 里，这条链路就是模型按上述顺序发起的 tool call：

```
[模型] 先确认后端就绪
→ 调 baby_cry_ssl_health
[模型] 后端就绪，分析这段哭声
→ 调 analyze_baby_cry_ssl(audio_path)
[模型] 置信度 0.82，读 warnings 无异常；做客观校验
→ 调 verify_baby_cry_result(结构化结果, expected_label='hungry')
[模型] 把结果映射到护理建议
→ 调 baby_cry_ssl_labels
[模型] 输出报告（top_label=hungry，建议=按需喂养…，无就医信号）
```

`run.mjs` 把这个序列确定性重放，使评分可复现、可回归。
