# 任务定义（Task Spec）

## 目标

给定一段婴儿哭声音频，由 AGH 智能体**连续**完成：任务规划 → 能力调用 → 反馈处理 → 结果验证，
产出一份**结构化哭声分类报告**（含类别、置信度、护理建议、就医信号、客观校验结果），
并把每一步的 tool call 留痕。

## 输入

| 项 | 说明 |
|---|---|
| `audio_path` | 待分析哭声音频绝对路径（wav/mp3/flac/ogg，3–10s） |
| `expected_label` | （可选）期望类别，用于回归断言；缺省则跳过 accuracy 检查 |
| `mode` | `offline`（默认，mock 后端）/ `live`（真实 8800 推理） |

## 必须完成的 3 个连续步骤（评分项）

1. **任务规划**：先探测后端就绪，再决定 analyze 顺序；预判正常/复核/就医 3 个终态分支。
2. **能力调用**：`baby_cry_ssl_health` → `analyze_baby_cry_ssl` → `baby_cry_ssl_labels`（有序、留痕）。
3. **反馈处理 + 结果验证**：消费 `warnings` 定 severity；`verify_baby_cry_result` 做确定性校验；
   `top_label` 映射到 `advice` 字典；合并输出结构化报告。

## 输出

- `fixture/result/<audio>.json`：结构化报告（见 workflow.md §3c 的 schema）
- stdout：每段一行摘要 + 末尾 `PASS · N/N` 或失败断言

## 验收标准（全部满足才算通过）

- [ ] 3 段 fixture 均被依次 `health → analyze → verify → labels` 处理
- [ ] 每段 `verify.passed == true`（`02_pain_lowconf` 为例外，见下）
- [ ] `01_hungry`：passed=true，severity=normal，accuracy 命中
- [ ] `02_pain_lowconf`：passed=false（反例），因 confidence=0.51<0.60 且 cry_ratio=0.21<0.30 同时不达标，severity=manual_review
- [ ] `03_fright`：passed=true，severity=urgent_care（命中 pain 之外的临床升级场景），accuracy 命中
- [ ] 任一段 `top_confidence < 0.60` 或 `cry_ratio < 0.30` → 该段 `verify.passed` 必须为 false（反例检查）
- [ ] 报告 `trace` 数组完整记录 4 类 tool call
- [ ] 后端未就绪（health.status != 'ready'）时，**不得**执行 analyze（短路）

> 注：「全部通过」指工作流编排正确、断言逻辑正确，而非每段都要 passed=true。
> `02_pain_lowconf` 段**故意**构造为低置信度+低哭声占比的反例，用来证明验证器能正确判
> `passed=false` 并降级 severity。汇总行会打印 `PASS · 2/3 段通过`（02 为预期失败），
> 只要 01/03 通过且 02 的失败原因命中预期，整体仍判 PASS。

## 非目标

- 不做真实模型训练 / 不重跑 checkpoint
- 不替代医生诊断（报告中必须保留 RED_FLAGS 与「仅供辅助」声明）
