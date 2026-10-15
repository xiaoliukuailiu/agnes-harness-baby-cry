# 婴儿哭声分诊智能体 · Baby Cry Triage

基于**本地自训练 Wav2Vec2 模型**的婴儿哭声识别与健康分诊智能体，以 Agnes Harness（AGH）为运行底座。
把家长听不懂的哭声，翻译成「可能的原因 + 置信度 + 可执行的护理动作 + 何时必须就医」。

> 本包为参赛提交用**可复现代码包**。模型权重另存于 GitHub（见下），不随包分发。

---

## 目录结构

```
├── backend-ssl/             # 推理服务（FastAPI，127.0.0.1:8800）
│   ├── server.py
│   ├── requirements.txt
│   └── feature_extractor/   # 本地 Wav2Vec2FeatureExtractor 配置（离线）
├── plugin/                  # AGH 工具插件（4 工具：health / analyze / verify / labels）
│   ├── index.mjs
│   ├── package.json
│   └── selftest.mjs         # 脱离 AGH 的本地自测
├── workflow/                # 可复现工作流 + 测试样例（评分核心）
│   ├── run.mjs              # 一键回归（offline mock / --live 真实）
│   ├── verify.mjs           # 确定性校验器
│   ├── workflow.md          # 三步流程（规划 → 调用 → 验证）
│   ├── task.md              # AGH 任务描述
│   └── fixture/             # 3 段 mock 返回 + 结构化报告
├── demo/
│   └── index.html           # Web demo（上传/录音 → 报告 → 导出）
├── training/                # 模型训练代码（数据预处理 + 微调）
│   ├── src/                 # 训练/评估脚本
│   ├── data/                # 平衡后数据清单（6 类 × 195）
│   └── results/             # 训练结果（UAR 96.99% 等）
├── start_backend_ssl.bat    # 一键起 8800 后端
├── start_demo.bat           # 一键起 后端 + Web demo
└── README.md
```

## 模型与数据

| 项 | 值 |
|---|---|
| 基座模型 | `facebook/wav2vec2-base` |
| 微调语料 | Baby Chillanto（参赛者自行处理） |
| 分类 | 6 类：`pain`疼痛 · `uncomfortable`难受 · `fright`受惊 · `hungry`饥饿 · `disturbance`不安 · `angry`生气 |
| 训练代码 | `training/`（本包内含：微调脚本 + 数据清单 + 训练结果） |
| checkpoint | 见下方「下载权重」（托管于 [`xiaoliukuailiu/Babycry-SSL`](https://github.com/xiaoliukuailiu/Babycry-SSL)，Git LFS） |

## 快速复现（3 步）

### 1. 装依赖

```bash
pip install -r backend-ssl/requirements.txt
```

### 2. 下载权重

训练代码已在本包的 `training/` 目录。模型权重（checkpoint）因体积较大，托管在 GitHub 仓库 [`xiaoliukuailiu/Babycry-SSL`](https://github.com/xiaoliukuailiu/Babycry-SSL)（`checkpoint/` 目录，Git LFS）：

```bash
git lfs install && git clone https://github.com/xiaoliukuailiu/Babycry-SSL.git
# → checkpoint 在 Babycry-SSL/checkpoint/
```

下载后把 `checkpoint/` 目录放到 `backend-ssl/checkpoint-1500/`（默认位置，直接可跑），
或把它的绝对路径写入环境变量 `BABY_CRY_CHECKPOINT` 再启动服务。

### 3. 起服务 + 跑工作流

```bash
# ① 一键起后端 + Web demo（浏览器打开 http://127.0.0.1:5500/）
start_demo.bat

# ② 工作流回归（评分核心；离线可跑，无需权重/AGH）
node workflow/run.mjs
#   期望输出：PASS · 2/3 段按预期通过 · 1 段为预期反例
```

## 测试样例（正常 / 边界 / 失败）

`workflow/run.mjs` 内置 3 段 fixture，覆盖三类测试样例：

| 样例 | 类型 | 输入特征 | 期望结果 |
|---|---|---|---|
| `01_hungry` | 正常 | 置信度 0.82 · 哭声占比 0.72 · F0 412Hz | PASS · severity=normal |
| `02_pain_lowconf` | 失败（反例） | 置信度 0.51 · 占比 0.21 | FAIL · severity=urgent_care（预期降级） |
| `03_fright_urgent` | 边界 | 占比 0.88 · F0 320Hz（区间内偏低） | PASS · severity=normal |

确定性校验器（`verify.mjs`）做 6 项客观检查：后端状态 / 置信度 / 哭声占比 / 基频 / 词表 / 准确率，
低置信度、非婴儿声源、临床红线都能被**显式识别并降级**，不硬给结论。

## AGH 集成（可选）

把 `plugin/` 安装进 Agnes Harness（Agnes Harness 是独立开源底座，见其官方仓库）：

```bash
agh install file:./plugin    # 再 trust + enable
```

注册 4 个工具：`baby_cry_ssl_health` / `analyze_baby_cry_ssl` / `verify_baby_cry_result` / `baby_cry_ssl_labels`。
Agnes 模型按「任务规划 → 能力调用 → 反馈处理与结果验证」三步连续调用这些工具。

## 免责声明

本项目为 AI 声学模型的辅助理解工具，**不构成医疗诊断**，不能替代儿科医生的专业判断。婴儿出现不适请及时就医。
