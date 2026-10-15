/**
 * Baby Cry SSL —— AGH 工具插件（自训练 checkpoint 版）
 *
 * 一个零依赖的 Cordis 对象插件，向 AGH 注册四个工具：
 *   1. analyze_baby_cry_ssl      分析一段婴儿哭声音频，返回 6 类分类结果
 *   2. baby_cry_ssl_health       检查后端推理服务是否就绪（用于结果验证步骤）
 *   3. baby_cry_ssl_labels       返回标签体系、护理建议与就医危险信号字典
 *   4. verify_baby_cry_result    对分析结果做确定性校验（置信度/哭声占比/基频/词表/准确率）
 *
 * 与旧插件 agh-plugin/ 的区别：
 *   - 后端端口 8800（旧插件为 8799）
 *   - 模型为参赛者自训练 Wav2Vec2 checkpoint，6 类标签
 *     （旧插件为第三方 DistilHuBERT，5 类标签）
 *   - 工具名带 _ssl 后缀，可与旧插件共存而不冲突
 *
 * 参数用 TypeBox Schema（手工打 Kind 标记，避免引入运行时依赖）。
 */

const Kind = Symbol.for('TypeBox.Kind')

const BACKEND_URL = process.env.BABY_CRY_SSL_BACKEND_URL || 'http://127.0.0.1:8800'
const REQUEST_TIMEOUT_MS = Number(process.env.BABY_CRY_SSL_TIMEOUT_MS || 120000)

/** 工具元信息：只读、无破坏性、无需人工审批、可重放 */
const toolMeta = {
  isReadOnly: true,
  isDestructive: false,
  isConcurrencySafe: true,
  isOpenWorld: false,
  replay: 'safe',
  costHint: undefined,
  deferLoading: false,
  requiresApproval: 'never',
}

/** analyze_baby_cry_ssl 的参数 Schema */
const analyzeParameters = {
  [Kind]: 'Object',
  type: 'object',
  properties: {
    audio_path: {
      [Kind]: 'String',
      type: 'string',
      description: '待分析音频文件的绝对路径（wav/mp3/flac/ogg，建议 3–10 秒）',
    },
    top_k: {
      [Kind]: 'Integer',
      type: 'integer',
      minimum: 1,
      maximum: 6,
      description: '返回前 k 个候选类别，默认 6（本模型共 6 类）',
    },
  },
  required: ['audio_path'],
  additionalProperties: false,
}

/** 无参数工具的 Schema（空对象） */
const emptyParameters = {
  [Kind]: 'Object',
  type: 'object',
  properties: {},
  additionalProperties: false,
}

/** verify_baby_cry_result 的参数 Schema */
const verifyParameters = {
  [Kind]: 'Object',
  type: 'object',
  properties: {
    audio_path: {
      [Kind]: 'String',
      type: 'string',
      description: '待校验的音频绝对路径；提供时会自动重新分析并校验（推荐，最省事）',
    },
    result: {
      [Kind]: 'Object',
      type: 'object',
      additionalProperties: true,
      description: '可选，直接传入 analyze_baby_cry_ssl 返回的结构化结果（含 status / top_confidence / acoustics / top_label / warnings）',
    },
    expected_label: {
      [Kind]: 'String',
      type: 'string',
      description: '可选，期望类别；提供时追加 accuracy 检查',
    },
  },
  required: [],
  additionalProperties: false,
}

async function callBackend(path, payload, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(`${BACKEND_URL}${path}`, {
      method: payload === undefined ? 'GET' : 'POST',
      headers: payload === undefined ? {} : { 'content-type': 'application/json' },
      body: payload === undefined ? undefined : JSON.stringify(payload),
      signal: controller.signal,
    })
    const text = await res.text()
    let json
    try {
      json = JSON.parse(text)
    } catch {
      json = { status: 'error', error_type: 'bad_response', message: text.slice(0, 500) }
    }
    return { httpStatus: res.status, json }
  } catch (err) {
    return {
      httpStatus: 0,
      json: {
        status: 'error',
        error_type: 'backend_unreachable',
        message:
          `无法连接哭声分析服务 ${BACKEND_URL}：${err?.message || err}。` +
          `请确认已启动：python backend-ssl/server.py`,
      },
    }
  } finally {
    clearTimeout(timer)
  }
}

/** 把结构化结果渲染成人话摘要，便于 LLM 直接引用也便于人工核对 */
function renderSummary(result) {
  if (result.status === 'ok') {
    const dist = (result.distribution || [])
      .slice(0, 3)
      .map((d) => `${d.label_zh}(${d.label}) ${(d.score * 100).toFixed(1)}%`)
      .join('，')
    const a = result.acoustics || {}
    const advice = (result.advice || []).map((x) => `  · ${x}`).join('\n')
    const redFlags = (result.red_flags || []).map((x) => `  · ${x}`).join('\n')
    return [
      `识别结果：${result.top_label_zh}（${result.top_label}），置信度 ${(result.top_confidence * 100).toFixed(1)}%`,
      `候选分布：${dist}`,
      `声学：时长 ${a.duration_sec}s、分析片段 ${a.analyzed_segment_sec}s（候选 ${a.candidate_segments} 段）、` +
        `哭声帧占比 ${(a.cry_segment_ratio * 100).toFixed(0)}%、估算基频 ${a.estimated_f0_hz ?? 'N/A'}Hz`,
      advice ? `护理建议：\n${advice}` : '护理建议：无',
      redFlags ? `就医危险信号：\n${redFlags}` : '就医危险信号：无',
      result.warnings?.length ? `提醒：${result.warnings.join('；')}` : '提醒：无',
    ].join('\n')
  }
  if (result.status === 'silence') {
    return `未检测到有效哭声：${result.message}。建议：${result.hint || '重新录制'}`
  }
  return `分析失败 [${result.error_type}]：${result.message}`
}

// ---------------------------------------------------------------- 确定性校验器
// 与 examples/baby-cry-ssl-workflow/verify.mjs 逻辑一致，作为 AGH 工具内联，
// 便于真实会话中由 Agnes 模型直接调用 verify_baby_cry_result 完成结果验证。

const DEFAULT_THRESHOLDS = {
  minConfidence: 0.6,
  minCryRatio: 0.1,
  f0Low: 150,
  f0High: 1500,
}
const URGENT_LABELS = ['pain']

function verifyResult(result, labels, expectedLabel, thresholds = {}) {
  const t = { ...DEFAULT_THRESHOLDS, ...thresholds }
  const checks = []

  checks.push({
    name: 'backend_status',
    pass: result.status === 'ok' || result.status === 'silence',
    value: result.status,
    detail: result.status === 'error' ? `error_type=${result.error_type} msg=${result.message}` : '',
  })

  if (result.status === 'silence') return summarize(['silence'], checks)
  if (result.status === 'error') return summarize(['error'], checks)

  const conf = result.top_confidence ?? 0
  checks.push({ name: 'confidence', pass: conf >= t.minConfidence, value: conf, threshold: t.minConfidence })

  const cryRatio = result.acoustics?.cry_segment_ratio
  if (cryRatio != null)
    checks.push({ name: 'cry_ratio', pass: cryRatio >= t.minCryRatio, value: cryRatio, threshold: t.minCryRatio })

  const f0 = result.acoustics?.estimated_f0_hz
  if (f0 != null) {
    const inRange = f0 >= t.f0Low && f0 <= t.f0High
    checks.push({
      name: 'acoustic_f0',
      pass: inRange,
      value: f0,
      range: [t.f0Low, t.f0High],
      detail: inRange ? '' : `主频 ${f0}Hz 不在婴儿哭声典型区间，请确认声源`,
    })
  }

  if (labels?.labels) {
    const inVocab = Object.keys(labels.labels).includes(result.top_label)
    checks.push({
      name: 'label_in_vocabulary',
      pass: inVocab,
      value: result.top_label,
      vocabulary: Object.keys(labels.labels),
    })
  }

  if (expectedLabel) {
    const hit = result.top_label === expectedLabel
    checks.push({
      name: 'accuracy',
      pass: hit,
      value: result.top_label,
      expected: expectedLabel,
      detail: hit ? '命中期望类别' : `期望 ${expectedLabel}，实际 ${result.top_label}`,
    })
  }

  const redFlagHit = URGENT_LABELS.includes(result.top_label)
  if (redFlagHit)
    checks.push({
      name: 'red_flag_scenario',
      pass: false,
      value: result.top_label,
      detail: '该类别关联临床红线信号，建议升级护理响应',
    })

  return summarize([result.top_label], checks, redFlagHit)
}

function summarize(labelsHit, checks, redFlagHit = false) {
  const hardFails = checks.filter((c) => !c.pass && c.name !== 'red_flag_scenario')
  const backendOk = checks.find((c) => c.name === 'backend_status')?.pass === true

  let severity
  if (!backendOk) severity = 'error'
  else if (redFlagHit) severity = 'urgent_care'
  else if (hardFails.some((c) => c.name === 'confidence') || hardFails.some((c) => c.name === 'cry_ratio'))
    severity = 'manual_review'
  else if (hardFails.some((c) => c.name === 'acoustic_f0')) severity = 'non_baby_source'
  else if (hardFails.length === 0) severity = 'normal'
  else severity = 'manual_review'

  const passed = backendOk && hardFails.length === 0
  return {
    passed,
    labels_hit: labelsHit,
    severity,
    red_flag_hit: redFlagHit,
    checks,
    failed_names: hardFails.map((c) => c.name),
  }
}

function renderVerifySummary(v) {
  const lines = v.checks
    .map((c) => {
      const mark = c.pass ? '✅' : '❌'
      const extra =
        c.name === 'accuracy'
          ? ` (期望 ${c.expected})`
          : c.threshold != null
            ? ` (≥${c.threshold})`
            : c.range
              ? ` (∈[${c.range[0]},${c.range[1]}])`
              : ''
      return `  ${mark} ${c.name}${extra}${c.detail ? ' · ' + c.detail : ''}`
    })
    .join('\n')
  const head = `验证结果 [${v.passed ? 'PASS' : 'FAIL'}] severity=${v.severity} 命中类别=${(v.labels_hit || []).join('/')}`
  return [head, lines, `未通过检查：${v.failed_names.join(', ') || '无'}`].join('\n')
}

export const babyCrySslPlugin = {
  inject: ['extension'],
  apply(ctx) {
    const agnes = ctx.extension()

    // ---- 工具 1：哭声分析 ------------------------------------------------
    agnes.registerTool({
      name: 'analyze_baby_cry_ssl',
      description:
        '分析一段婴儿哭声音频，推断哭闹原因（疼痛/难受不适/受惊/饥饿/不安/生气，共 6 类），' +
        '返回 top-k 分类结果、置信度分布、声学特征、护理建议、就医危险信号与异常提醒。' +
        '输入为音频文件绝对路径。分析完成后应继续调用 verify_baby_cry_result（传入同一 audio_path）对结果做一次确定性校验，' +
        '并结合 baby_cry_ssl_labels 解释护理建议。仅用于辅助理解，不构成医疗诊断。',
      parameters: analyzeParameters,
      meta: toolMeta,
      async execute({ audio_path, top_k }) {
        const { httpStatus, json } = await callBackend('/analyze', {
          audio_path,
          top_k: top_k ?? 6,
          source: 'agh',
        })
        const structured = { httpStatus, backend: BACKEND_URL, ...json }
        return {
          content: [{ type: 'text', text: renderSummary(json) }],
          structured,
        }
      },
    })

    // ---- 工具 2：后端健康检查 --------------------------------------------
    agnes.registerTool({
      name: 'baby_cry_ssl_health',
      description:
        '检查本地 SSL 哭声分析服务是否就绪，返回 checkpoint 路径、标签集合与加载错误（如有）。' +
        '执行哭声分析失败时应先调用它排障。',
      parameters: emptyParameters,
      meta: toolMeta,
      async execute() {
        const { httpStatus, json } = await callBackend('/health')
        const structured = { httpStatus, backend: BACKEND_URL, ...json }
        const text =
          json.status === 'ready'
            ? `哭声服务就绪：${json.model_base} @ ${json.num_labels} 类，` +
              `标签 ${(json.labels || []).join('/')}，采样率 ${json.native_hz}Hz，窗口 ${json.window_sec}s`
            : `哭声服务异常：${json.model_error || json.message || '未知原因'}`
        return { content: [{ type: 'text', text }], structured }
      },
    })

    // ---- 工具 3：标签、护理建议与就医危险信号字典 ------------------------
    agnes.registerTool({
      name: 'baby_cry_ssl_labels',
      description:
        '返回本模型的 6 类标签中英文对照、每类护理建议与需要就医的危险信号清单。' +
        '用于向用户解释分类结果时提供循证护理指引。',
      parameters: emptyParameters,
      meta: toolMeta,
      async execute() {
        const { httpStatus, json } = await callBackend('/labels')
        const structured = { httpStatus, backend: BACKEND_URL, ...json }
        const lines = Object.entries(json.labels || {}).map(([en, zh]) => `  · ${en} = ${zh}`)
        const advice = Object.entries(json.advice || {}).map(
          ([label, items]) => `  · ${label}：${(items || []).join('；')}`,
        )
        const flags = (json.red_flags || []).map((f) => `  · ${f}`)
        const text = [
          `标签体系（${lines.length} 类）：`,
          ...lines,
          '',
          `护理建议（${advice.length} 类）：`,
          ...advice,
          '',
          `就医危险信号（${flags.length} 条）：`,
          ...flags,
        ].join('\n')
        return { content: [{ type: 'text', text }], structured }
      },
    })

    // ---- 工具 4：确定性结果校验 ------------------------------------------
    agnes.registerTool({
      name: 'verify_baby_cry_result',
      description:
        '对婴儿哭声分析做确定性校验（后端状态 / 置信度 / 哭声占比 / 基频 / 词表 / 准确率），' +
        '返回 PASS/FAIL、severity 分级与逐项检查明细。客观、可回归，不依赖主观判断。' +
        '传入 audio_path 会自动重新分析并校验；也可直接传入 analyze 的结构化 result。',
      parameters: verifyParameters,
      meta: toolMeta,
      async execute({ audio_path, result, expected_label }) {
        let r = result && typeof result === 'object' && !Array.isArray(result) ? result : null
        if (!r && audio_path) {
          const { json } = await callBackend('/analyze', { audio_path, top_k: 6, source: 'agh' })
          r = json && typeof json === 'object' ? json : null
        }
        if (!r) {
          return {
            content: [{ type: 'text', text: '校验失败：请提供 audio_path（自动重新分析）或 result（结构化结果）之一。' }],
            structured: { passed: false, severity: 'error', labels_hit: [], checks: [], failed_names: [] },
          }
        }
        let labels = null
        try {
          const { json } = await callBackend('/labels')
          if (json?.labels) labels = json
        } catch {
          labels = null
        }
        const v = verifyResult(r, labels, expected_label || undefined)
        return { content: [{ type: 'text', text: renderVerifySummary(v) }], structured: v }
      },
    })

    agnes.on('session_start', () => {
      agnes.ctx.log.info(`baby-cry-ssl tools ready (backend=${BACKEND_URL})`)
    })
  },
}
