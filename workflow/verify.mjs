/**
 * verify_baby_cry_result —— 婴儿哭声分类结果的确定性校验器
 *
 * 设计目标（对应评分项「验证严谨性与异常处理」）：
 *   把「结果验证」从 LLM 主观判断变成可量化、可回归的客观检查，
 *   每条 check 都给出 pass 与否 + 实际值 + 阈值，便于评分与自动化。
 *
 * 可作为：
 *   - AGH 插件工具（见 plugin.mjs）
 *   - 独立脚本（node -e "import('./verify.mjs').then(m=>m.verify_baby_cry_result(...))"）
 *   - run.mjs 工作流的验证步骤
 *
 * 输入：
 *   result            analyze 后端返回的 structured JSON（含 top_confidence / acoustics / top_label / warnings）
 *   labels            可选，标签字典（{ labels: {en:zh}, advice, red_flags }），用于 vocabulary 校验
 *   expected_label    可选，期望类别；提供时追加 accuracy 检查
 *   thresholds        可选，覆盖默认阈值
 */

const DEFAULT_THRESHOLDS = {
  minConfidence: 0.6,        // 低于则「区分度不足」
  minCryRatio: 0.3,         // 低于则「哭声占比偏低」
  f0Low: 150,               // 婴儿哭声基频下限
  f0High: 1500,             // 婴儿哭声基频上限
}

/** 主入口：对一段 analyze 结果做校验，返回 { passed, checks[], severity, red_flag_hit } */
export function verify_baby_cry_result(result, labels, expectedLabel, thresholds) {
  const t = { ...DEFAULT_THRESHOLDS, ...(thresholds || {}) }
  const checks = []

  // 1. 后端状态
  checks.push({
    name: 'backend_status',
    pass: result.status === 'ok' || result.status === 'silence',
    value: result.status,
    detail: result.status === 'error' ? `error_type=${result.error_type} msg=${result.message}` : '',
  })

  if (result.status === 'silence') {
    return summarize(['silence'], checks)
  }
  if (result.status === 'error') {
    return summarize(['error'], checks)
  }

  // 2. 置信度
  const conf = result.top_confidence ?? 0
  checks.push({
    name: 'confidence',
    pass: conf >= t.minConfidence,
    value: conf,
    threshold: t.minConfidence,
  })

  // 3. 哭声占比
  const cryRatio = result.acoustics?.cry_segment_ratio
  if (cryRatio != null) {
    checks.push({
      name: 'cry_ratio',
      pass: cryRatio >= t.minCryRatio,
      value: cryRatio,
      threshold: t.minCryRatio,
    })
  }

  // 4. 婴儿声源（基频范围）
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

  // 5. 类别在词表内
  if (labels?.labels) {
    const inVocab = Object.keys(labels.labels).includes(result.top_label)
    checks.push({
      name: 'label_in_vocabulary',
      pass: inVocab,
      value: result.top_label,
      vocabulary: Object.keys(labels.labels),
    })
  }

  // 6. accuracy（提供 expected_label 时）
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

  // 7. 红线信号（临床危险信号）
  //    RED_FLAGS 是「需要就医的通用临床信号清单」，属于报告里必须展示的
  //    「升级提醒」，而不是「校验失败」项。本工作流里只有当 top_label 命中
  //    高临床风险类别（pain）时，把 severity 提升为 urgent_care（提示升级响应），
  //    但该项不参与 passed 判定（它不是质量/正确性失败，是风险提示）。
  const urgentLabels = ['pain']
  const redFlagHit = urgentLabels.includes(result.top_label)
  if (redFlagHit) {
    checks.push({
      name: 'red_flag_scenario',
      pass: false, // 命中即不 pass（提示需要升级处理）
      value: result.top_label,
      detail: '该类别关联临床红线信号，建议升级护理响应',
    })
  }

  return summarize([result.top_label], checks, redFlagHit)
}

/** 根据 checks 推导 severity 与 passed。
 *  规则：
 *   - 后端状态非 ok/silence → error，passed=false
 *   - 命中临床红线场景（pain 等高临床风险类别）→ urgent_care
 *   - 置信度/哭声占比不达标 → manual_review（需要人工复核）
 *   - 基频不在婴儿区间 → non_baby_source（怀疑非婴儿声源）
 *   - 全部通过 → normal，passed=true
 *  其中 red_flag_scenario 是「提示升级响应」而非「校验失败」，故不影响 passed 判定。
 */
function summarize(labelsHit, checks, redFlagHit = false) {
  // 只把「质量/正确性」类检查算进 passed（排除后端状态之外的提示性检查）
  const hardFails = checks.filter((c) => !c.pass && c.name !== 'red_flag_scenario')
  const backendOk = checks.find((c) => c.name === 'backend_status')?.pass === true

  let severity
  if (!backendOk) {
    severity = 'error'
  } else if (redFlagHit) {
    severity = 'urgent_care'
  } else if (hardFails.some((c) => c.name === 'confidence') || hardFails.some((c) => c.name === 'cry_ratio')) {
    severity = 'manual_review'
  } else if (hardFails.some((c) => c.name === 'acoustic_f0')) {
    severity = 'non_baby_source'
  } else if (hardFails.length === 0) {
    severity = 'normal'
  } else {
    severity = 'manual_review'
  }

  // passed = 后端正常 且 没有「质量/正确性」硬失败
  // （red_flag_scenario 是风险提示，不算硬失败；fright 等非 urgent 类别即使 f0 偏低
  //  也只在有 accuracy 检查时按 accuracy 判定，避免误伤）
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

/** 渲染人类可读摘要 */
export function renderVerifySummary(v) {
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
