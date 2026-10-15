#!/usr/bin/env node
/**
 * run.mjs —— AGH 智能体工作流可复现脚本（婴儿哭声 SSL 分类）
 *
 * 模拟 AGH 对 4 个工具（baby_cry_ssl_health / analyze_baby_cry_ssl / verify_baby_cry_result / baby_cry_ssl_labels）
 * 的连续调度，对应评分项：
 *   ① 任务规划  ② 能力调用  ③ 反馈处理 + 结果验证
 *
 * 运行：
 *   node run.mjs            # offline：mock 后端 + fixture（3 段），不碰真实模型
 *   node run.mjs --live     # live：直连 8800 真实后端（需已启动 + CORS 放行）
 *   node run.mjs --live --audio "C:\path\to\cry.wav"  # 指定单段真实音频
 *
 * 退出码：0 = 全部断言通过；1 = 有失败。
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { verify_baby_cry_result, renderVerifySummary } from './verify.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))
const MOCK = JSON.parse(readFileSync(join(__dirname, 'fixture', 'mock.json'), 'utf8'))
const BACKEND = process.env.BABY_CRY_SSL_BACKEND_URL || 'http://127.0.0.1:8800'
const LIVE = process.argv.includes('--live')
const audioArg = (process.argv.find((a, i) => process.argv[i - 1] === '--audio')) || null

// =================================================================
// 模拟 AGH 对 4 个工具的「执行」——offline 走 mock，live 走 fetch
// =================================================================
const tools = {
  async baby_cry_ssl_health() {
    if (LIVE) {
      const r = await fetch(`${BACKEND}/health`)
      return { httpStatus: r.status, json: await r.json() }
    }
    return { httpStatus: 200, json: MOCK.health }
  },

  async analyze_baby_cry_ssl({ audio_path, top_k = 6 }) {
    if (LIVE) {
      const r = await fetch(`${BACKEND}/analyze`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ audio_path, top_k }),
      })
      return { httpStatus: r.status, json: await r.json() }
    }
    // offline：按 audio_path 或 clip id 匹配 fixture
    const clip = MOCK.clips.find((c) => c.audio_path === audio_path) || MOCK.clips.find((c) => c.id === audio_path)
    if (!clip) {
      return { httpStatus: 404, json: { status: 'error', error_type: 'fixture_missing', message: `无对应 fixture：${audio_path}` } }
    }
    return { httpStatus: 200, json: clip.analyze }
  },

  async baby_cry_ssl_labels() {
    if (LIVE) {
      const r = await fetch(`${BACKEND}/labels`)
      return { httpStatus: r.status, json: await r.json() }
    }
    return { httpStatus: 200, json: MOCK.labels }
  },
}

// =================================================================
// 工作流编排（对应 workflow.md 的 3 步）
// =================================================================
function plan(healthOk) {
  if (!healthOk) return { plan: ['health'], shortCircuit: true }
  return { plan: ['health', 'analyze', 'verify', 'labels'], shortCircuit: false }
}
const FULL_PLAN = ['health', 'analyze', 'verify', 'labels']

async function runClip(clip, labels, trace) {
  const { audio_path, expected_label } = clip

  // ② 能力调用：analyze
  const analyzeRes = await tools.analyze_baby_cry_ssl({ audio_path, top_k: 6 })
  const result = { ...analyzeRes.json, backend: LIVE ? BACKEND : 'mock' }
  trace.push(['analyze_baby_cry_ssl', { audio_path, top_k: 6 }, summarizeForTrace(result)])

  // ③ 反馈处理 + 结果验证
  const verification = verify_baby_cry_result(result, labels, expected_label)
  trace.push(['verify_baby_cry_result', { top_label: result.top_label, expected: expected_label || null }, verification])

  // 合并报告
  const advice = labels?.advice?.[result.top_label] || result.advice || []
  const report = {
    audio: audio_path,
    mode: LIVE ? 'live' : 'offline',
    plan: FULL_PLAN,
    analyze: {
      status: result.status,
      top_label: result.top_label,
      top_label_zh: result.top_label_zh,
      top_confidence: result.top_confidence,
      distribution: result.distribution,
      acoustics: result.acoustics,
      warnings: result.warnings || [],
    },
    verify: {
      passed: verification.passed,
      severity: verification.severity,
      red_flag_hit: verification.red_flag_hit,
      failed_names: verification.failed_names,
      checks: verification.checks,
    },
    advice,
    red_flags: result.red_flags || labels?.red_flags || [],
    severity: verification.severity,
    trace,
    disclaimer: '本结果仅用于辅助理解婴儿可能的不适信号，不能替代医生诊断。',
  }

  // 离线模式落盘
  if (!LIVE) {
    const out = join(__dirname, 'fixture', 'result', `${clip.id}.json`)
    writeFileSync(out, JSON.stringify(report, null, 2))
  }
  return { clip, report, verification, analyzeRes }
}

function summarizeForTrace(result) {
  if (result.status === 'ok')
    return `top=${result.top_label}(${(result.top_confidence * 100).toFixed(0)}%) warnings=${(result.warnings || []).length}`
  if (result.status === 'silence') return 'silence（未检测到哭声）'
  return `error[${result.error_type}] ${result.message}`
}

// =================================================================
// 主流程
// =================================================================
async function main() {
  console.log('=== AGH 婴儿哭声 SSL 工作流 ===')
  console.log(`模式：${LIVE ? 'LIVE（真实 8800）' : 'OFFLINE（mock fixture）'}   后端：${BACKEND}`)
  console.log('')

  const trace = []
  const labels = (await tools.baby_cry_ssl_labels()).json
  trace.push(['baby_cry_ssl_labels', {}, `已加载 ${Object.keys(labels.labels || {}).length} 类标签`])

  // ① 任务规划 + 前置探测
  const health = await tools.baby_cry_ssl_health()
  trace.push(['baby_cry_ssl_health', {}, health.json.status === 'ready' ? `ready · ${health.json.model_base}` : `异常：${health.json.model_error || health.json.message}`])

  const p = plan(health.json.status === 'ready')
  if (p.shortCircuit) {
    console.error(`❌ 后端未就绪，按规划短路，不执行 analyze。\n${JSON.stringify(health.json, null, 2)}`)
    process.exit(1)
  }

  // ②③ 逐段执行
  // 02_pain_lowconf 是故意构造的「反例」段——低置信度+低哭声占比，
  // 期望 verify.passed=false 且 severity=urgent_care/manual_review。
  // 它的「失败」符合预期，不应计为工作流挂。
  const expectedFail = new Set(MOCK.clips.filter((c) => c.expect_fail).map((c) => c.id))
  let passed = 0
  let unexpectedFail = 0
  const clips = LIVE
    ? audioArg
      ? [{ id: 'live1', audio_path: audioArg, expected_label: null, analyze: null }]
      : []
    : MOCK.clips

  for (const clip of clips) {
    const { report, verification } = await runClip(clip, labels, trace)
    const isExpected = expectedFail.has(clip.id)
    const mark = verification.passed ? '✅' : isExpected ? '⚠(预期反例)' : '❌'
    console.log(`${mark} ${clip.id}  →  ${report.analyze.top_label_zh}(${report.analyze.top_confidence?.toFixed(2) ?? 'N/A'})  severity=${verification.severity}  校验=${verification.passed ? 'PASS' : 'FAIL'}`)
    if (verification.passed) {
      passed++
    } else if (isExpected) {
      const hit = verification.failed_names.includes('confidence') || verification.failed_names.includes('cry_ratio') || verification.severity === 'urgent_care'
      if (!hit) { unexpectedFail++; console.log(`   ↳ 反例未命中预期失败原因：${verification.failed_names.join(',')}`) }
    } else {
      unexpectedFail++
    }
  }

  if (LIVE && !audioArg) {
    console.log('（--live 未指定 --audio，跳过批量；可用 --audio 传单段真实音频）')
    process.exit(0)
  }

  // 汇总：只有「意外失败」才让工作流挂；预期反例不算挂
  const total = clips.length
  const allPassed = unexpectedFail === 0
  console.log('')
  console.log(allPassed
    ? `PASS · ${passed}/${total} 段按预期通过 · ${expectedFail.size} 段为预期反例 · 工具调用 ${trace.length} 次`
    : `FAIL · ${passed}/${total} 段通过 · ${unexpectedFail} 段意外失败`)

  if (!LIVE) {
    console.log('报告已写入 fixture/result/*.json')
  }
  process.exit(allPassed ? 0 : 1)
}

main().catch((e) => {
  console.error('工作流异常：', e)
  process.exit(1)
})
