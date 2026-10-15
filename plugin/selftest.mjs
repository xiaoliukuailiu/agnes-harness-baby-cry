/**
 * 插件本地自测桩（SSL 版）：不启动 AGH，用一个假的 Cordis ctx 驱动插件，
 * 验证 registerTool / execute / 返回值格式都正确。
 *
 * 用法（先启动后端服务）：
 *   node agh-plugin-ssl/selftest.mjs [音频绝对路径]
 *
 * 作用：
 *   1. 在没有 AGH Web 界面的情况下排查插件逻辑
 *   2. 产出可直接放进作品的「工具调用记录」
 */

import { babyCrySslPlugin } from './index.mjs'

const AUDIO = process.argv[2] || ''

/** 模拟 AGH 注入的 extension API */
function makeCtx() {
  const tools = new Map()
  const listeners = new Map()
  return {
    ctx: { log: { info: (m) => console.log(`[session] ${m}`) } },
    extension: () => ({
      registerTool: (tool) => tools.set(tool.name, tool),
      on: (event, fn) => listeners.set(event, fn),
    }),
    _tools: tools,
    _listeners: listeners,
  }
}

const ctx = makeCtx()
babyCrySslPlugin.apply(ctx)

console.log('=== 已注册工具 ===')
for (const name of ctx._tools.keys()) console.log(' -', name)

async function main() {
  const health = ctx._tools.get('baby_cry_ssl_health')
  const labels = ctx._tools.get('baby_cry_ssl_labels')
  const analyze = ctx._tools.get('analyze_baby_cry_ssl')

  console.log('\n=== [步骤 1] 调用 baby_cry_ssl_health ===')
  const r1 = await health.execute({})
  console.log(r1.content[0].text)
  if (r1.structured.status !== 'ready') {
    console.log('后端未就绪，终止。')
    process.exit(1)
  }

  console.log('\n=== [步骤 2] 调用 baby_cry_ssl_labels ===')
  const r0 = await labels.execute({})
  console.log(r0.content[0].text)

  if (AUDIO) {
    console.log('\n=== [步骤 3] 调用 analyze_baby_cry_ssl ===')
    console.log('入参:', JSON.stringify({ audio_path: AUDIO, top_k: 6 }))
    const r2 = await analyze.execute({ audio_path: AUDIO, top_k: 6 })
    console.log(r2.content[0].text)
    console.log('\n结构化返回:')
    console.log(JSON.stringify(r2.structured, null, 2))
  } else {
    console.log('\n=== [步骤 3] 未提供音频路径，跳过分析 ===')
  }

  console.log('\n=== [步骤 4] 异常分支：不存在的路径 ===')
  const r3 = await analyze.execute({
    audio_path: 'C:\\nonexistent\\missing.wav',
    top_k: 6,
  })
  console.log(r3.content[0].text)
}

main()
