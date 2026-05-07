"""
LLM 报告生成服务 - 通过 DeepSeek API 生成专业分析报告。

环境变量：
    DEEPSEEK_API_KEY: API 密钥
    DEEPSEEK_BASE_URL: API 地址（默认 https://api.deepseek.com）
    DEEPSEEK_MODEL: 模型名称（默认 deepseek-chat）
"""

import json
import logging
import os
from datetime import datetime

import requests
from django.conf import settings

from apps.detection.models import DetectionResult, DetectionTask
from .models import Report
from .services import _compute_analysis, METHOD_EXPLANATIONS, RISK_TYPE_LABELS, SUGGESTION_LABELS

logger = logging.getLogger('dpds.llm_report')

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')


def _build_prompt(task, analysis: dict, top_samples: list) -> str:
    """构建发送给 LLM 的 prompt。"""
    dataset = task.dataset
    a = analysis

    detector_summary = []
    for d in a['per_detector']:
        detector_summary.append(
            f"- {d['display_name']}（{d['type_label']}）：发现 {d['count']} 个可疑样本，"
            f"平均置信度 {d['avg_conf']*100:.1f}%，最高 {d['max_conf']*100:.1f}%"
        )

    sample_lines = []
    for r in top_samples[:10]:
        sample_lines.append(
            f"  - 样本 {r.sample_id}：{RISK_TYPE_LABELS.get(r.risk_type, r.risk_type)}，"
            f"置信度 {r.confidence*100:.1f}%，检测器 {r.detector_name}，"
            f"建议 {SUGGESTION_LABELS.get(r.suggestion, r.suggestion)}"
        )

    prompt = f"""你是一位资深的数据安全分析师，请根据以下数据投毒检测结果撰写一份专业的中文分析报告。

## 基本信息
- 数据集：{dataset.dataset_name}
- 样本总数：{a['total_samples']:,}
- 检测时间：{task.finished_at.strftime('%Y-%m-%d %H:%M:%S') if task.finished_at else '未知'}
- 任务编号：{task.task_no}

## 检测结果概览
- 综合风险分数：{a['risk_score']*100:.2f}%
- 风险等级：{a['risk_level_text']}
- 可疑样本数：{a['total_suspicious']}（占 {a['pct']:.2f}%）
- 多检测器交叉命中：{a['cross_overlap']} 个样本

## 各检测器结果
{chr(10).join(detector_summary)}

## 置信度分布
- 高置信度（≥80%）：{a['conf_stats']['high']} 个
- 中置信度（50%-80%）：{a['conf_stats']['medium']} 个
- 低置信度（<50%）：{a['conf_stats']['low']} 个

## Top 可疑样本
{chr(10).join(sample_lines)}

## 风险分解
{json.dumps(a['risk_decomposition'], ensure_ascii=False, indent=2)}

请输出以下内容（纯文本，不要 markdown 代码块）：
1. 【风险概述】：2-3 句话总结数据集安全状况
2. 【深度分析】：分析投毒攻击可能的类型、手法和影响范围
3. 【关键发现】：列出最重要的 3-5 个发现
4. 【处置建议】：给出具体、可操作的处置步骤
5. 【后续防护】：建议的长期防护措施

请确保分析专业、客观，建议具体可执行。"""

    return prompt


def generate_llm_report(task_id: str, user=None) -> Report:
    """调用 DeepSeek API 生成 LLM 分析报告。"""
    if not DEEPSEEK_API_KEY:
        raise ValueError('未配置 DEEPSEEK_API_KEY 环境变量，无法生成 LLM 报告')

    task = DetectionTask.objects.select_related('dataset', 'created_by').get(id=task_id)
    analysis = _compute_analysis(task)
    top_samples = list(DetectionResult.objects.filter(task=task).order_by('-confidence')[:20])

    prompt = _build_prompt(task, analysis, top_samples)

    logger.info('调用 DeepSeek API 生成报告: task=%s, model=%s', task.task_no, DEEPSEEK_MODEL)

    try:
        resp = requests.post(
            f'{DEEPSEEK_BASE_URL}/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'model': DEEPSEEK_MODEL,
                'messages': [
                    {'role': 'system', 'content': '你是一位资深的数据安全分析师，擅长分析数据投毒攻击并撰写专业报告。'},
                    {'role': 'user', 'content': prompt},
                ],
                'temperature': 0.3,
                'max_tokens': 4000,
            },
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()
        llm_content = result['choices'][0]['message']['content']
        logger.info('DeepSeek API 调用成功: task=%s, tokens=%s', task.task_no, result.get('usage', {}))
    except requests.RequestException as e:
        logger.exception('DeepSeek API 调用失败: task=%s', task.task_no)
        raise RuntimeError(f'LLM API 调用失败: {e}')

    # 构建标题和摘要
    title = f'LLM 深度分析报告 - {task.task_no}'
    summary = (
        f'数据集「{task.dataset.dataset_name}」综合风险分数 {analysis["risk_score"]*100:.1f}%，'
        f'等级 {analysis["risk_level_text"]}，发现 {analysis["total_suspicious"]} 个可疑样本。'
    )

    report = Report.objects.create(
        task=task,
        title=title,
        report_type='html',
        summary=summary,
        llm_content=llm_content,
        analysis_json=analysis,
        created_by=user,
    )

    # 同时生成 HTML 文件
    html_content = _build_llm_report_html(task, analysis, llm_content, top_samples)
    out_dir = settings.MEDIA_ROOT / 'reports'
    os.makedirs(out_dir, exist_ok=True)
    import uuid as _uuid
    out_path = str(out_dir / f'{_uuid.uuid4().hex}.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    report.file_path = out_path
    report.save(update_fields=['file_path'])

    return report


def _build_llm_report_html(task, analysis: dict, llm_content: str, top_samples: list) -> str:
    """将 LLM 输出嵌入 HTML 报告模板。"""
    a = analysis
    risk_cls = a['risk_level']
    color_map = {'high': '#d32f2f', 'medium': '#f57c00', 'low': '#388e3c'}
    color = color_map.get(risk_cls, '#666')
    icon_map = {'high': '⚠️', 'medium': '⚡', 'low': '✅'}
    icon = icon_map.get(risk_cls, '')

    # 样本表
    sample_rows = ''
    for r in top_samples:
        sample_rows += (
            f'<tr><td>{r.sample_id}</td>'
            f'<td>{RISK_TYPE_LABELS.get(r.risk_type, r.risk_type)}</td>'
            f'<td>{r.confidence*100:.1f}%</td>'
            f'<td>{r.detector_name}</td>'
            f'<td>{SUGGESTION_LABELS.get(r.suggestion, r.suggestion)}</td></tr>\n'
        )

    # LLM 内容格式化（简单段落处理）
    llm_html = ''
    for para in llm_content.split('\n'):
        para = para.strip()
        if not para:
            continue
        if para.startswith('【') and '】' in para:
            title_end = para.index('】') + 1
            llm_html += f'<h3 style="color:#1a73e8;margin-top:24px;">{para[:title_end]}</h3>'
            rest = para[title_end:].strip()
            if rest:
                llm_html += f'<p>{rest}</p>'
        else:
            llm_html += f'<p style="margin:8px 0;line-height:1.8;">{para}</p>'

    html = f'''<!DOCTYPE html>
<html lang="zh-hans">
<head>
<meta charset="utf-8">
<title>DPDS LLM 分析报告 - {task.task_no}</title>
<style>
  body {{ font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif; margin: 40px; color: #333; line-height: 1.6; }}
  h1 {{ color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 12px; }}
  h2 {{ color: #555; border-bottom: 1px solid #ddd; padding-bottom: 8px; margin-top: 32px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 16px 0; }}
  .stat-card {{ background: #f8f9fa; border-radius: 8px; padding: 16px; text-align: center; }}
  .stat-value {{ font-size: 28px; font-weight: bold; color: #1a73e8; }}
  .stat-label {{ font-size: 13px; color: #666; margin-top: 4px; }}
  .conclusion-box {{ background: #f8f9fa; border-radius: 8px; padding: 20px; margin: 16px 0; border-left: 5px solid {color}; }}
  .llm-section {{ background: #fafbfc; border-radius: 8px; padding: 24px; margin: 16px 0; border: 1px solid #e8ecf0; }}
  .meta {{ color: #999; font-size: 12px; margin-top: 40px; }}
</style>
</head>
<body>
<h1>数据投毒检测 - LLM 深度分析报告</h1>

<h2>1. 概览</h2>
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-value">{a["total_suspicious"]}</div>
    <div class="stat-label">可疑样本数</div>
  </div>
  <div class="stat-card">
    <div class="stat-value" style="color:{color};">{a["risk_score"]*100:.1f}%</div>
    <div class="stat-label">综合风险分数</div>
  </div>
  <div class="stat-card">
    <div class="stat-value" style="color:{color};font-size:20px;">{a["risk_level_text"]}</div>
    <div class="stat-label">风险等级</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{a["total_samples"]:,}</div>
    <div class="stat-label">数据集样本总数</div>
  </div>
</div>

<div class="conclusion-box">
  <div style="font-size:18px;font-weight:bold;color:{color};margin-bottom:8px;">
    {icon} {a["risk_level_text"]} — 检测结论
  </div>
  <p style="margin:0;">该数据集存在{"高度" if a["risk_level"]=="high" else "中等" if a["risk_level"]=="medium" else "较低"}数据投毒风险
  （综合风险分数 {a["risk_score"]*100:.1f}%），共发现 {a["total_suspicious"]} 个可疑样本
  （约占 {a["pct"]:.1f}%）。</p>
</div>

<h2>2. 数据集信息</h2>
<table>
  <tr><th>数据集名称</th><td>{task.dataset.dataset_name}</td></tr>
  <tr><th>样本总数</th><td>{a["total_samples"]:,}</td></tr>
  <tr><th>标签字段</th><td>{task.dataset.label_field or "-"}</td></tr>
  <tr><th>任务编号</th><td>{task.task_no}</td></tr>
  <tr><th>完成时间</th><td>{task.finished_at.strftime("%Y-%m-%d %H:%M:%S") if task.finished_at else "-"}</td></tr>
</table>

<h2>3. LLM 专业分析</h2>
<div class="llm-section">
{llm_html}
</div>

<h2>4. 高风险样本（Top 20）</h2>
<table>
  <tr><th>样本 ID</th><th>风险类型</th><th>置信度</th><th>检测器</th><th>处理建议</th></tr>
  {sample_rows}
</table>

<p class="meta">
  报告生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | DPDS 数据投毒检测系统<br>
  本报告由 LLM（{DEEPSEEK_MODEL}）辅助生成，建议结合人工审查做出最终决策。
</p>
</body>
</html>'''
    return html
