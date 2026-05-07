import logging
import math
import os
import uuid
from collections import defaultdict
from datetime import datetime

from django.conf import settings

from apps.detection.models import AlgorithmConfig, DetectionResult, DetectionTask
from apps.defense.models import CleanResult
from .models import Report

logger = logging.getLogger('dpds.reports')

# ── 方法说明 ────────────────────────────────────────────────────────────────
METHOD_EXPLANATIONS = {
    'cleanlab': '通过训练逻辑回归分类器，对比每个样本的真实标签与模型预测标签的置信度差异来识别误标样本。',
    'isolation_forest': '孤立森林通过随机分割特征空间来孤立样本，需要分割次数越少的样本越可能是异常点。',
    'lof': '局部离群因子通过比较样本与邻域密度来判断异常，邻域密度远低于邻居的样本为异常。',
    'ks_drift': 'KS 检验逐列比较数据集与基准数据集的数值分布差异，KS 统计量越大漂移越显著。',
    'mmd_drift': '最大均值差异在高维特征空间中比较两个数据集的整体分布差异。',
    'spectral_signature': '对特征矩阵做 SVD 分解，识别在主成分方向上投影异常大的样本。',
    'activation_clustering': '激活聚类对模型中间层激活值进行聚类，后门样本会形成独立小簇。',
    'influence': '影响函数计算每个样本对模型损失的影响，投毒样本有异常大的负面影响。',
    'onion': 'ONION 通过检测罕见词和异常长度识别后门触发器。',
    'strip': 'STRIP 通过扰动输入观察模型输出变化判断后门样本。',
    'statistical': '统计方法检测罕见标签、异常长度、重复词等特征。',
    'comprehensive': '综合投票聚合多个检测器结果，通过多数投票给出最终判定。',
}

RISK_TYPE_LABELS = {
    'label_poison': '标签投毒', 'backdoor': '后门攻击',
    'distribution_shift': '分布偏移', 'anomaly': '异常样本', 'unknown': '未知风险',
}

SUGGESTION_LABELS = {
    'remove': '建议删除', 'relabel': '建议重标', 'review': '人工复查', 'ignore': '忽略',
}


def _compute_analysis(task):
    """计算检测任务的结构化分析数据。"""
    results = list(DetectionResult.objects.filter(task=task))
    total_suspicious = len(results)
    total_samples = task.dataset.sample_count or 1
    risk_score = task.risk_score or 0

    config_map = {c.detector_name: c for c in AlgorithmConfig.objects.all()}
    det_groups = defaultdict(list)
    for r in results:
        det_groups[r.detector_name].append(r)

    # 风险分解
    type_contribution = defaultdict(lambda: {'count': 0, 'weight': 0, 'max_ratio': 0})
    for det_name, items in det_groups.items():
        cfg = config_map.get(det_name)
        det_type = cfg.detector_type if cfg else 'unknown'
        weight = cfg.weight if cfg else 0.25
        ratio = len(items) / total_samples
        type_contribution[det_type]['count'] += len(items)
        type_contribution[det_type]['weight'] = max(type_contribution[det_type]['weight'], weight)
        type_contribution[det_type]['max_ratio'] = max(type_contribution[det_type]['max_ratio'], ratio)

    risk_decomposition = []
    for det_type, info in type_contribution.items():
        risk_decomposition.append({
            'detector_type': det_type,
            'label': RISK_TYPE_LABELS.get(det_type, det_type),
            'weight': info['weight'],
            'ratio': info['max_ratio'],
            'contribution': info['weight'] * info['max_ratio'],
            'count': info['count'],
        })
    risk_decomposition.sort(key=lambda x: x['contribution'], reverse=True)

    # 各检测器分析
    per_detector = []
    for det_name, items in det_groups.items():
        cfg = config_map.get(det_name)
        confs = [r.confidence for r in items]
        det_type = cfg.detector_type if cfg else 'unknown'
        display_name = (cfg.display_name or det_name) if cfg else det_name
        rt_groups = defaultdict(int)
        sug_groups = defaultdict(int)
        for r in items:
            rt_groups[r.risk_type] += 1
            sug_groups[r.suggestion] += 1
        top_type = max(rt_groups.items(), key=lambda x: x[1])
        per_detector.append({
            'name': det_name,
            'display_name': display_name,
            'type': det_type,
            'type_label': RISK_TYPE_LABELS.get(det_type, det_type),
            'count': len(items),
            'ratio': len(items) / total_samples,
            'avg_conf': sum(confs) / len(confs),
            'max_conf': max(confs),
            'explanation': METHOD_EXPLANATIONS.get(det_name, ''),
            'findings': f"发现 {len(items)} 个可疑样本，平均置信度 {sum(confs)/len(confs)*100:.1f}%，最高 {max(confs)*100:.1f}%。主要风险类型：{RISK_TYPE_LABELS.get(top_type[0], top_type[0])}（{top_type[1]} 条）。",
            'rt_dist': {RISK_TYPE_LABELS.get(k, k): v for k, v in rt_groups.items()},
            'sug_dist': {SUGGESTION_LABELS.get(k, k): v for k, v in sug_groups.items()},
        })
    per_detector.sort(key=lambda x: x['count'], reverse=True)

    # 置信度统计
    all_conf = sorted([r.confidence for r in results])
    n = len(all_conf)
    if n > 0:
        mean = sum(all_conf) / n
        std = math.sqrt(sum((c - mean) ** 2 for c in all_conf) / n) if n > 1 else 0
        conf_stats = {
            'mean': mean, 'std': std,
            'min': all_conf[0], 'max': all_conf[-1],
            'high': sum(1 for c in all_conf if c >= 0.8),
            'medium': sum(1 for c in all_conf if 0.5 <= c < 0.8),
            'low': sum(1 for c in all_conf if c < 0.5),
        }
    else:
        conf_stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'high': 0, 'medium': 0, 'low': 0}

    # 建议分布
    sug_counts = defaultdict(int)
    for r in results:
        sug_counts[r.suggestion] += 1

    # 交叉检测
    sample_dets = defaultdict(set)
    for r in results:
        sample_dets[r.sample_id].add(r.detector_name)
    cross_overlap = sum(1 for dets in sample_dets.values() if len(dets) > 1)

    # 结论
    risk_level = 'high' if risk_score >= 0.15 else 'medium' if risk_score >= 0.05 else 'low'
    pct = total_suspicious / total_samples * 100 if total_samples else 0

    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_level_text': {'high': '高风险', 'medium': '中风险', 'low': '低风险'}[risk_level],
        'total_samples': total_samples,
        'total_suspicious': total_suspicious,
        'pct': pct,
        'risk_decomposition': risk_decomposition,
        'per_detector': per_detector,
        'conf_stats': conf_stats,
        'sug_counts': dict(sug_counts),
        'cross_overlap': cross_overlap,
    }


def _build_report_html(task, analysis):
    """根据分析数据构建完整 HTML 报告。"""
    dataset = task.dataset
    a = analysis
    risk_cls = a['risk_level']

    # ── 检测器结果表 ──
    det_rows = ''
    for d in a['per_detector']:
        det_rows += (
            f'<tr><td><strong>{d["display_name"]}</strong></td>'
            f'<td>{d["type_label"]}</td>'
            f'<td>{d["count"]} / {a["total_samples"]}</td>'
            f'<td>{d["ratio"]*100:.1f}%</td>'
            f'<td>{d["avg_conf"]*100:.1f}%</td>'
            f'<td>{d["max_conf"]*100:.1f}%</td></tr>\n'
        )

    # ── 风险分解表 ──
    decomp_rows = ''
    for d in a['risk_decomposition']:
        decomp_rows += (
            f'<tr><td>{d["label"]}</td>'
            f'<td>{d["weight"]:.2f}</td>'
            f'<td>{d["ratio"]*100:.2f}%</td>'
            f'<td><strong>{d["contribution"]*100:.2f}%</strong></td>'
            f'<td>{d["count"]}</td></tr>\n'
        )

    # ── 各检测器分析卡片 ──
    detector_cards = ''
    for d in a['per_detector']:
        rt_html = '、'.join(f'{k}({v})' for k, v in d['rt_dist'].items())
        sug_html = '、'.join(f'{k}({v})' for k, v in d['sug_dist'].items())
        detector_cards += f'''
        <div style="background:#f8f9fa;border-radius:8px;padding:16px;margin-bottom:12px;border-left:4px solid #409eff;">
          <h4 style="margin:0 0 8px;color:#1a73e8;">{d["display_name"]} <span style="font-size:12px;color:#666;">({d["type_label"]})</span></h4>
          <p style="color:#555;font-size:13px;margin:0 0 8px;"><strong>检测原理：</strong>{d["explanation"]}</p>
          <p style="color:#333;font-size:13px;margin:0 0 8px;"><strong>分析发现：</strong>{d["findings"]}</p>
          <div style="display:flex;gap:20px;font-size:12px;color:#666;">
            <span>风险类型：{rt_html}</span>
            <span>处理建议：{sug_html}</span>
          </div>
        </div>'''

    # ── 建议分布 ──
    sug_rows = ''
    for sug_key in ['remove', 'relabel', 'review', 'ignore']:
        count = a['sug_counts'].get(sug_key, 0)
        if count > 0:
            pct = count / a['total_suspicious'] * 100
            label = SUGGESTION_LABELS.get(sug_key, sug_key)
            sug_rows += f'<tr><td>{label}</td><td><strong>{count}</strong></td><td>{pct:.1f}%</td></tr>\n'

    # ── 高风险样本表 ──
    top_results = DetectionResult.objects.filter(task=task).order_by('-confidence')[:20]
    sample_rows = ''
    for r in top_results:
        conf_cls = 'high' if r.confidence >= 0.8 else 'medium' if r.confidence >= 0.5 else ''
        sample_rows += (
            f'<tr><td>{r.sample_id}</td>'
            f'<td>{RISK_TYPE_LABELS.get(r.risk_type, r.risk_type)}</td>'
            f'<td class="{conf_cls}">{r.confidence*100:.1f}%</td>'
            f'<td>{SUGGESTION_LABELS.get(r.suggestion, r.suggestion)}</td>'
            f'<td>{r.detector_name}</td></tr>\n'
        )

    # ── 净化信息 ──
    clean = CleanResult.objects.filter(task=task).order_by('-created_at').first()
    clean_section = ''
    if clean:
        clean_section = f'''
        <h2>9. 无害化处理结果</h2>
        <table>
          <tr><th>删除样本数</th><td>{clean.removed_count}</td></tr>
          <tr><th>重新标注数</th><td>{clean.relabel_count}</td></tr>
          <tr><th>忽略样本数</th><td>{clean.ignored_count}</td></tr>
          <tr><th>输出路径</th><td>{clean.clean_dataset_path}</td></tr>
        </table>'''

    # ── 综合结论 ──
    if a['risk_level'] == 'high':
        conclusion_color = '#d32f2f'
        conclusion_icon = '⚠️'
    elif a['risk_level'] == 'medium':
        conclusion_color = '#f57c00'
        conclusion_icon = '⚡'
    else:
        conclusion_color = '#388e3c'
        conclusion_icon = '✅'

    recs_html = ''
    if a['conf_stats']['high'] > 0:
        recs_html += f'<li>建议对 {a["conf_stats"]["high"]} 个高置信度（≥80%）可疑样本执行删除操作</li>'
    if a['sug_counts'].get('relabel', 0) > 0:
        recs_html += f'<li>建议对 {a["sug_counts"]["relabel"]} 个标签投毒样本进行人工复核并重新标注</li>'
    if a['sug_counts'].get('review', 0) > 0:
        recs_html += f'<li>建议对 {a["sug_counts"]["review"]} 个待复查样本进行人工审查确认</li>'
    if a['cross_overlap'] > 0:
        recs_html += f'<li>有 {a["cross_overlap"]} 个样本被多个检测器同时标记，应优先处理</li>'
    if not recs_html:
        recs_html = '<li>数据集整体风险较低，可安全用于模型训练</li>'

    html = f'''<!DOCTYPE html>
<html lang="zh-hans">
<head>
<meta charset="utf-8">
<title>DPDS 检测报告 - {task.task_no}</title>
<style>
  body {{ font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif; margin: 40px; color: #333; line-height: 1.6; }}
  h1 {{ color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 12px; }}
  h2 {{ color: #555; border-bottom: 1px solid #ddd; padding-bottom: 8px; margin-top: 32px; }}
  h3 {{ color: #444; margin-top: 20px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  .high {{ color: #d32f2f; font-weight: bold; }}
  .medium {{ color: #f57c00; }}
  .low {{ color: #388e3c; }}
  .conclusion-box {{ background: #f8f9fa; border-radius: 8px; padding: 20px; margin: 16px 0; border-left: 5px solid {conclusion_color}; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 16px 0; }}
  .stat-card {{ background: #f8f9fa; border-radius: 8px; padding: 16px; text-align: center; }}
  .stat-value {{ font-size: 28px; font-weight: bold; color: #1a73e8; }}
  .stat-label {{ font-size: 13px; color: #666; margin-top: 4px; }}
  .method-card {{ background: #f8f9fa; border-radius: 8px; padding: 16px; margin-bottom: 12px; border-left: 4px solid #409eff; }}
  .meta {{ color: #999; font-size: 12px; margin-top: 40px; }}
</style>
</head>
<body>
<h1>数据投毒检测报告</h1>

<h2>1. 概览</h2>
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-value">{a["total_suspicious"]}</div>
    <div class="stat-label">可疑样本数</div>
  </div>
  <div class="stat-card">
    <div class="stat-value" style="color:{conclusion_color};">{a["risk_score"]*100:.1f}%</div>
    <div class="stat-label">综合风险分数</div>
  </div>
  <div class="stat-card">
    <div class="stat-value" style="color:{conclusion_color};font-size:20px;">{a["risk_level_text"]}</div>
    <div class="stat-label">风险等级</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{a["total_samples"]:,}</div>
    <div class="stat-label">数据集样本总数</div>
  </div>
</div>

<div class="conclusion-box">
  <div style="font-size:18px;font-weight:bold;color:{conclusion_color};margin-bottom:8px;">
    {conclusion_icon} {a["risk_level_text"]} — 检测结论
  </div>
  <p style="margin:0;">该数据集存在{"高度" if a["risk_level"]=="high" else "中等" if a["risk_level"]=="medium" else "较低"}数据投毒风险
  （综合风险分数 {a["risk_score"]*100:.1f}%），共发现 {a["total_suspicious"]} 个可疑样本
  （约占 {a["pct"]:.1f}%）。</p>
</div>

<h2>2. 数据集信息</h2>
<table>
  <tr><th>数据集名称</th><td>{dataset.dataset_name}</td></tr>
  <tr><th>数据类型</th><td>{dataset.dataset_type}</td></tr>
  <tr><th>样本总数</th><td>{a["total_samples"]:,}</td></tr>
  <tr><th>标签字段</th><td>{dataset.label_field or "-"}</td></tr>
  <tr><th>任务编号</th><td>{task.task_no}</td></tr>
  <tr><th>完成时间</th><td>{task.finished_at.strftime("%Y-%m-%d %H:%M:%S") if task.finished_at else "-"}</td></tr>
</table>

<h2>3. 风险评分分解</h2>
<p>综合风险分数由各检测器类型的可疑比例加权求和得到，公式：<code>risk_score = Σ(weight × ratio)</code></p>
<table>
  <tr><th>风险类型</th><th>权重</th><th>可疑比例</th><th>贡献值</th><th>可疑样本数</th></tr>
  {decomp_rows}
  <tr style="background:#f0f0f0;font-weight:bold;">
    <td>合计</td><td>-</td><td>-<td>{a["risk_score"]*100:.2f}%</td><td>{a["total_suspicious"]}</td>
  </tr>
</table>

<h2>4. 检测方法说明</h2>
<p>本次检测使用了以下 {len(a["per_detector"])} 种检测算法：</p>
{"".join(f'<div class="method-card"><strong>{d["display_name"]}</strong>（{d["type_label"]}）<br><span style="color:#555;">{d["explanation"]}</span></div>' for d in a["per_detector"])}

<h2>5. 各检测器分析结果</h2>
<table>
  <tr><th>检测器</th><th>类型</th><th>可疑/总数</th><th>可疑比例</th><th>平均置信度</th><th>最高置信度</th></tr>
  {det_rows}
</table>
{detector_cards}

<h2>6. 置信度统计</h2>
<table>
  <tr><th>统计指标</th><th>值</th></tr>
  <tr><td>平均置信度</td><td><strong>{a["conf_stats"]["mean"]*100:.1f}%</strong></td></tr>
  <tr><td>标准差</td><td>{a["conf_stats"]["std"]*100:.1f}%</td></tr>
  <tr><td>最低置信度</td><td>{a["conf_stats"]["min"]*100:.1f}%</td></tr>
  <tr><td>最高置信度</td><td>{a["conf_stats"]["max"]*100:.1f}%</td></tr>
  <tr><td>高置信度（≥80%）样本数</td><td class="high">{a["conf_stats"]["high"]}</td></tr>
  <tr><td>中置信度（50%-80%）样本数</td><td class="medium">{a["conf_stats"]["medium"]}</td></tr>
  <tr><td>低置信度（＜50%）样本数</td><td>{a["conf_stats"]["low"]}</td></tr>
</table>

<h2>7. 处理建议分布</h2>
<table>
  <tr><th>建议类型</th><th>数量</th><th>占比</th></tr>
  {sug_rows}
</table>

<h2>8. 高风险样本（Top 20）</h2>
<table>
  <tr><th>样本 ID</th><th>风险类型</th><th>置信度</th><th>处理建议</th><th>检测器</th></tr>
  {sample_rows}
</table>

{clean_section}

<h2>{"10" if clean else "9"}. 综合结论与建议</h2>
<div class="conclusion-box">
  <h3 style="margin-top:0;">行动建议</h3>
  <ol style="margin:0;padding-left:20px;">
    {recs_html}
  </ol>
</div>

<p class="meta">
  报告生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | DPDS 数据投毒检测系统<br>
  本报告基于自动化检测算法生成，建议结合人工审查做出最终决策。
</p>
</body>
</html>'''
    return html


def generate_report(task_id: str, report_type: str = 'html', user=None) -> Report:
    task = DetectionTask.objects.select_related('dataset', 'created_by').get(id=task_id)
    analysis = _compute_analysis(task)
    html_content = _build_report_html(task, analysis)

    out_dir = settings.MEDIA_ROOT / 'reports'
    os.makedirs(out_dir, exist_ok=True)
    out_path = str(out_dir / f'{uuid.uuid4().hex}.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return Report.objects.create(
        task=task,
        report_type=report_type,
        file_path=out_path,
        created_by=user,
    )
