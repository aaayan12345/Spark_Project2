#!/usr/bin/env python3
"""
Generate an editable PowerPoint roadshow deck for InsightPilot AI.

This script writes a PPTX directly with Office Open XML so it does not require
python-pptx. All text blocks and shapes are editable in PowerPoint/WPS.
"""

from __future__ import annotations

import html
import os
import zipfile
from dataclasses import dataclass


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "InsightPilot_AI_Roadshow_Pitch_Deck.pptx")

SLIDE_W = 12192000
SLIDE_H = 6858000
EMU = 914400

NAVY = "0B1F3A"
BLUE = "1F6FEB"
CYAN = "31C6D4"
GREEN = "23A559"
ORANGE = "F59E0B"
RED = "D92D20"
INK = "172033"
MUTED = "5B667A"
LIGHT = "F5F8FC"
LINE = "D8E0EA"
WHITE = "FFFFFF"
SILVER = "E9EEF5"


def x(inch: float) -> int:
    return int(inch * EMU)


def esc(s: str) -> str:
    return html.escape(s, quote=False)


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    text: str = ""
    fill: str = WHITE
    line: str = LINE
    radius: bool = True
    font: int = 18
    color: str = INK
    bold: bool = False
    align: str = "l"
    valign: str = "mid"


def tx_body(text: str, font: int, color: str, bold: bool = False, align: str = "l", valign: str = "mid") -> str:
    anchor = {"top": "t", "mid": "mid", "bottom": "b"}.get(valign, "mid")
    p_align = {"l": "l", "c": "ctr", "r": "r"}.get(align, "l")
    lines = text.split("\n")
    paras = []
    for line in lines:
        if not line:
            paras.append(f'<a:p><a:pPr algn="{p_align}"/><a:endParaRPr lang="zh-CN" sz="{font*100}"/></a:p>')
            continue
        paras.append(
            f'<a:p><a:pPr algn="{p_align}"/>'
            f'<a:r><a:rPr lang="zh-CN" sz="{font*100}"{" b=\"1\"" if bold else ""}>'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            f'<a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/>'
            f'</a:rPr><a:t>{esc(line)}</a:t></a:r></a:p>'
        )
    return f'<p:txBody><a:bodyPr wrap="square" anchor="{anchor}" lIns="120000" rIns="120000" tIns="70000" bIns="70000"/><a:lstStyle/>{"".join(paras)}</p:txBody>'


def shape_xml(idx: int, b: Box) -> str:
    prst = "roundRect" if b.radius else "rect"
    line_xml = (
        f'<a:ln w="9000"><a:solidFill><a:srgbClr val="{b.line}"/></a:solidFill></a:ln>'
        if b.line else '<a:ln><a:noFill/></a:ln>'
    )
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{idx}" name="Shape {idx}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{x(b.x)}" y="{x(b.y)}"/><a:ext cx="{x(b.w)}" cy="{x(b.h)}"/></a:xfrm>
    <a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>
    <a:solidFill><a:srgbClr val="{b.fill}"/></a:solidFill>
    {line_xml}
  </p:spPr>
  {tx_body(b.text, b.font, b.color, b.bold, b.align, b.valign)}
</p:sp>"""


def line_xml(idx: int, x1: float, y1: float, x2: float, y2: float, color: str = LINE, width: int = 18000) -> str:
    return f"""
<p:cxnSp>
  <p:nvCxnSpPr><p:cNvPr id="{idx}" name="Line {idx}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{x(x1)}" y="{x(y1)}"/><a:ext cx="{x(x2-x1)}" cy="{x(y2-y1)}"/></a:xfrm>
    <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
    <a:ln w="{width}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:ln>
  </p:spPr>
</p:cxnSp>"""


def title(slide_no: int, section: str, heading: str, sub: str = "") -> list[Box]:
    boxes = [
        Box(0.45, 0.28, 1.6, 0.35, f"{slide_no:02d} / {section}", fill=LIGHT, line="", font=10, color=MUTED, bold=True, align="c"),
        Box(0.62, 0.75, 10.8, 0.45, heading, fill=WHITE, line="", font=24, color=NAVY, bold=True, radius=False, valign="mid"),
    ]
    if sub:
        boxes.append(Box(0.62, 1.23, 10.6, 0.35, sub, fill=WHITE, line="", font=11, color=MUTED, radius=False))
    return boxes


slides: list[list[Box | tuple]] = []


slides.append([
    Box(0, 0, 13.333, 7.5, "", fill=NAVY, line="", radius=False),
    Box(0.55, 0.45, 2.1, 0.34, "SEED ROADSHOW", fill=BLUE, line="", font=11, color=WHITE, bold=True, align="c"),
    Box(0.75, 1.35, 8.3, 1.15, "InsightPilot AI", fill=NAVY, line="", font=44, color=WHITE, bold=True, radius=False),
    Box(0.8, 2.55, 8.5, 0.55, "面向中小企业的 AI Data Analyst in a Box", fill=NAVY, line="", font=22, color="BFE7FF", bold=True, radius=False),
    Box(0.82, 3.35, 7.4, 0.8, "上传数据，用自然语言提问，自动获得分析代码、图表、洞察和可导出报告。", fill=NAVY, line="", font=15, color=SILVER, radius=False),
    Box(8.7, 1.15, 3.6, 4.75, "", fill="12345C", line="2C5F94"),
    Box(9.05, 1.55, 2.9, 0.42, "融资目标", fill="12345C", line="", font=16, color=SILVER, bold=True, align="c"),
    Box(9.15, 2.12, 2.7, 0.85, "$500K", fill="12345C", line="", font=34, color=WHITE, bold=True, align="c"),
    Box(9.3, 3.08, 2.4, 0.38, "18 个月 runway", fill="12345C", line="", font=13, color="BFE7FF", align="c"),
    Box(9.25, 4.05, 2.5, 1.15, "用于产品工程、LLM/API 成本、早期获客与安全合规", fill="12345C", line="", font=13, color=SILVER, align="c"),
    Box(0.75, 6.62, 7.8, 0.28, "Project 2 | LLM Startup Business Plan | Editable Pitch Deck", fill=NAVY, line="", font=10, color="9FB3C8", radius=False),
])


slides.append([
    *title(2, "Problem", "中小企业有数据，但没有数据团队", "传统 BI 太重，人工 analyst 太贵，通用 chatbot 又缺少可复现 workflow。"),
    Box(0.75, 1.85, 3.55, 3.8, "数据分散\n\n销售、广告、库存、客户数据散落在 CSV、Excel、Shopify、Google Ads、CRM 中。", fill=WHITE, line=LINE, font=16, color=INK, bold=False),
    Box(4.65, 1.85, 3.55, 3.8, "分析太慢\n\n业务人员依赖手工表格或外部 analyst，问题到答案之间常常相隔数小时或数天。", fill=WHITE, line=LINE, font=16, color=INK),
    Box(8.55, 1.85, 3.55, 3.8, "结果难复现\n\n普通 AI chat 可以回答问题，但分析过程、代码、图表和团队协作难以长期沉淀。", fill=WHITE, line=LINE, font=16, color=INK),
    Box(0.85, 6.25, 11.3, 0.42, "痛点结论：SMEs 需要一个可解释、可执行、可复现的 AI 数据分析工作流，而不只是一个聊天窗口。", fill=LIGHT, line="", font=15, color=NAVY, bold=True, align="c"),
])


slides.append([
    *title(3, "Solution", "InsightPilot AI：AI Data Analyst in a Box", "把上传数据、自然语言提问、代码执行、图表生成和报告导出放进同一工作流。"),
    Box(0.7, 1.65, 2.3, 1.25, "1\n上传数据\nCSV / Excel / Sheets / DB", fill=LIGHT, line=LINE, font=15, color=NAVY, bold=True, align="c"),
    Box(3.25, 1.65, 2.3, 1.25, "2\n自然语言提问\n业务问题而非 SQL", fill=LIGHT, line=LINE, font=15, color=NAVY, bold=True, align="c"),
    Box(5.8, 1.65, 2.3, 1.25, "3\nAgent 分析\n计划 + Python/SQL", fill=LIGHT, line=LINE, font=15, color=NAVY, bold=True, align="c"),
    Box(8.35, 1.65, 2.3, 1.25, "4\n沙箱执行\n可验证结果", fill=LIGHT, line=LINE, font=15, color=NAVY, bold=True, align="c"),
    Box(10.9, 1.65, 1.75, 1.25, "5\n报告\n图表 + 洞察", fill=LIGHT, line=LINE, font=15, color=NAVY, bold=True, align="c"),
    Box(1.0, 4.0, 3.4, 1.05, "给 Founder\n看懂收入、客户、库存变化", fill=WHITE, line=BLUE, font=16, color=INK, bold=True),
    Box(4.95, 4.0, 3.4, 1.05, "给 Marketing\n解释 ROI 与投放浪费", fill=WHITE, line=CYAN, font=16, color=INK, bold=True),
    Box(8.9, 4.0, 3.0, 1.05, "给 Analyst\n减少重复报表劳动", fill=WHITE, line=GREEN, font=16, color=INK, bold=True),
    ("line", 2.98, 2.28, 3.24, 2.28, BLUE),
    ("line", 5.54, 2.28, 5.79, 2.28, BLUE),
    ("line", 8.09, 2.28, 8.34, 2.28, BLUE),
    ("line", 10.65, 2.28, 10.88, 2.28, BLUE),
])


slides.append([
    *title(4, "Product", "核心产品能力", "面向中小团队的自然语言数据分析 SaaS。"),
    Box(0.65, 1.55, 3.75, 1.35, "Data Upload & Connectors\n\nCSV/Excel MVP，后续接入 Google Sheets、Shopify、Stripe、HubSpot、PostgreSQL。", fill=WHITE, line=LINE, font=13, color=INK, bold=True),
    Box(4.65, 1.55, 3.75, 1.35, "Natural-language Analysis\n\n用户用业务语言提问，Agent 自动生成分析计划、代码和解释。", fill=WHITE, line=LINE, font=13, color=INK, bold=True),
    Box(8.65, 1.55, 3.75, 1.35, "Data Cleaning & Profiling\n\n自动检测缺失值、重复行、异常值、日期列和字段类型。", fill=WHITE, line=LINE, font=13, color=INK, bold=True),
    Box(0.65, 3.35, 3.75, 1.35, "Charts & Reports\n\n自动生成趋势图、柱状图、漏斗图、cohort 表和 PDF/PPT 报告。", fill=WHITE, line=LINE, font=13, color=INK, bold=True),
    Box(4.65, 3.35, 3.75, 1.35, "Transparent Code\n\n展示 Python/SQL code 与执行日志，让分析过程可验证、可复现。", fill=WHITE, line=LINE, font=13, color=INK, bold=True),
    Box(8.65, 3.35, 3.75, 1.35, "Reusable Workflow\n\n保存历史分析、常用指标定义和团队共享项目。", fill=WHITE, line=LINE, font=13, color=INK, bold=True),
    Box(0.8, 5.65, 11.8, 0.6, "MVP 已可复用 Q1 Data Agent Demo：文件上传 + 数据摘要注入 + 流式问答 + Code Runner + 图表输出。", fill=NAVY, line="", font=15, color=WHITE, bold=True, align="c"),
])


slides.append([
    *title(5, "Market", "Beachhead Market：电商与数字营销团队", "先从频繁使用数据、但缺少专职数据团队的 SMEs 切入。"),
    Box(0.75, 1.7, 3.5, 1.2, "5-100 人团队", fill=LIGHT, line="", font=24, color=BLUE, bold=True, align="c"),
    Box(4.55, 1.7, 3.5, 1.2, "高频分析场景", fill=LIGHT, line="", font=24, color=BLUE, bold=True, align="c"),
    Box(8.35, 1.7, 3.5, 1.2, "低数据团队配置", fill=LIGHT, line="", font=24, color=BLUE, bold=True, align="c"),
    Box(0.8, 3.35, 2.6, 1.45, "E-commerce Founder\n\n关心收入、产品表现、库存、退款。", fill=WHITE, line=LINE, font=13, color=INK, bold=True),
    Box(3.65, 3.35, 2.6, 1.45, "Marketing Manager\n\n关心 ROI、渠道表现、投放浪费。", fill=WHITE, line=LINE, font=13, color=INK, bold=True),
    Box(6.5, 3.35, 2.6, 1.45, "Operations Manager\n\n关心履约、供应链、异常检测。", fill=WHITE, line=LINE, font=13, color=INK, bold=True),
    Box(9.35, 3.35, 2.6, 1.45, "Junior Analyst\n\n关心重复报表和可复现分析。", fill=WHITE, line=LINE, font=13, color=INK, bold=True),
    Box(1.15, 5.8, 10.9, 0.5, "典型问题：Which product category grew fastest? Which campaign wasted budget? Why did revenue drop last week?", fill=WHITE, line=BLUE, font=14, color=NAVY, bold=True, align="c"),
])


slides.append([
    *title(6, "Competition", "竞品格局与差异化", "InsightPilot AI 位于通用 AI chat 与企业 BI 之间的空缺地带。"),
    Box(0.7, 1.55, 2.4, 4.6, "Generic AI Chat\n\nChatGPT Advanced Data Analysis\n\n优势：灵活强大\n不足：项目/团队 workflow 弱", fill=WHITE, line=LINE, font=13, color=INK, bold=True),
    Box(3.25, 1.55, 2.4, 4.6, "No-code AI Analysis\n\nJulius AI / Rows AI / Capalyze\n\n优势：上手快\n不足：复杂分析与工作流沉淀不足", fill=WHITE, line=LINE, font=13, color=INK, bold=True),
    Box(5.8, 1.55, 2.4, 4.6, "Technical Notebooks\n\nHex / Deepnote\n\n优势：协作和代码强\n不足：非技术用户门槛高", fill=WHITE, line=LINE, font=13, color=INK, bold=True),
    Box(8.35, 1.55, 2.4, 4.6, "Enterprise BI\n\nThoughtSpot / Tableau / Power BI\n\n优势：企业生态成熟\n不足：setup cost 高", fill=WHITE, line=LINE, font=13, color=INK, bold=True),
    Box(10.9, 1.55, 1.65, 4.6, "Our Edge\n\nSME workflow\nTransparent code\nReusable reports\nLow setup cost", fill=NAVY, line="", font=13, color=WHITE, bold=True, align="c"),
])


slides.append([
    *title(7, "Business Model", "SaaS 订阅 + 高价值服务", "用 freemium 获客，用团队协作、连接器、报告导出和大数据集能力变现。"),
    Box(0.6, 1.55, 2.35, 3.9, "Free\n\n$0/mo\n\n有限上传\n有限消息\n导出水印", fill=LIGHT, line=LINE, font=15, color=INK, bold=True, align="c"),
    Box(3.1, 1.55, 2.35, 3.9, "Starter\n\n$19/mo\n\nCSV/Excel 分析\n图表导出\n保存项目", fill=WHITE, line=BLUE, font=15, color=INK, bold=True, align="c"),
    Box(5.6, 1.55, 2.35, 3.9, "Team\n\n$49/user/mo\n\n共享 workspace\nscheduled reports\nconnectors", fill=WHITE, line=BLUE, font=15, color=INK, bold=True, align="c"),
    Box(8.1, 1.55, 2.35, 3.9, "Business\n\n$299+/mo\n\nDB connectors\n权限管理\naudit logs", fill=WHITE, line=BLUE, font=15, color=INK, bold=True, align="c"),
    Box(10.6, 1.55, 2.25, 3.9, "Services\n\nCustom\n\n数据接入\n模板配置\n企业培训", fill=WHITE, line=LINE, font=15, color=INK, bold=True, align="c"),
    Box(0.85, 6.0, 11.5, 0.5, "付费触发点：更大文件、更多分析次数、团队协作、数据连接器、PDF/PPT export、定期报告。", fill=NAVY, line="", font=15, color=WHITE, bold=True, align="c"),
])


slides.append([
    *title(8, "Technology", "技术壁垒：Agent + Sandbox + Reproducible Workflow", "不是单纯包装 LLM API，而是把分析链路产品化。"),
    Box(0.7, 1.55, 3.0, 4.85, "LLM Engine\n\n- LLM Gateway\n- Model routing\n- Fine-tuned Qwen\n- Fallback model\n- Token/cost tracking", fill=WHITE, line=BLUE, font=14, color=INK, bold=True),
    Box(3.95, 1.55, 3.0, 4.85, "Agent Orchestrator\n\n- Data understanding\n- Analysis planning\n- Python/SQL generation\n- Validation Agent\n- Report Agent", fill=WHITE, line=CYAN, font=14, color=INK, bold=True),
    Box(7.2, 1.55, 3.0, 4.85, "Execution Sandbox\n\n- Isolated container\n- Time/memory limit\n- Restricted network\n- Code scan\n- Logs + charts", fill=WHITE, line=GREEN, font=14, color=INK, bold=True),
    Box(10.45, 1.55, 2.0, 4.85, "Memory & Reports\n\n- Project memory\n- Metric definitions\n- Result cache\n- PDF/PPT export", fill=NAVY, line="", font=14, color=WHITE, bold=True, align="c"),
])


slides.append([
    *title(9, "Architecture", "工业级架构支持 100,000-level concurrency", "Hot path 快速响应，Cold path 异步处理重型 AI 任务。"),
    Box(0.55, 1.5, 2.0, 0.75, "Users", fill=LIGHT, line=LINE, font=18, color=NAVY, bold=True, align="c"),
    Box(2.9, 1.5, 2.0, 0.75, "CDN / WAF\nLoad Balancer", fill=LIGHT, line=LINE, font=14, color=NAVY, bold=True, align="c"),
    Box(5.25, 1.5, 2.0, 0.75, "API Gateway\nMicroservices", fill=LIGHT, line=LINE, font=14, color=NAVY, bold=True, align="c"),
    Box(7.6, 1.5, 2.0, 0.75, "Message Queue\nKafka/RabbitMQ", fill=LIGHT, line=LINE, font=14, color=NAVY, bold=True, align="c"),
    Box(9.95, 1.5, 2.65, 0.75, "Agent Workers\nSandbox Workers", fill=LIGHT, line=LINE, font=14, color=NAVY, bold=True, align="c"),
    Box(0.8, 3.25, 3.1, 1.15, "High Concurrency\n\nStateless API + autoscaling + Redis + queue backpressure", fill=WHITE, line=BLUE, font=13, color=INK, bold=True),
    Box(4.25, 3.25, 3.1, 1.15, "Data Layer\n\nPostgreSQL metadata + Object Storage + Vector DB + Result Store", fill=WHITE, line=CYAN, font=13, color=INK, bold=True),
    Box(7.7, 3.25, 3.1, 1.15, "Monitoring\n\nPrometheus + Grafana + OpenTelemetry + Alertmanager", fill=WHITE, line=GREEN, font=13, color=INK, bold=True),
    Box(2.0, 5.45, 9.2, 0.6, "关键设计：普通请求横向扩展，LLM 推理/代码执行进入队列，按 plan 和 quota 做优先级调度。", fill=NAVY, line="", font=15, color=WHITE, bold=True, align="c"),
    ("line", 2.55, 1.88, 2.9, 1.88, BLUE),
    ("line", 4.9, 1.88, 5.25, 1.88, BLUE),
    ("line", 7.25, 1.88, 7.6, 1.88, BLUE),
    ("line", 9.6, 1.88, 9.95, 1.88, BLUE),
])


slides.append([
    *title(10, "Go-to-Market", "低成本 indie GTM：模板 + 内容 + Product Hunt", "先验证真实需求，再扩大获客。"),
    Box(0.8, 1.7, 3.3, 3.8, "Phase 1\nMVP Validation\n\n- 基于 Q1 demo 构建 MVP\n- 访谈 20-30 位用户\n- 10 位 active testers\n- Shopify / Ads / Customer 模板", fill=WHITE, line=BLUE, font=14, color=INK, bold=True),
    Box(4.55, 1.7, 3.3, 3.8, "Phase 2\nCommunity Growth\n\n- Product Hunt launch\n- LinkedIn / X / YouTube demo\n- SEO: AI CSV analysis tool\n- 案例文章和短视频", fill=WHITE, line=CYAN, font=14, color=INK, bold=True),
    Box(8.3, 1.7, 3.3, 3.8, "Phase 3\nPaid Conversion\n\n- Team workspace\n- Scheduled reports\n- Connectors\n- PDF/PPT exports\n- Usage-based upgrade", fill=WHITE, line=GREEN, font=14, color=INK, bold=True),
    Box(1.3, 6.0, 10.6, 0.48, "核心增长假设：用户先用模板解决一个真实业务问题，再因复用、协作、连接器和报告能力升级付费。", fill=LIGHT, line="", font=14, color=NAVY, bold=True, align="c"),
])


slides.append([
    *title(11, "Milestones", "18 个月里程碑与收入目标", "从 MVP 到 seed-ready traction。"),
    Box(0.7, 1.55, 2.25, 3.8, "Month 1-2\n\nMVP + 用户访谈\n\n30 interviewed users\n10 active testers", fill=WHITE, line=LINE, font=14, color=INK, bold=True, align="c"),
    Box(3.15, 1.55, 2.25, 3.8, "Month 3\n\nPublic Beta\n\n500 signups\n50 WAU", fill=WHITE, line=LINE, font=14, color=INK, bold=True, align="c"),
    Box(5.6, 1.55, 2.25, 3.8, "Month 6\n\nFirst Revenue\n\n100 paying users\nMRR $2K-$5K", fill=WHITE, line=BLUE, font=14, color=INK, bold=True, align="c"),
    Box(8.05, 1.55, 2.25, 3.8, "Month 12\n\nTeam Plan\n\n500 paying users\nMRR $20K+", fill=WHITE, line=BLUE, font=14, color=INK, bold=True, align="c"),
    Box(10.5, 1.55, 2.05, 3.8, "Month 18\n\nSeed-ready\n\nRetention\nConnectors\nPilots", fill=NAVY, line="", font=14, color=WHITE, bold=True, align="c"),
    Box(1.05, 5.95, 11.0, 0.55, "衡量指标：weekly active analysis、code execution success rate、report export rate、paid conversion、MRR。", fill=LIGHT, line="", font=14, color=NAVY, bold=True, align="c"),
])


slides.append([
    *title(12, "Funding Ask", "融资 $500K，支持 18 个月 runway", "资金用于把 Q1 技术原型推进为 production-ready SaaS。"),
    Box(0.8, 1.55, 3.45, 4.7, "$500K\n\nSeed Funding Ask\n\n目标：18 个月 runway\n完成产品化、连接器、早期付费用户和安全合规。", fill=NAVY, line="", font=20, color=WHITE, bold=True, align="c"),
    Box(4.75, 1.65, 1.9, 2.7, "45%\n\nProduct\nEngineering", fill=WHITE, line=BLUE, font=18, color=NAVY, bold=True, align="c"),
    Box(6.9, 1.65, 1.9, 2.7, "25%\n\nLLM/API\nCloud", fill=WHITE, line=CYAN, font=18, color=NAVY, bold=True, align="c"),
    Box(9.05, 1.65, 1.9, 2.7, "20%\n\nSales\nMarketing", fill=WHITE, line=GREEN, font=18, color=NAVY, bold=True, align="c"),
    Box(11.2, 1.65, 1.35, 2.7, "10%\n\nLegal\nOps", fill=WHITE, line=ORANGE, font=17, color=NAVY, bold=True, align="c"),
    Box(4.8, 5.05, 7.7, 0.75, "融资后成果：production-ready SaaS、5-8 个 connectors、1,000+ paying users 或 100+ team customers。", fill=LIGHT, line="", font=15, color=NAVY, bold=True, align="c"),
])


slides.append([
    Box(0, 0, 13.333, 7.5, "", fill=NAVY, line="", radius=False),
    Box(0.85, 0.85, 8.8, 0.95, "Thank You", fill=NAVY, line="", font=40, color=WHITE, bold=True, radius=False),
    Box(0.9, 2.1, 8.8, 0.58, "InsightPilot AI", fill=NAVY, line="", font=28, color="BFE7FF", bold=True, radius=False),
    Box(0.92, 2.95, 7.8, 0.6, "让每个小团队都拥有一个可解释、可执行、可复现的 AI Data Analyst。", fill=NAVY, line="", font=18, color=SILVER, radius=False),
    Box(8.9, 1.25, 3.5, 4.35, "Next Step\n\n- 完成 MVP 上线\n- 补齐 Shopify / Sheets connectors\n- 获取 10 个 active testers\n- 准备 Product Hunt launch\n- 寻找 seed investors", fill="12345C", line="2C5F94", font=16, color=WHITE, bold=True),
    Box(0.95, 6.55, 7.8, 0.32, "LLM assisted by OpenAI ChatGPT / Codex based on GPT-5; final content checked by student.", fill=NAVY, line="", font=10, color="9FB3C8", radius=False),
])


def slide_xml(items: list[Box | tuple]) -> str:
    body = []
    idx = 2
    for item in items:
        if isinstance(item, Box):
            body.append(shape_xml(idx, item))
            idx += 1
        else:
            _, x1, y1, x2, y2, color = item
            body.append(line_xml(idx, x1, y1, x2, y2, color))
            idx += 1
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {''.join(body)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''


def content_types(nslides: int) -> str:
    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
    ]
    overrides.extend(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, nslides + 1)
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  {''.join(overrides)}
</Types>'''


def presentation_xml(nslides: int) -> str:
    sld_ids = ''.join(
        f'<p:sldId id="{255+i}" r:id="rId{i+1}"/>' for i in range(1, nslides + 1)
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{sld_ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle/>
</p:presentation>'''


def presentation_rels(nslides: int) -> str:
    rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>']
    rels.extend(
        f'<Relationship Id="rId{i+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, nslides + 1)
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(rels)}
</Relationships>'''


ROOT_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>'''


SLIDE_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>'''


MASTER_XML = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
  </p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>'''


MASTER_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>'''


LAYOUT_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
  </p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>'''


LAYOUT_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>'''


THEME_XML = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="InsightPilot">
  <a:themeElements>
    <a:clrScheme name="InsightPilot">
      <a:dk1><a:srgbClr val="{NAVY}"/></a:dk1><a:lt1><a:srgbClr val="{WHITE}"/></a:lt1>
      <a:dk2><a:srgbClr val="{INK}"/></a:dk2><a:lt2><a:srgbClr val="{LIGHT}"/></a:lt2>
      <a:accent1><a:srgbClr val="{BLUE}"/></a:accent1><a:accent2><a:srgbClr val="{CYAN}"/></a:accent2>
      <a:accent3><a:srgbClr val="{GREEN}"/></a:accent3><a:accent4><a:srgbClr val="{ORANGE}"/></a:accent4>
      <a:accent5><a:srgbClr val="{RED}"/></a:accent5><a:accent6><a:srgbClr val="{MUTED}"/></a:accent6>
      <a:hlink><a:srgbClr val="{BLUE}"/></a:hlink><a:folHlink><a:srgbClr val="{MUTED}"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Microsoft YaHei"><a:majorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:majorFont><a:minorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="InsightPilot"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/><a:extraClrSchemeLst/>
</a:theme>'''


def build() -> None:
    nslides = len(slides)
    with zipfile.ZipFile(OUT_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(nslides))
        z.writestr("_rels/.rels", ROOT_RELS)
        z.writestr("ppt/presentation.xml", presentation_xml(nslides))
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(nslides))
        z.writestr("ppt/slideMasters/slideMaster1.xml", MASTER_XML)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", MASTER_RELS)
        z.writestr("ppt/slideLayouts/slideLayout1.xml", LAYOUT_XML)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", LAYOUT_RELS)
        z.writestr("ppt/theme/theme1.xml", THEME_XML)
        for i, slide in enumerate(slides, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(slide))
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", SLIDE_RELS)
    print(OUT_PATH)


if __name__ == "__main__":
    build()
