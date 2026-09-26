"""
IWriting 国际中文写作智能反馈系统
统一入口：登录后按角色自动进入学生端或教师端

重要说明：
本文件不修改 final_agent.py 的任何内容，
批改逻辑、评分量表、提示词全部通过 import 复用。
"""

import sys

# 修复：Windows 下某些启动方式给 Python 进程分配的默认控制台编码是 GBK，
# 而 final_agent.py 内部有调试打印语句包含 emoji 字符，GBK 无法编码会导致
# UnicodeEncodeError 未捕获异常，使当次请求在服务端直接崩溃（前端表现为一直
# processing、最终页面空白，且没有任何浏览器端报错）。这里在程序入口最早的位置
# 把标准输出/错误强制设为 UTF-8，避免此类打印语句造成请求处理中途崩溃。
# 不改动 final_agent.py 任何内容。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import base64
import secrets
import time
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from fastapi import Form
from fastapi.responses import HTMLResponse, RedirectResponse
from html import escape as html_escape

import db_helper as db

# 从 final_agent 复用批改能力，不做任何修改
from final_agent import (
    get_feedback,
    build_report_html,
    build_translated_html,
    EMPTY_RIGHT_HTML,
    css as base_css,
)

DIMS = ["内容与切题", "篇章结构", "语篇连贯性", "词汇运用", "语法准确性"]
COL_MAP = {
    "内容与切题": "score_content",
    "篇章结构": "score_structure",
    "语篇连贯性": "score_coherence",
    "词汇运用": "score_vocabulary",
    "语法准确性": "score_grammar",
}
PRACTICE_LABEL = "自由练习"
ALL_STUDENTS = "全班"
ALL_ASSIGN = "全部作业"
TOTAL_DIM = "总分"
ALL_DIMS = "五维全部"

BLUE = "#378ADD"
DARK = "#0C447C"


# ==================== 公共工具 ====================

def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def img_html(b64, max_width=None):
    if not b64:
        return "<div style='padding:36px;text-align:center;color:#B0C4DE;font-size:12px;'>暂无数据</div>"
    if max_width:
        return (f"<div style='text-align:center;'>"
                f"<img src='data:image/png;base64,{b64}' "
                f"style='width:100%;max-width:{max_width}px;'/></div>")
    return f"<img src='data:image/png;base64,{b64}' style='width:100%;'/>"


def metric_cards(items):
    cards = ""
    for label, value, color in items:
        cards += f"""
<div style='background:#F8FBFF;border-radius:8px;padding:12px 14px;'>
  <div style='font-size:10px;color:#5F7FA5;'>{label}</div>
  <div style='font-size:22px;font-weight:500;color:{color};margin-top:3px;'>{value}</div>
</div>"""
    n = len(items)
    return f"<div style='display:grid;grid-template-columns:repeat({n},1fr);gap:8px;'>{cards}</div>"


def make_header(user, subtitle=""):
    sub = subtitle or f"{user['role']}　{user['class_name']}"
    return f"""
<div style='background:#1D5FA5;padding:14px 20px;border-radius:12px;'>
  <div style='color:white;font-size:16px;font-weight:500;'>IWriting 国际中文写作智能反馈系统</div>
  <div style='color:#B5D4F4;font-size:11px;margin-top:2px;'>{user['real_name']}　{sub}</div>
</div>"""


# ==================== 学生端功能 ====================

def build_assignment_choices():
    items = db.get_open_assignments()
    choices = [a["title"] for a in items]
    choices.append(PRACTICE_LABEL)
    return choices, items


def find_assignment(label, items):
    for a in items:
        if a["title"] == label:
            return a
    return None


def theme_box_html(a):
    if a is None:
        return ""
    deadline = a.get("deadline") or "不限"
    req = a.get("requirement") or ""
    req_html = (f"<div style='font-size:11px;color:#185FA5;margin-top:6px;line-height:1.6;'>{req}</div>"
                if req else "")
    return f"""
<div style='background:#E6F1FB;border-radius:8px;padding:10px 12px;margin:6px 0;'>
  <div style='font-size:10px;color:#5F7FA5;'>本次主题</div>
  <div style='font-size:14px;color:#0C447C;font-weight:500;margin-top:2px;'>{a['title']}</div>
  <div style='font-size:10px;color:#5F7FA5;margin-top:4px;'>{a['hsk_level']}　截止 {deadline}</div>
  {req_html}
</div>"""


def stu_progress_chart(rows, dim_key=None):
    if not rows:
        return None
    xs = list(range(1, len(rows) + 1))
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    if dim_key is None:
        ys = [r["score_total"] for r in rows]
        ax.set_ylim(0, 25)
        ax.set_ylabel("总分", fontsize=9)
    else:
        ys = [r[COL_MAP[dim_key]] for r in rows]
        ax.set_ylim(0, 5)
        ax.set_ylabel(dim_key, fontsize=9)
    ax.plot(xs, ys, color=BLUE, linewidth=1.8, marker="o", markersize=5)
    ax.set_xlabel("提交次序", fontsize=9)
    ax.set_xticks(xs)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:g}", (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8, color=DARK)
    return fig_to_b64(fig)


def stu_records_html(rows, is_homework):
    if not rows:
        empty = "还没有作业提交记录" if is_homework else "还没有自由练习记录"
        return f"<div style='padding:24px;text-align:center;color:#B0C4DE;font-size:12px;'>{empty}</div>"
    html = ""
    for r in rows:
        date_str = r["submitted_at"][:10]
        total = int(r["score_total"])
        sub_label = (r.get("assignment_title") or "作业") if is_homework else "自由练习"
        html += f"""
<div style='border-bottom:0.5px solid #E6F1FB;padding:10px 0;display:flex;justify-content:space-between;align-items:center;'>
  <div style='flex:1;min-width:0;'>
    <div style='font-size:13px;color:#0C447C;'>{r['essay_title']}</div>
    <div style='font-size:10px;color:#9BB5D0;margin-top:3px;'>{sub_label}　{date_str}　{r['hsk_level']}　编号 #{r['id']}</div>
  </div>
  <div style='flex-shrink:0;'>
    <span style='font-size:13px;color:#1D5FA5;font-weight:500;'>{total}/25</span>
  </div>
</div>"""
    return html


def stu_summary_html(rows):
    if not rows:
        return ""
    totals = [r["score_total"] for r in rows]
    return metric_cards([
        ("提交次数", len(rows), DARK),
        ("平均分", f"{sum(totals)/len(totals):.1f}", DARK),
        ("最高分", f"{int(max(totals))}", DARK),
    ])


# ==================== 教师端功能 ====================

def student_choices():
    items = db.get_all_students()
    return [ALL_STUDENTS] + [s["real_name"] for s in items], items


def assignment_choices():
    items = db.get_all_assignments()
    return [ALL_ASSIGN] + [a["title"] for a in items], items


def resolve_student(label):
    _, items = student_choices()
    for s in items:
        if s["real_name"] == label:
            return s["username"]
    return None


def resolve_assignment(label):
    _, items = assignment_choices()
    for a in items:
        if a["title"] == label:
            return a["id"]
    return None


def chart_class_bar(rows, dim_label):
    if not rows:
        return None
    names = [r.get("student_name", r.get("real_name", "")) for r in rows]
    if dim_label == TOTAL_DIM:
        vals = [r["score_total"] for r in rows]
        ymax = 25
    else:
        vals = [r[COL_MAP[dim_label]] for r in rows]
        ymax = 5
    fig, ax = plt.subplots(figsize=(9, 3.2))
    bars = ax.bar(names, vals, color=BLUE, edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(dim_label, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    avg = sum(vals) / len(vals)
    ax.axhline(avg, color="#E24B4A", linestyle="--", linewidth=1)
    ax.annotate(f"平均 {avg:.1f}", xy=(len(names) - 0.5, avg),
                fontsize=8, color="#E24B4A", va="bottom", ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax * 0.02,
                f"{v:g}", ha="center", fontsize=7, color=DARK)
    return fig_to_b64(fig)


def chart_trend(rows, dim_label, x_labels=None):
    if not rows:
        return None
    xs = list(range(1, len(rows) + 1))
    if dim_label == TOTAL_DIM:
        vals = [r["score_total"] for r in rows]
        ymax = 25
    else:
        vals = [r[COL_MAP[dim_label]] for r in rows]
        ymax = 5
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(xs, vals, color=BLUE, linewidth=2, marker="o", markersize=6)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(dim_label, fontsize=9)
    ax.set_xticks(xs)
    if x_labels:
        ax.set_xticklabels(x_labels, fontsize=8, rotation=30, ha="right")
    else:
        ax.set_xlabel("提交次序", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8)
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:g}", (x, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=DARK)
    return fig_to_b64(fig)


def chart_radar(dim_values, title=""):
    labels = ["内容切题", "篇章结构", "语篇连贯", "词汇运用", "语法准确"]
    vals = [dim_values[d] for d in DIMS]
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
    vals2 = vals + [vals[0]]
    angles2 = angles + angles[:1]
    fig, ax = plt.subplots(figsize=(3.4, 3.4), subplot_kw=dict(polar=True))
    ax.fill(angles2, vals2, color=BLUE, alpha=0.25)
    ax.plot(angles2, vals2, color=BLUE, linewidth=1.6, marker="o", markersize=4)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], size=7, color="#9BB5D0")
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=8.5, color=DARK)
    ax.grid(color="#C5DAFA", linewidth=0.6)
    ax.spines["polar"].set_color("#C5DAFA")
    ax.set_facecolor("#FCFDFF")
    if title:
        ax.set_title(title, size=9.5, pad=14, color="#5F7FA5")
    return fig_to_b64(fig)


def chart_dim_bars(dim_values):
    labels = DIMS[::-1]
    vals = [dim_values[d] for d in labels]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    bars = ax.barh(labels, vals, color=BLUE, height=0.55)
    ax.set_xlim(0, 5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=9)
    for b, v in zip(bars, vals):
        ax.text(v + 0.08, b.get_y() + b.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=8, color=DARK)
    return fig_to_b64(fig)


def detail_table(rows, not_submitted=None):
    if not rows and not not_submitted:
        return "<div style='padding:24px;text-align:center;color:#B0C4DE;font-size:12px;'>暂无提交记录</div>"
    html = """
<table style='width:100%;border-collapse:collapse;font-size:11px;table-layout:fixed;'>
<tr style='background:#E6F1FB;'>
  <th style='padding:7px 8px;color:#1D5FA5;text-align:center;width:8%;'>编号</th>
  <th style='padding:7px 8px;color:#1D5FA5;text-align:left;width:16%;'>学生</th>
  <th style='padding:7px 8px;color:#1D5FA5;text-align:left;width:26%;'>题目</th>
  <th style='padding:7px 8px;color:#1D5FA5;text-align:left;width:20%;'>主题</th>
  <th style='padding:7px 8px;color:#1D5FA5;text-align:center;width:8%;'>内容</th>
  <th style='padding:7px 8px;color:#1D5FA5;text-align:center;width:8%;'>结构</th>
  <th style='padding:7px 8px;color:#1D5FA5;text-align:center;width:8%;'>连贯</th>
  <th style='padding:7px 8px;color:#1D5FA5;text-align:center;width:8%;'>词汇</th>
  <th style='padding:7px 8px;color:#1D5FA5;text-align:center;width:8%;'>语法</th>
  <th style='padding:7px 8px;color:#1D5FA5;text-align:right;width:10%;'>总分</th>
</tr>"""
    for r in rows:
        name = r.get("student_name") or r.get("real_name", "")
        theme = r.get("assignment_title") or "自由练习"
        html += f"""
<tr style='border-bottom:0.5px solid #E8F1FB;'>
  <td style='padding:6px 8px;text-align:center;color:#9BB5D0;'>{r['id']}</td>
  <td style='padding:6px 8px;color:#0C447C;'>{name}</td>
  <td style='padding:6px 8px;color:#444;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{r['essay_title']}</td>
  <td style='padding:6px 8px;color:#9BB5D0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{theme}</td>
  <td style='padding:6px 8px;text-align:center;color:#444;'>{r['score_content']:.0f}</td>
  <td style='padding:6px 8px;text-align:center;color:#444;'>{r['score_structure']:.0f}</td>
  <td style='padding:6px 8px;text-align:center;color:#444;'>{r['score_coherence']:.0f}</td>
  <td style='padding:6px 8px;text-align:center;color:#444;'>{r['score_vocabulary']:.0f}</td>
  <td style='padding:6px 8px;text-align:center;color:#444;'>{r['score_grammar']:.0f}</td>
  <td style='padding:6px 8px;text-align:right;color:#1D5FA5;font-weight:500;'>{r['score_total']:.0f}</td>
</tr>"""
    if not_submitted:
        for s in not_submitted:
            html += f"""
<tr style='border-bottom:0.5px solid #E8F1FB;background:#FFFCF5;'>
  <td style='padding:6px 8px;text-align:center;color:#B0C4DE;'>—</td>
  <td style='padding:6px 8px;color:#9BB5D0;'>{s['real_name']}</td>
  <td colspan='7' style='padding:6px 8px;color:#BA7517;'>未提交</td>
  <td style='padding:6px 8px;text-align:right;color:#B0C4DE;'>—</td>
</tr>"""
    html += "</table>"
    return html


def assignment_list_html(items):
    if not items:
        return "<div style='padding:24px;text-align:center;color:#B0C4DE;font-size:12px;'>还没有发布任何作业</div>"
    html = ""
    for a in items:
        st = db.get_submission_stats(a["id"])
        total = st["已提交"] + st["未提交"]
        if a["is_open"]:
            badge = "<span style='font-size:10px;background:#EAF3DE;color:#3B6D11;padding:2px 8px;border-radius:6px;'>开放中</span>"
        else:
            badge = "<span style='font-size:10px;background:#F1EFE8;color:#5F5E5A;padding:2px 8px;border-radius:6px;'>已关闭</span>"
        deadline = a.get("deadline") or "不限"
        html += f"""
<div style='border-bottom:0.5px solid #E6F1FB;padding:10px 0;display:flex;justify-content:space-between;align-items:center;'>
  <div style='flex:1;min-width:0;'>
    <div style='font-size:13px;color:#0C447C;'>#{a['id']}　{a['title']}</div>
    <div style='font-size:10px;color:#9BB5D0;margin-top:3px;'>{a['hsk_level']}　截止 {deadline}　已交 {st['已提交']}/{total}</div>
  </div>
  <div style='flex-shrink:0;'>{badge}</div>
</div>"""
    return html


# ==================== 事件处理 ====================

def do_login(username, password):
    """统一登录，按角色分流"""
    user = db.verify_login(username, password)

    if user is None:
        return (
            None,
            gr.update(visible=True),      # 登录区
            gr.update(visible=False),     # 学生区
            gr.update(visible=False),     # 教师区
            "<div style='color:#D93025;font-size:12px;padding:6px 0;'>账号或密码不正确</div>",
            "", "",                        # 学生头部、教师头部
            gr.update(), "",               # 作业下拉、主题框
            gr.update(), gr.update(), "",  # 学生下拉、作业下拉、作业列表
        )

    header = make_header(user)

    if user["role"] == "教师":
        s_ch, _ = student_choices()
        a_ch, _ = assignment_choices()
        alist = assignment_list_html(db.get_all_assignments())
        return (
            user,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            "",
            "", header,
            gr.update(), "",
            gr.update(choices=s_ch, value=ALL_STUDENTS),
            gr.update(choices=a_ch, value=ALL_ASSIGN),
            alist,
        )

    # 学生
    choices, items = build_assignment_choices()
    first = choices[0] if choices else PRACTICE_LABEL
    a = find_assignment(first, items)
    return (
        user,
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        "",
        header, "",
        gr.update(choices=choices, value=first),
        theme_box_html(a),
        gr.update(), gr.update(), "",
    )


def do_logout():
    return (
        None,
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        "", "", "", "",
    )


def on_assignment_change(label):
    _, items = build_assignment_choices()
    if label == PRACTICE_LABEL:
        return "", gr.update(interactive=True)
    a = find_assignment(label, items)
    if a is None:
        return "", gr.update(interactive=True)
    return theme_box_html(a), gr.update(value=a["hsk_level"], interactive=False)


def do_submit(user, assign_label, title, essay, hsk_level, focus_areas):
    fail = (EMPTY_RIGHT_HTML, [(essay or "", None)], gr.update(visible=False), {}, "")

    if user is None:
        return fail + ("<div style='color:#D93025;font-size:12px;'>请先登录</div>",)
    if not title or not title.strip():
        return fail + ("<div style='color:#D93025;font-size:12px;'>请先填写作文题目</div>",)
    if not essay or len(essay.strip()) < 20:
        return fail + ("<div style='color:#D93025;font-size:12px;'>正文太短，请至少写20个字</div>",)

    _, items = build_assignment_choices()
    a = find_assignment(assign_label, items)

    if a is None:
        assignment_id, submit_type = None, "练习"
        title_for_ai = title.strip()
    else:
        assignment_id, submit_type = a["id"], "作业"
        title_for_ai = f"{title.strip()}（本次作业主题：{a['title']}）"

    results = get_feedback(title_for_ai, essay, hsk_level, focus_areas or [])
    (strengths, opening, suggestions, feedback_html,
     ref_html, standard_html, score_html, highlighted, zh_content) = results

    if strengths == "生成失败":
        return fail + ("<div style='color:#D93025;font-size:12px;'>批改失败，请稍后重试</div>",)

    dim_scores = zh_content.get("dim_scores", {})
    scores = [dim_scores.get(d, 3) for d in DIMS]

    try:
        db.save_submission(
            username=user["username"], real_name=user["real_name"],
            assignment_id=assignment_id, essay_title=title.strip(),
            essay_content=essay, hsk_level=hsk_level,
            focus_areas=focus_areas or [], scores=scores,
            feedback_data=zh_content, submit_type=submit_type,
        )
        msg = "<div style='color:#1D9E75;font-size:12px;'>批改完成，记录已保存</div>"
    except Exception as e:
        msg = f"<div style='color:#D93025;font-size:12px;'>批改完成，但保存失败：{e}</div>"

    html = build_report_html(strengths, opening, suggestions, feedback_html,
                             ref_html, standard_html, score_html)
    return html, highlighted, gr.update(visible=True), zh_content, score_html, msg


def refresh_records(user, scope, dim_label=TOTAL_DIM):
    if user is None:
        return "", "", ""
    if scope == "作业":
        rows = db.get_user_homework(user["username"])
        list_html = stu_records_html(rows, True)
    else:
        rows = db.get_user_practice(user["username"])
        list_html = stu_records_html(rows, False)
    summary = stu_summary_html(rows)
    prog = sorted(rows, key=lambda r: r["submitted_at"])
    dim_key = None if dim_label == TOTAL_DIM else dim_label
    b64 = stu_progress_chart(prog, dim_key)
    chart = img_html(b64) if b64 else "<div style='padding:30px;text-align:center;color:#B0C4DE;font-size:12px;'>暂无数据</div>"
    return summary, chart, list_html


def do_query(user, stu_label, asg_label, dim_label):
    if user is None:
        return "", "", "", ""

    username = resolve_student(stu_label) if stu_label != ALL_STUDENTS else None
    assignment_id = resolve_assignment(asg_label) if asg_label != ALL_ASSIGN else None

    # 全班 + 单次作业
    if username is None and assignment_id is not None:
        rows = db.get_latest_per_student(assignment_id)
        st = db.get_submission_stats(assignment_id)
        cards = metric_cards([
            ("已提交", st["已提交"], DARK),
            ("未提交", st["未提交"], "#BA7517" if st["未提交"] else DARK),
            ("平均分", f"{st['平均分']:.1f}", DARK),
            ("最高分", f"{st['最高分']:.0f}", DARK),
        ])
        if dim_label == ALL_DIMS:
            avg = db.get_class_dim_average(assignment_id)
            main = img_html(chart_radar(avg, "全班五维平均"), 340) if avg else img_html(None)
            sub = img_html(chart_dim_bars(avg), 560) if avg else ""
        else:
            main = img_html(chart_class_bar(rows, dim_label))
            sub = ""
        return cards, main, sub, detail_table(rows, st["未提交名单"])

    # 全班 + 全部作业
    if username is None and assignment_id is None:
        all_asg = sorted(db.get_all_assignments(), key=lambda a: a["id"])
        trend_rows, labels = [], []
        for a in all_asg:
            avg = db.get_class_dim_average(a["id"])
            if avg and avg["数量"] > 0:
                trend_rows.append({
                    "score_total": avg["总分"],
                    "score_content": avg["内容与切题"],
                    "score_structure": avg["篇章结构"],
                    "score_coherence": avg["语篇连贯性"],
                    "score_vocabulary": avg["词汇运用"],
                    "score_grammar": avg["语法准确性"],
                })
                labels.append(a["title"])

        overall = db.get_class_dim_average(None)
        cards = metric_cards([
            ("总提交数", overall["数量"], DARK),
            ("作业数", len(all_asg), DARK),
            ("班级平均", f"{overall['总分']:.1f}", DARK),
            ("学生数", len(db.get_all_students()), DARK),
        ]) if overall else ""

        if dim_label == ALL_DIMS:
            main = img_html(chart_radar(overall, "全班五维平均"), 340) if overall else img_html(None)
            sub = img_html(chart_dim_bars(overall), 560) if overall else ""
        else:
            main = img_html(chart_trend(trend_rows, dim_label, labels))
            sub = ""
        rows = db.query_submissions()
        return cards, main, sub, detail_table(rows[-30:])

    # 单个学生
    rows = db.query_submissions(username=username, assignment_id=assignment_id)
    if not rows:
        return metric_cards([("提交次数", 0, DARK)]), img_html(None), "", detail_table([])

    totals = [r["score_total"] for r in rows]
    cards = metric_cards([
        ("提交次数", len(rows), DARK),
        ("平均分", f"{sum(totals)/len(totals):.1f}", DARK),
        ("最高分", f"{max(totals):.0f}", DARK),
        ("最近一次", f"{totals[-1]:.0f}", DARK),
    ])

    if dim_label == ALL_DIMS:
        last = rows[-1]
        dv = {d: last[COL_MAP[d]] for d in DIMS}
        main = img_html(chart_radar(dv, f"{stu_label} 最近一次"), 340)
        avg_dv = {d: sum(r[COL_MAP[d]] for r in rows) / len(rows) for d in DIMS}
        sub = img_html(chart_dim_bars(avg_dv), 560)
    else:
        labels = [r["essay_title"][:8] for r in rows]
        main = img_html(chart_trend(rows, dim_label, labels))
        sub = ""
    return cards, main, sub, detail_table(rows)


def do_export(user, stu_label, asg_label):
    if user is None:
        return gr.update(visible=False)
    username = resolve_student(stu_label) if stu_label != ALL_STUDENTS else None
    assignment_id = resolve_assignment(asg_label) if asg_label != ALL_ASSIGN else None
    rows = db.query_submissions(username=username, assignment_id=assignment_id)
    if not rows:
        return gr.update(visible=False)

    data = [{
        "学生": r.get("student_name") or r.get("real_name", ""),
        "账号": r["username"],
        "作业主题": r.get("assignment_title") or "自由练习",
        "作文题目": r["essay_title"],
        "HSK等级": r["hsk_level"],
        "内容与切题": r["score_content"],
        "篇章结构": r["score_structure"],
        "语篇连贯性": r["score_coherence"],
        "词汇运用": r["score_vocabulary"],
        "语法准确性": r["score_grammar"],
        "总分": r["score_total"],
        "提交时间": r["submitted_at"],
        "作文正文": r["essay_content"],
    } for r in rows]

    fname = f"IWriting成绩_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    pd.DataFrame(data).to_excel(fname, index=False)
    return gr.update(value=fname, visible=True)


def do_publish(user, title, requirement, hsk_level, deadline):
    if user is None:
        return "<div style='color:#D93025;font-size:12px;'>请先登录</div>", "", gr.update()
    if not title or not title.strip():
        return "<div style='color:#D93025;font-size:12px;'>请填写作业主题</div>", "", gr.update()

    db.create_assignment(
        title=title.strip(), requirement=(requirement or "").strip(),
        hsk_level=hsk_level, deadline=(deadline or "").strip(),
        created_by=user["username"],
    )
    msg = f"<div style='color:#1D9E75;font-size:12px;'>作业「{title.strip()}」已发布</div>"
    a_ch, _ = assignment_choices()
    return msg, assignment_list_html(db.get_all_assignments()), gr.update(choices=a_ch)


def do_toggle(user, asg_id_text, action):
    if user is None:
        return "<div style='color:#D93025;font-size:12px;'>请先登录</div>", ""
    try:
        aid = int(str(asg_id_text).strip())
    except Exception:
        return "<div style='color:#D93025;font-size:12px;'>请输入正确的作业编号</div>", ""
    a = db.get_assignment_by_id(aid)
    if a is None:
        return f"<div style='color:#D93025;font-size:12px;'>找不到编号 {aid} 的作业</div>", ""
    db.toggle_assignment(aid, action == "开启")
    msg = f"<div style='color:#1D9E75;font-size:12px;'>作业「{a['title']}」已{action}</div>"
    return msg, assignment_list_html(db.get_all_assignments())


# ---------- 查看详情：临时令牌存储 ----------
# 说明：原实现把完整报告 HTML（约4万字符）通过 Gradio 队列随可见性切换一起推送给浏览器，
# 在本机网络环境下多次复现"内容已生成但可见性切换未完成、页面变空白"的问题（已排查排除
# v2rayN 系统代理这一单一变量后仍复现，怀疑本机网络环境存在其他干扰因素，成本过高不再深挖）。
# 现改为：可见性切换仍走 Gradio（内容很小，历次测试一直稳定）；报告正文改为浏览器对
# /detail_view/<token> 发起一次普通 HTTP GET 单独加载，从而绕开队列推送这条失败路径。
# 权限校验时机不变：仍在生成令牌之前完成（学生只能看自己的记录、教师不受限），
# 令牌本身随机不可猜测且很快过期，未通过权限校验的请求永远不会拿到令牌，
# 因此校验没有被绕开或减弱，只是把"关卡"从读取内容时挪到了发放令牌时。
_DETAIL_TOKEN_TTL_TEACHER = 1800  # 教师端令牌有效期（秒）：写评语可能需要更长时间，从10分钟延长到30分钟
_DETAIL_TOKEN_TTL_STUDENT = 600   # 学生端令牌有效期（秒）：仅查看，维持原时长不变
_detail_tokens = {}  # token -> {"html", "submission_id", "role", "username", "expires_at"}


def _store_detail_token(html_str, submission_id, role, username):
    """权限校验通过、报告内容生成完毕后调用：生成一次性令牌并存储内容。
    额外存 submission_id/role/username：
    - role/username 用于 /detail_view 路由校验"只有教师本人才能保存评语"，
      不依赖前端表单传来的任何身份信息；
    - submission_id 用于教师评语区域每次访问时现查数据库，保证保存后刷新页面能看到最新内容。
    """
    now = time.time()
    expired = [t for t, v in _detail_tokens.items() if v["expires_at"] < now]
    for t in expired:
        _detail_tokens.pop(t, None)
    token = secrets.token_urlsafe(24)
    ttl = _DETAIL_TOKEN_TTL_TEACHER if role == "教师" else _DETAIL_TOKEN_TTL_STUDENT
    _detail_tokens[token] = {
        "html": html_str,
        "submission_id": submission_id,
        "role": role,
        "username": username,
        "expires_at": now + ttl,
    }
    return token


def _render_comment_block(token, submission_id, role):
    """教师评语区域（放在AI报告下方）：
    - 教师端：返回可编辑表单，预填已有评语（如果有），普通 HTML <form> 提交，不涉及任何 JS 或
      Gradio 组件，不会重现此前"隐藏 Textbox 导致前端崩溃"的问题。
    - 学生端：只有这条记录存在教师评语时才返回带明显视觉标注的只读展示块；
      没有评语时返回空字符串，不渲染空框。
    每次调用都直接查库（不用 token 里的静态快照），保证教师保存评语后刷新页面能看到最新内容。"""
    existing = db.get_teacher_comment(submission_id)
    comment_text = (existing["comment"] if existing else "") or ""

    if role == "教师":
        return (
            "<div style='margin-top:20px;padding:14px;background:#FFF9F0;"
            "border:0.5px solid #F0D9A8;border-radius:8px;'>"
            "<div style='font-weight:500;font-size:13px;color:#B8860B;margin-bottom:8px;'>"
            "教师评语（仅教师可见与编辑，不影响AI评分与批改逻辑）</div>"
            f"<form method='post' action='/detail_view/{token}/comment'>"
            "<textarea name='comment' rows='4' style='width:100%;box-sizing:border-box;"
            "padding:8px;border:0.5px solid #C5DAFA;border-radius:6px;font-size:13px;"
            f"font-family:inherit;'>{html_escape(comment_text)}</textarea>"
            "<div style='margin-top:8px;'>"
            "<button type='submit' style='padding:6px 16px;background:#1D5FA5;color:#fff;"
            "border:none;border-radius:6px;font-size:12px;cursor:pointer;'>保存评语</button>"
            "</div></form></div>"
        )

    if comment_text.strip():
        return (
            "<div style='margin-top:20px;padding:14px;background:#F0F7F0;"
            "border:0.5px solid #B8DAC0;border-radius:8px;'>"
            "<div style='font-weight:500;font-size:13px;color:#1D9E75;margin-bottom:8px;'>"
            "📌 教师评语</div>"
            "<div style='white-space:pre-wrap;font-size:13px;line-height:1.7;color:#333;'>"
            f"{html_escape(comment_text)}</div></div>"
        )
    return ""


def show_detail(user, submission_id_text):
    """按记录编号查看详情。复用 build_report_html 渲染 feedback_json 中保存的历史反馈，
    与学生提交时看到的内容完全一致；不重新调用批改逻辑，不改动 final_agent.py。
    权限校验：学生只能查看自己的记录，教师可查看本班任意记录，必须保留，不能跳过。

    方案调整说明：原本打算用一次性令牌 + 隐藏 Textbox(visible=False) 传给
    .then(fn=None, js=...) 触发 window.open() 在新标签页自动打开报告。
    但实测确认：这个隐藏 Textbox 在 visible=False 时前端根本没有挂载对应的
    DOM 节点（用 JS 直接查询页面上所有 textbox 组件，这个隐藏组件完全不在
    其中），导致紧接着那一步"把它的值交给 js 函数"时触发 Gradio 前端自身的
    未捕获异常（Uncaught TypeError: e.i18n is not a function，来自 Gradio
    自带的 i18n-*.js），页面随即陷入"崩溃-自动重连-再崩溃"的死循环，表现为
    页面空白/卡住。这是 Gradio 这个版本的前端缺陷，不是本项目代码问题。

    因此改为：完全不再依赖隐藏组件 + js 回调这条路径，而是直接把一次性令牌
    拼进 HTML 输出里、做成一个 target="_blank" 的可点击链接，交给一直很稳定
    的 detail_msg_t/detail_msg_s（普通 HTML 组件，登录、退出登录等功能全程
    都在用，从未出过问题）显示出来，用户点一下链接即可在新标签页打开报告。
    权限校验逻辑本身一步没少、一步没变。"""
    if user is None:
        return "<div style='color:#D93025;font-size:12px;'>请先登录</div>"

    try:
        sid = int(str(submission_id_text).strip())
    except Exception:
        return "<div style='color:#D93025;font-size:12px;'>请输入正确的记录编号</div>"

    row = db.get_submission_by_id(sid)
    if row is None:
        return f"<div style='color:#D93025;font-size:12px;'>找不到编号 {sid} 的记录</div>"

    # 权限校验：学生只能看自己的记录，教师不受限
    if user["role"] == "学生" and row["username"] != user["username"]:
        return "<div style='color:#D93025;font-size:12px;'>无权查看该记录</div>"

    try:
        zh_content = json.loads(row["feedback_json"])
    except Exception:
        return "<div style='color:#D93025;font-size:12px;'>该记录数据已损坏，无法查看</div>"

    # 方案C：score_html 未存库，不重新生成雷达图，改用 metric_cards 显示分数概览
    score_cards = metric_cards([
        ("内容与切题", f"{row['score_content']:.0f}", DARK),
        ("篇章结构", f"{row['score_structure']:.0f}", DARK),
        ("语篇连贯性", f"{row['score_coherence']:.0f}", DARK),
        ("词汇运用", f"{row['score_vocabulary']:.0f}", DARK),
        ("语法准确性", f"{row['score_grammar']:.0f}", DARK),
        ("总分", f"{row['score_total']:.0f}", "#1D5FA5"),
    ])

    html = build_report_html(
        zh_content.get("strengths", ""), zh_content.get("opening", ""),
        zh_content.get("suggestions", ""), zh_content.get("feedback_html", ""),
        zh_content.get("ref_html", ""), zh_content.get("standard_html", ""),
        score_cards,
    )
    meta = (f"<div style='font-size:12px;color:#5F7FA5;margin-bottom:10px;'>"
            f"记录编号 #{row['id']}　{row.get('real_name','')}　{row['essay_title']}　{row['submitted_at'][:16]}</div>")

    # 功能：学生作文原文展示区，直接读 essay_content 字段，放在AI报告前面；
    # 转义防止作文正文里的特殊字符破坏页面布局，不涉及任何批改逻辑
    essay_html = (
        "<div style='margin-bottom:16px;'>"
        "<div style='font-weight:500;font-size:13px;color:#1D5FA5;margin-bottom:6px;'>学生作文原文</div>"
        "<div style='white-space:pre-wrap;font-size:13px;line-height:1.7;padding:12px;"
        "background:#F8FBFF;border:0.5px solid #C5DAFA;border-radius:8px;'>"
        f"{html_escape(row['essay_content'] or '')}"
        "</div></div>"
    )

    # 权限校验已在上面完成，才会走到这里生成令牌；令牌随机不可猜测、限时有效
    token = _store_detail_token(meta + essay_html + html, row["id"], user["role"], user["username"])
    link_html = (
        f"<a href='/detail_view/{token}' target='_blank' rel='noopener' "
        f"style='display:inline-block;padding:6px 14px;background:#1D5FA5;color:#fff;"
        f"border-radius:6px;font-size:12px;text-decoration:none;'>"
        f"点击在新标签页查看记录 #{row['id']} 的详情报告 →</a>"
    )
    return link_html



# ==================== 界面 ====================

APP_CSS = base_css + """
#login-card { max-width: 380px; margin: 50px auto; background: white;
    border: 0.5px solid #C5DAFA; border-radius: 12px; padding: 30px 26px; }
#login-card input { border: 0.5px solid #C5DAFA !important; border-radius: 8px !important;
    background: #F8FBFF !important; }
.card { background: white !important; border: 0.5px solid #C5DAFA !important;
    border-radius: 12px !important; padding: 16px !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), title="IWriting", css=APP_CSS) as demo:

    user_state = gr.State(None)
    zh_state = gr.State({})
    score_state = gr.State("")

    # ---------- 登录 ----------
    with gr.Column(visible=True, elem_id="login-card") as login_area:
        gr.HTML("""
<div style='text-align:center;margin-bottom:22px;'>
  <div style='font-size:20px;font-weight:500;color:#1D5FA5;'>IWriting</div>
  <div style='font-size:11px;color:#5F7FA5;margin-top:4px;'>国际中文写作智能反馈系统</div>
</div>""")
        u_in = gr.Textbox(label="账号", placeholder="请输入账号")
        p_in = gr.Textbox(label="密码", placeholder="请输入密码", type="password")
        login_msg = gr.HTML("")
        login_btn = gr.Button("登录", variant="primary")

    # ---------- 学生端 ----------
    with gr.Column(visible=False) as student_area:
        with gr.Row():
            stu_header = gr.HTML("")
            stu_logout = gr.Button("退出登录", size="sm", scale=0)

        with gr.Tabs():
            with gr.TabItem("写作批改"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, elem_classes=["input-col"]):
                        gr.HTML("<div style='padding:16px 16px 0;font-size:13px;font-weight:500;color:#1D5FA5;'>作文输入</div>")
                        assign_dd = gr.Dropdown(label="选择作业", choices=[PRACTICE_LABEL],
                                                value=PRACTICE_LABEL, interactive=True)
                        theme_html = gr.HTML("")
                        t_in = gr.Textbox(label="作文题目", placeholder="围绕主题自己拟一个题目...")
                        e_in = gr.Textbox(label="正文内容", lines=10, placeholder="请输入作文正文...")
                        with gr.Row():
                            l_in = gr.Dropdown(["HSK3", "HSK4", "HSK5", "HSK6"],
                                               label="目标HSK等级", value="HSK4")
                            f_in = gr.CheckboxGroup(DIMS, label="重点反馈维度", value=[])
                        submit_msg = gr.HTML("")
                        btn = gr.Button("提交诊断", variant="primary")
                        gr.HTML("<div style='padding:0 16px 4px;font-size:11px;font-weight:500;color:#5F7FA5;'>作文标注分析</div><div style='padding:0 16px 6px;font-size:10px;color:#9BB5D0;'>以下标注由AI自动生成，仅供参考，具体以教师反馈为准</div>")
                        annotation_out = gr.HighlightedText(
                            color_map={"超纲词汇": "#4CAF50", "词汇问题": "#FFC107", "语法问题": "#F44336"},
                            show_legend=True, visible=False, container=False,
                            elem_id="annotation-fixed")
                        gr.HTML("<div style='height:12px;'></div>")

                    with gr.Column(scale=3):
                        with gr.Row():
                            lang_zh = gr.Button("中文", size="sm", scale=1, elem_id="lang-zh")
                            lang_en = gr.Button("EN", size="sm", scale=1, elem_id="lang-en")
                            lang_ru = gr.Button("RU", size="sm", scale=1, elem_id="lang-ru")
                            lang_ar = gr.Button("AR", size="sm", scale=1, elem_id="lang-ar")
                        combined_out = gr.HTML(value=EMPTY_RIGHT_HTML)

            with gr.TabItem("我的记录"):
                with gr.Row():
                    scope_radio = gr.Radio(["作业", "自由练习"], value="作业",
                                           label="记录类型", scale=2)
                    stu_dim_dd = gr.Dropdown([TOTAL_DIM] + DIMS, value=TOTAL_DIM,
                                             label="查看维度", scale=2)
                    refresh_btn = gr.Button("刷新", size="sm", scale=1)
                stu_summary = gr.HTML("")
                gr.HTML("<div style='font-size:12px;font-weight:500;color:#0C447C;margin:12px 0 6px;'>进步曲线</div>")
                stu_chart = gr.HTML("<div style='padding:30px;text-align:center;color:#B0C4DE;font-size:12px;'>点击刷新查看</div>")
                gr.HTML("<div style='font-size:12px;font-weight:500;color:#0C447C;margin:16px 0 6px;'>历史提交</div>")
                stu_list = gr.HTML("")

                gr.HTML("<div style='height:10px;'></div>")
                with gr.Row():
                    detail_id_s = gr.Textbox(label="记录编号", placeholder="输入上方记录中的编号查看详情", scale=3)
                    detail_btn_s = gr.Button("查看详情", scale=1)
                detail_msg_s = gr.HTML("")

    # ---------- 教师端 ----------
    with gr.Column(visible=False) as teacher_area:
        with gr.Row():
            tch_header = gr.HTML("")
            tch_logout = gr.Button("退出登录", size="sm", scale=0)

        with gr.Tabs():
            with gr.TabItem("成绩统计"):
                with gr.Group(elem_classes=["card"]):
                    with gr.Row():
                        stu_dd = gr.Dropdown([ALL_STUDENTS], value=ALL_STUDENTS, label="查看对象")
                        asg_dd = gr.Dropdown([ALL_ASSIGN], value=ALL_ASSIGN, label="作业")
                        dim_dd = gr.Dropdown([TOTAL_DIM] + DIMS + [ALL_DIMS],
                                             value=TOTAL_DIM, label="维度")
                    with gr.Row():
                        query_btn = gr.Button("查询", variant="primary", scale=3)
                        export_btn = gr.Button("导出 Excel", scale=1)

                export_file = gr.File(label="下载成绩表", visible=False)
                cards_html = gr.HTML("")
                gr.HTML("<div style='height:10px;'></div>")
                main_chart = gr.HTML("<div style='padding:36px;text-align:center;color:#B0C4DE;font-size:12px;'>点击查询查看统计</div>")
                sub_chart = gr.HTML("")
                gr.HTML("<div style='font-size:12px;font-weight:500;color:#0C447C;margin:16px 0 6px;'>成绩明细</div>")
                table_html = gr.HTML("")

                gr.HTML("<div style='height:10px;'></div>")
                with gr.Row():
                    detail_id_t = gr.Textbox(label="记录编号", placeholder="输入上表中的编号查看详情", scale=3)
                    detail_btn_t = gr.Button("查看详情", scale=1)
                detail_msg_t = gr.HTML("")

            with gr.TabItem("作业管理"):
                with gr.Group(elem_classes=["card"]):
                    gr.HTML("<div style='font-size:13px;font-weight:500;color:#1D5FA5;margin-bottom:8px;'>发布新作业</div>")
                    new_title = gr.Textbox(label="作业主题", placeholder="例如：难忘的旅行")
                    new_req = gr.Textbox(label="要求说明", lines=2,
                                         placeholder="例如：围绕主题自己拟一个题目，不少于300字")
                    with gr.Row():
                        new_level = gr.Dropdown(["HSK3", "HSK4", "HSK5", "HSK6"],
                                                value="HSK4", label="目标HSK等级")
                        new_deadline = gr.Textbox(label="截止时间", placeholder="2026-09-20")
                    publish_msg = gr.HTML("")
                    publish_btn = gr.Button("发布作业", variant="primary")

                gr.HTML("<div style='height:12px;'></div>")

                with gr.Group(elem_classes=["card"]):
                    gr.HTML("<div style='font-size:13px;font-weight:500;color:#1D5FA5;margin-bottom:8px;'>开启或关闭作业</div>")
                    with gr.Row():
                        toggle_id = gr.Textbox(label="作业编号", placeholder="输入下方列表中的编号")
                        toggle_action = gr.Radio(["开启", "关闭"], value="关闭", label="操作")
                    toggle_msg = gr.HTML("")
                    toggle_btn = gr.Button("确认")

                gr.HTML("<div style='font-size:12px;font-weight:500;color:#0C447C;margin:16px 0 6px;'>已发布作业</div>")
                assign_list = gr.HTML("")

    # ---------- 事件绑定 ----------

    login_outputs = [user_state, login_area, student_area, teacher_area, login_msg,
                     stu_header, tch_header, assign_dd, theme_html,
                     stu_dd, asg_dd, assign_list]

    login_btn.click(fn=do_login, inputs=[u_in, p_in], outputs=login_outputs)
    p_in.submit(fn=do_login, inputs=[u_in, p_in], outputs=login_outputs)

    logout_outputs = [user_state, login_area, student_area, teacher_area,
                      login_msg, stu_header, tch_header, u_in]
    for lb in (stu_logout, tch_logout):
        lb.click(fn=do_logout, inputs=[], outputs=logout_outputs
                 ).then(fn=lambda: "", inputs=[], outputs=[p_in])

    assign_dd.change(fn=on_assignment_change, inputs=[assign_dd],
                     outputs=[theme_html, l_in])

    btn.click(fn=do_submit,
              inputs=[user_state, assign_dd, t_in, e_in, l_in, f_in],
              outputs=[combined_out, annotation_out, annotation_out,
                       zh_state, score_state, submit_msg])

    def _switch(lang):
        def inner(zh_content, score_html):
            if not zh_content:
                return gr.update()
            args = (zh_content.get("strengths", ""), zh_content.get("opening", ""),
                    zh_content.get("suggestions", ""), zh_content.get("feedback_html", ""),
                    zh_content.get("ref_html", ""), zh_content.get("standard_html", ""),
                    score_html)
            if lang == "zh":
                return build_report_html(*args)
            return build_translated_html(*args, lang, zh_content)
        return inner

    lang_zh.click(fn=_switch("zh"), inputs=[zh_state, score_state], outputs=[combined_out])
    lang_en.click(fn=_switch("en"), inputs=[zh_state, score_state], outputs=[combined_out])
    lang_ru.click(fn=_switch("ru"), inputs=[zh_state, score_state], outputs=[combined_out])
    lang_ar.click(fn=_switch("ar"), inputs=[zh_state, score_state], outputs=[combined_out])

    refresh_btn.click(fn=refresh_records, inputs=[user_state, scope_radio, stu_dim_dd],
                      outputs=[stu_summary, stu_chart, stu_list])
    scope_radio.change(fn=refresh_records, inputs=[user_state, scope_radio, stu_dim_dd],
                       outputs=[stu_summary, stu_chart, stu_list])
    stu_dim_dd.change(fn=refresh_records, inputs=[user_state, scope_radio, stu_dim_dd],
                      outputs=[stu_summary, stu_chart, stu_list])

    query_btn.click(fn=do_query, inputs=[user_state, stu_dd, asg_dd, dim_dd],
                    outputs=[cards_html, main_chart, sub_chart, table_html])
    for comp in (stu_dd, asg_dd, dim_dd):
        comp.change(fn=do_query, inputs=[user_state, stu_dd, asg_dd, dim_dd],
                    outputs=[cards_html, main_chart, sub_chart, table_html])

    export_btn.click(fn=do_export, inputs=[user_state, stu_dd, asg_dd],
                     outputs=[export_file])

    publish_btn.click(fn=do_publish,
                      inputs=[user_state, new_title, new_req, new_level, new_deadline],
                      outputs=[publish_msg, assign_list, asg_dd])

    toggle_btn.click(fn=do_toggle, inputs=[user_state, toggle_id, toggle_action],
                     outputs=[toggle_msg, assign_list])

    detail_btn_t.click(fn=show_detail, inputs=[user_state, detail_id_t],
                       outputs=[detail_msg_t])
    detail_btn_s.click(fn=show_detail, inputs=[user_state, detail_id_s],
                       outputs=[detail_msg_s])


if __name__ == "__main__":
    print(">>> IWriting 启动中...")
    demo.launch(share=False, prevent_thread_lock=True)

    @demo.app.get("/detail_view/{token}")
    def get_detail_view(token: str, saved: str = ""):
        """"查看详情"报告正文单独加载入口，绕开 Gradio 队列推送大段 HTML 的失败点。
        权限校验已在 show_detail() 生成令牌前完成，这里只做令牌存在性与有效期检查；
        教师评语区域每次访问都现查数据库（见 _render_comment_block），
        保证教师保存评语后刷新页面能看到最新内容。"""
        entry = _detail_tokens.get(token)
        if entry is None or time.time() > entry["expires_at"]:
            return HTMLResponse(
                "<div style='padding:20px;color:#D93025;font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;'>"
                "链接已失效，请返回后重新点击「查看详情」</div>",
                status_code=404,
            )
        comment_block = _render_comment_block(token, entry["submission_id"], entry["role"])
        saved_hint = ""
        if saved == "1" and entry["role"] == "教师":
            saved_hint = "<div style='margin-top:8px;font-size:12px;color:#1D9E75;'>已保存</div>"
        return HTMLResponse(
            "<html><head><meta charset='utf-8'></head>"
            "<body style='margin:0;padding:16px;font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;'>"
            f"{entry['html']}{comment_block}{saved_hint}</body></html>"
        )

    @demo.app.post("/detail_view/{token}/comment")
    async def post_detail_comment(token: str, comment: str = Form("")):
        """教师保存评语。权限硬校验：只有 token 当初签发给教师（role=="教师"）才允许保存，
        teacher_username 取自 token 签发时存的信息，不信任表单里的任何身份字段，
        防止绕过页面直接构造请求冒充教师身份。"""
        entry = _detail_tokens.get(token)
        if entry is None or time.time() > entry["expires_at"]:
            return HTMLResponse(
                "<div style='padding:20px;color:#D93025;font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;'>"
                "链接已失效，请返回后重新点击「查看详情」</div>",
                status_code=404,
            )
        if entry["role"] != "教师":
            return HTMLResponse(
                "<div style='padding:20px;color:#D93025;font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;'>"
                "无权保存评语</div>",
                status_code=403,
            )
        db.save_teacher_comment(entry["submission_id"], entry["username"], comment.strip())
        return RedirectResponse(url=f"/detail_view/{token}?saved=1", status_code=303)

    demo.block_thread()
