"""
IWriting 第三阶段：教师端
成绩统计 + 作业管理

独立文件，不影响 final_agent.py 和 student_portal.py
"""

import os
import base64
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

import db_helper as db

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

DIMS = ["内容与切题", "篇章结构", "语篇连贯性", "词汇运用", "语法准确性"]
COL_MAP = {
    "内容与切题": "score_content",
    "篇章结构": "score_structure",
    "语篇连贯性": "score_coherence",
    "词汇运用": "score_vocabulary",
    "语法准确性": "score_grammar",
}
ALL_STUDENTS = "全班"
ALL_ASSIGN = "全部作业"
TOTAL_DIM = "总分"
ALL_DIMS = "五维全部"

BLUE = "#378ADD"
DARK = "#0C447C"


# ---------- 下拉选项 ----------

def student_choices():
    items = db.get_all_students()
    return [ALL_STUDENTS] + [f"{s['real_name']}" for s in items], items


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


# ---------- 图表 ----------

def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def chart_class_bar(rows, dim_label):
    """全班单次作业柱状图"""
    if not rows:
        return None
    names = [r.get("student_name", r["real_name"]) for r in rows]
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
    """趋势折线图"""
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
    """五维雷达图"""
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
    ax.set_xticklabels(labels, size=8.5, color="#0C447C")
    ax.grid(color="#C5DAFA", linewidth=0.6)
    ax.spines["polar"].set_color("#C5DAFA")
    ax.set_facecolor("#FCFDFF")
    if title:
        ax.set_title(title, size=9.5, pad=14, color="#5F7FA5")
    return fig_to_b64(fig)


def chart_dim_bars(dim_values):
    """五维横向条形图"""
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


def img_html(b64, max_width=None):
    if not b64:
        return "<div style='padding:36px;text-align:center;color:#B0C4DE;font-size:12px;'>暂无数据</div>"
    if max_width:
        return (f"<div style='text-align:center;'>"
                f"<img src='data:image/png;base64,{b64}' "
                f"style='width:100%;max-width:{max_width}px;'/></div>")
    return f"<img src='data:image/png;base64,{b64}' style='width:100%;'/>"


# ---------- HTML 片段 ----------

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


def detail_table(rows, not_submitted=None):
    if not rows and not not_submitted:
        return "<div style='padding:24px;text-align:center;color:#B0C4DE;font-size:12px;'>暂无提交记录</div>"

    html = """
<table style='width:100%;border-collapse:collapse;font-size:11px;table-layout:fixed;'>
<tr style='background:#E6F1FB;'>
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


# ---------- 事件 ----------

def do_login(username, password):
    user = db.verify_login(username, password)
    if user is None:
        return (None, gr.update(visible=True), gr.update(visible=False),
                "<div style='color:#D93025;font-size:12px;padding:6px 0;'>账号或密码不正确</div>",
                "", gr.update(), gr.update(), "")

    if user["role"] != "教师":
        return (None, gr.update(visible=True), gr.update(visible=False),
                "<div style='color:#D93025;font-size:12px;padding:6px 0;'>这是教师端，学生请使用学生端登录</div>",
                "", gr.update(), gr.update(), "")

    header = f"""
<div style='background:#1D5FA5;padding:14px 20px;border-radius:12px;display:flex;justify-content:space-between;align-items:center;'>
  <div>
    <div style='color:white;font-size:16px;font-weight:500;'>IWriting 教师端</div>
    <div style='color:#B5D4F4;font-size:11px;margin-top:2px;'>{user['real_name']}　{user['class_name']}</div>
  </div>
</div>"""

    s_ch, _ = student_choices()
    a_ch, _ = assignment_choices()
    alist = assignment_list_html(db.get_all_assignments())

    return (user, gr.update(visible=False), gr.update(visible=True), "",
            header, gr.update(choices=s_ch, value=ALL_STUDENTS),
            gr.update(choices=a_ch, value=ALL_ASSIGN), alist)


def do_logout():
    return (None, gr.update(visible=True), gr.update(visible=False), "", "", "")


def do_query(user, stu_label, asg_label, dim_label):
    """核心统计查询，根据三个筛选条件组合出不同的图表"""
    if user is None:
        return "", "", "", ""

    username = resolve_student(stu_label) if stu_label != ALL_STUDENTS else None
    assignment_id = resolve_assignment(asg_label) if asg_label != ALL_ASSIGN else None

    # 情况一：全班 + 单次作业 → 柱状图对比
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
        table = detail_table(rows, st["未提交名单"])
        return cards, main, sub, table

    # 情况二：全班 + 全部作业 → 各次作业平均分趋势
    if username is None and assignment_id is None:
        all_asg = db.get_all_assignments()
        all_asg = sorted(all_asg, key=lambda a: a["id"])
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
        if overall:
            cards = metric_cards([
                ("总提交数", overall["数量"], DARK),
                ("作业数", len(all_asg), DARK),
                ("班级平均", f"{overall['总分']:.1f}", DARK),
                ("学生数", len(db.get_all_students()), DARK),
            ])
        else:
            cards = ""

        if dim_label == ALL_DIMS:
            main = img_html(chart_radar(overall, "全班五维平均"), 340) if overall else img_html(None)
            sub = img_html(chart_dim_bars(overall), 560) if overall else ""
        else:
            main = img_html(chart_trend(trend_rows, dim_label, labels))
            sub = ""

        rows = db.query_submissions()
        return cards, main, sub, detail_table(rows[-30:])

    # 情况三：单个学生 → 进步趋势
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

    data = []
    for r in rows:
        data.append({
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
        })

    df = pd.DataFrame(data)
    fname = f"IWriting成绩_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    df.to_excel(fname, index=False)
    return gr.update(value=fname, visible=True)


def do_publish(user, title, requirement, hsk_level, deadline):
    if user is None:
        return "<div style='color:#D93025;font-size:12px;'>请先登录</div>", "", gr.update()
    if not title or not title.strip():
        return "<div style='color:#D93025;font-size:12px;'>请填写作业主题</div>", "", gr.update()

    db.create_assignment(
        title=title.strip(),
        requirement=(requirement or "").strip(),
        hsk_level=hsk_level,
        deadline=(deadline or "").strip(),
        created_by=user["username"],
    )
    msg = f"<div style='color:#1D9E75;font-size:12px;'>作业「{title.strip()}」已发布</div>"
    alist = assignment_list_html(db.get_all_assignments())
    a_ch, _ = assignment_choices()
    return msg, alist, gr.update(choices=a_ch)


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


# ---------- 界面 ----------

CSS = """
.gradio-container { background: #F0F6FF !important; }
#login-card { max-width: 380px; margin: 50px auto; background: white;
    border: 0.5px solid #C5DAFA; border-radius: 12px; padding: 30px 26px; }
#login-card input { border: 0.5px solid #C5DAFA !important; border-radius: 8px !important;
    background: #F8FBFF !important; }
.card { background: white !important; border: 0.5px solid #C5DAFA !important;
    border-radius: 12px !important; padding: 16px !important; }
.gr-button-primary { background: #1D5FA5 !important; border: none !important;
    border-radius: 8px !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), title="IWriting 教师端", css=CSS) as demo:

    user_state = gr.State(None)

    # ===== 登录 =====
    with gr.Column(visible=True, elem_id="login-card") as login_area:
        gr.HTML("""
<div style='text-align:center;margin-bottom:22px;'>
  <div style='font-size:20px;font-weight:500;color:#1D5FA5;'>IWriting 教师端</div>
  <div style='font-size:11px;color:#5F7FA5;margin-top:4px;'>成绩统计与作业管理</div>
</div>""")
        u_in = gr.Textbox(label="账号", placeholder="请输入教师账号")
        p_in = gr.Textbox(label="密码", placeholder="请输入密码", type="password")
        login_msg = gr.HTML("")
        login_btn = gr.Button("登录", variant="primary")

    # ===== 主界面 =====
    with gr.Column(visible=False) as main_area:
        with gr.Row():
            header_html = gr.HTML("")
            logout_btn = gr.Button("退出登录", size="sm", scale=0)

        with gr.Tabs():
            # ---- 成绩统计 ----
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

            # ---- 作业管理 ----
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

    # ===== 事件 =====

    login_btn.click(
        fn=do_login, inputs=[u_in, p_in],
        outputs=[user_state, login_area, main_area, login_msg,
                 header_html, stu_dd, asg_dd, assign_list])
    p_in.submit(
        fn=do_login, inputs=[u_in, p_in],
        outputs=[user_state, login_area, main_area, login_msg,
                 header_html, stu_dd, asg_dd, assign_list])

    logout_btn.click(
        fn=do_logout, inputs=[],
        outputs=[user_state, login_area, main_area, login_msg, header_html, u_in]
    ).then(fn=lambda: "", inputs=[], outputs=[p_in])

    for comp in (query_btn,):
        comp.click(fn=do_query, inputs=[user_state, stu_dd, asg_dd, dim_dd],
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


if __name__ == "__main__":
    print(">>> IWriting 教师端启动中...")
    demo.launch(share=False, server_port=7861)
