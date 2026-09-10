"""
IWriting 第二阶段：学生端
登录 → 选作业 → 写作提交 → 我的记录

这是一个独立文件，不影响 final_agent.py
批改逻辑通过 import 复用 final_agent.py 里的 get_feedback
"""

import os
import json
import base64
from io import BytesIO
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

import db_helper as db

# 复用现有批改逻辑与界面构建函数
from final_agent import get_feedback, build_report_html, build_translated_html, EMPTY_RIGHT_HTML, css as base_css

DIMS = ["内容与切题", "篇章结构", "语篇连贯性", "词汇运用", "语法准确性"]
PRACTICE_LABEL = "自由练习"


# ---------- 工具函数 ----------

def build_assignment_choices():
    """构建作业下拉框选项，最后一项固定是自由练习"""
    items = db.get_open_assignments()
    choices = [f"{a['title']}" for a in items]
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
    req_html = f"<div style='font-size:11px;color:#185FA5;margin-top:6px;line-height:1.6;'>{req}</div>" if req else ""
    return f"""
<div style='background:#E6F1FB;border-radius:8px;padding:10px 12px;margin:6px 0;'>
  <div style='font-size:10px;color:#5F7FA5;'>本次主题</div>
  <div style='font-size:14px;color:#0C447C;font-weight:500;margin-top:2px;'>{a['title']}</div>
  <div style='font-size:10px;color:#5F7FA5;margin-top:4px;'>{a['hsk_level']}　截止 {deadline}</div>
  {req_html}
</div>"""


def make_progress_chart(rows, dim_key=None):
    """
    生成进步曲线
    dim_key 为 None 时画总分，否则画指定维度
    """
    if not rows:
        return None

    xs = list(range(1, len(rows) + 1))
    fig, ax = plt.subplots(figsize=(6.4, 2.8))

    if dim_key is None:
        ys = [r["score_total"] for r in rows]
        ax.plot(xs, ys, color="#378ADD", linewidth=1.8, marker="o", markersize=5)
        ax.set_ylim(0, 25)
        ax.set_ylabel("总分", fontsize=9)
    else:
        col_map = {
            "内容与切题": "score_content",
            "篇章结构": "score_structure",
            "语篇连贯性": "score_coherence",
            "词汇运用": "score_vocabulary",
            "语法准确性": "score_grammar",
        }
        col = col_map[dim_key]
        ys = [r[col] for r in rows]
        ax.plot(xs, ys, color="#378ADD", linewidth=1.8, marker="o", markersize=5)
        ax.set_ylim(0, 5)
        ax.set_ylabel(dim_key, fontsize=9)

    ax.set_xlabel("提交次序", fontsize=9)
    ax.set_xticks(xs)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)

    for x, y in zip(xs, ys):
        ax.annotate(f"{y:g}", (x, y), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, color="#0C447C")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


def records_list_html(rows, is_homework):
    """渲染历史记录列表"""
    if not rows:
        empty = "还没有作业提交记录" if is_homework else "还没有自由练习记录"
        return f"<div style='padding:24px;text-align:center;color:#B0C4DE;font-size:12px;'>{empty}</div>"

    html = ""
    for r in rows:
        date_str = r["submitted_at"][:10]
        total = int(r["score_total"])
        if is_homework:
            sub_label = r.get("assignment_title") or "作业"
        else:
            sub_label = "自由练习"
        html += f"""
<div style='border-bottom:0.5px solid #E6F1FB;padding:10px 0;display:flex;justify-content:space-between;align-items:center;'>
  <div style='flex:1;min-width:0;'>
    <div style='font-size:13px;color:#0C447C;'>{r['essay_title']}</div>
    <div style='font-size:10px;color:#9BB5D0;margin-top:3px;'>{sub_label}　{date_str}　{r['hsk_level']}</div>
  </div>
  <div style='display:flex;align-items:center;gap:10px;flex-shrink:0;'>
    <span style='font-size:13px;color:#1D5FA5;font-weight:500;'>{total}/25</span>
  </div>
</div>"""
    return html


def stats_summary_html(rows):
    if not rows:
        return ""
    totals = [r["score_total"] for r in rows]
    avg = sum(totals) / len(totals)
    best = max(totals)
    return f"""
<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px;'>
  <div style='background:#F8FBFF;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:10px;color:#5F7FA5;'>提交次数</div>
    <div style='font-size:20px;font-weight:500;color:#0C447C;margin-top:2px;'>{len(rows)}</div>
  </div>
  <div style='background:#F8FBFF;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:10px;color:#5F7FA5;'>平均分</div>
    <div style='font-size:20px;font-weight:500;color:#0C447C;margin-top:2px;'>{avg:.1f}</div>
  </div>
  <div style='background:#F8FBFF;border-radius:8px;padding:10px 12px;'>
    <div style='font-size:10px;color:#5F7FA5;'>最高分</div>
    <div style='font-size:20px;font-weight:500;color:#0C447C;margin-top:2px;'>{int(best)}</div>
  </div>
</div>"""


# ---------- 事件处理 ----------

def do_login(username, password):
    user = db.verify_login(username, password)
    if user is None:
        return (
            None,
            gr.update(visible=True),
            gr.update(visible=False),
            "<div style='color:#D93025;font-size:12px;padding:6px 0;'>账号或密码不正确</div>",
            gr.update(),
            "",
            "",
        )

    if user["role"] != "学生":
        return (
            None,
            gr.update(visible=True),
            gr.update(visible=False),
            "<div style='color:#D93025;font-size:12px;padding:6px 0;'>这是学生端，教师请使用教师端登录</div>",
            gr.update(),
            "",
            "",
        )

    choices, items = build_assignment_choices()
    first = choices[0] if choices else PRACTICE_LABEL
    a = find_assignment(first, items)

    header = f"""
<div style='background:#1D5FA5;padding:14px 20px;border-radius:12px;display:flex;justify-content:space-between;align-items:center;'>
  <div>
    <div style='color:white;font-size:16px;font-weight:500;'>IWriting 写作反馈</div>
    <div style='color:#B5D4F4;font-size:11px;margin-top:2px;'>{user['real_name']}　{user['class_name']}</div>
  </div>
</div>"""

    return (
        user,
        gr.update(visible=False),
        gr.update(visible=True),
        "",
        gr.update(choices=choices, value=first),
        header,
        theme_box_html(a),
    )


def do_logout():
    return (
        None,
        gr.update(visible=True),
        gr.update(visible=False),
        "",
        "",
        "",
        "",
    )


def on_assignment_change(label):
    """切换作业时更新主题框和HSK等级"""
    _, items = build_assignment_choices()
    if label == PRACTICE_LABEL:
        return "", gr.update(interactive=True)
    a = find_assignment(label, items)
    if a is None:
        return "", gr.update(interactive=True)
    return theme_box_html(a), gr.update(value=a["hsk_level"], interactive=False)


def do_submit(user, assign_label, title, essay, hsk_level, focus_areas):
    """提交批改并存库"""
    if user is None:
        return (EMPTY_RIGHT_HTML, [(essay or "", None)], gr.update(visible=False), {}, "",
                "<div style='color:#D93025;font-size:12px;'>请先登录</div>")

    if not title or not title.strip():
        return (EMPTY_RIGHT_HTML, [(essay or "", None)], gr.update(visible=False), {}, "",
                "<div style='color:#D93025;font-size:12px;'>请先填写作文题目</div>")

    if not essay or len(essay.strip()) < 20:
        return (EMPTY_RIGHT_HTML, [(essay or "", None)], gr.update(visible=False), {}, "",
                "<div style='color:#D93025;font-size:12px;'>正文太短，请至少写20个字</div>")

    _, items = build_assignment_choices()
    a = find_assignment(assign_label, items)

    if a is None:
        assignment_id = None
        submit_type = "练习"
        full_title = title.strip()
    else:
        assignment_id = a["id"]
        submit_type = "作业"
        full_title = title.strip()

    # 把主题信息拼进标题传给AI，用于内容与切题维度判断
    if a is not None:
        title_for_ai = f"{title.strip()}（本次作业主题：{a['title']}）"
    else:
        title_for_ai = title.strip()

    results = get_feedback(title_for_ai, essay, hsk_level, focus_areas or [])

    strengths, opening, suggestions, feedback_html, ref_html, standard_html, score_html, highlighted, zh_content = results

    if strengths == "生成失败":
        return (EMPTY_RIGHT_HTML, [(essay, None)], gr.update(visible=False), {}, "",
                "<div style='color:#D93025;font-size:12px;'>批改失败，请稍后重试</div>")

    dim_scores = zh_content.get("dim_scores", {})
    scores = [dim_scores.get(d, 3) for d in DIMS]

    try:
        db.save_submission(
            username=user["username"],
            real_name=user["real_name"],
            assignment_id=assignment_id,
            essay_title=full_title,
            essay_content=essay,
            hsk_level=hsk_level,
            focus_areas=focus_areas or [],
            scores=scores,
            feedback_data=zh_content,
            submit_type=submit_type,
        )
        save_msg = "<div style='color:#1D9E75;font-size:12px;'>批改完成，记录已保存</div>"
    except Exception as e:
        save_msg = f"<div style='color:#D93025;font-size:12px;'>批改完成，但保存失败：{e}</div>"

    html = build_report_html(strengths, opening, suggestions, feedback_html,
                             ref_html, standard_html, score_html)

    return html, highlighted, gr.update(visible=True), zh_content, score_html, save_msg


def refresh_records(user, scope):
    """刷新我的记录"""
    if user is None:
        return "", "", ""

    if scope == "作业":
        rows = db.get_user_homework(user["username"])
        list_html = records_list_html(rows, True)
    else:
        rows = db.get_user_practice(user["username"])
        list_html = records_list_html(rows, False)

    summary = stats_summary_html(rows)

    prog_rows = sorted(rows, key=lambda r: r["submitted_at"])
    b64 = make_progress_chart(prog_rows, None)
    if b64:
        chart_html = f"<img src='data:image/png;base64,{b64}' style='width:100%;'/>"
    else:
        chart_html = "<div style='padding:30px;text-align:center;color:#B0C4DE;font-size:12px;'>暂无数据</div>"

    return summary, chart_html, list_html


def refresh_assignments(user):
    choices, items = build_assignment_choices()
    first = choices[0] if choices else PRACTICE_LABEL
    a = find_assignment(first, items)
    return gr.update(choices=choices, value=first), theme_box_html(a)


# ---------- 界面 ----------

EXTRA_CSS = base_css + """
#login-card { max-width: 380px; margin: 50px auto; background: white;
    border: 0.5px solid #C5DAFA; border-radius: 12px; padding: 30px 26px; }
#login-card input { border: 0.5px solid #C5DAFA !important; border-radius: 8px !important;
    background: #F8FBFF !important; }
.panel-card { background: white; border: 0.5px solid #C5DAFA;
    border-radius: 12px; padding: 16px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), title="IWriting 学生端", css=EXTRA_CSS) as demo:

    user_state = gr.State(None)
    zh_state = gr.State({})
    score_state = gr.State("")

    # ===== 登录 =====
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

    # ===== 主界面 =====
    with gr.Column(visible=False) as main_area:
        with gr.Row():
            header_html = gr.HTML("")
            logout_btn = gr.Button("退出登录", size="sm", scale=0)

        with gr.Tabs():
            # ---- 写作批改 ----
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

            # ---- 我的记录 ----
            with gr.TabItem("我的记录") as record_tab:
                with gr.Row():
                    scope_radio = gr.Radio(["作业", "自由练习"], value="作业",
                                           label="记录类型", scale=3)
                    refresh_btn = gr.Button("刷新", size="sm", scale=1)

                summary_html = gr.HTML("")

                gr.HTML("<div style='font-size:12px;font-weight:500;color:#0C447C;margin:12px 0 6px;'>进步曲线</div>")
                chart_html = gr.HTML("<div style='padding:30px;text-align:center;color:#B0C4DE;font-size:12px;'>点击刷新查看</div>")

                gr.HTML("<div style='font-size:12px;font-weight:500;color:#0C447C;margin:16px 0 6px;'>历史提交</div>")
                list_html = gr.HTML("")

    # ===== 事件 =====

    login_btn.click(
        fn=do_login,
        inputs=[u_in, p_in],
        outputs=[user_state, login_area, main_area, login_msg, assign_dd, header_html, theme_html],
    )
    p_in.submit(
        fn=do_login,
        inputs=[u_in, p_in],
        outputs=[user_state, login_area, main_area, login_msg, assign_dd, header_html, theme_html],
    )

    logout_btn.click(
        fn=do_logout,
        inputs=[],
        outputs=[user_state, login_area, main_area, login_msg, header_html, theme_html, u_in],
    ).then(fn=lambda: "", inputs=[], outputs=[p_in])

    assign_dd.change(
        fn=on_assignment_change,
        inputs=[assign_dd],
        outputs=[theme_html, l_in],
    )

    btn.click(
        fn=do_submit,
        inputs=[user_state, assign_dd, t_in, e_in, l_in, f_in],
        outputs=[combined_out, annotation_out, annotation_out, zh_state, score_state, submit_msg],
    )

    def _switch(lang):
        def inner(zh_content, score_html):
            if not zh_content:
                return gr.update()
            if lang == "zh":
                return build_report_html(
                    zh_content.get("strengths", ""), zh_content.get("opening", ""),
                    zh_content.get("suggestions", ""), zh_content.get("feedback_html", ""),
                    zh_content.get("ref_html", ""), zh_content.get("standard_html", ""),
                    score_html)
            return build_translated_html(
                zh_content.get("strengths", ""), zh_content.get("opening", ""),
                zh_content.get("suggestions", ""), zh_content.get("feedback_html", ""),
                zh_content.get("ref_html", ""), zh_content.get("standard_html", ""),
                score_html, lang, zh_content)
        return inner

    lang_zh.click(fn=_switch("zh"), inputs=[zh_state, score_state], outputs=[combined_out])
    lang_en.click(fn=_switch("en"), inputs=[zh_state, score_state], outputs=[combined_out])
    lang_ru.click(fn=_switch("ru"), inputs=[zh_state, score_state], outputs=[combined_out])
    lang_ar.click(fn=_switch("ar"), inputs=[zh_state, score_state], outputs=[combined_out])

    refresh_btn.click(
        fn=refresh_records,
        inputs=[user_state, scope_radio],
        outputs=[summary_html, chart_html, list_html],
    )
    scope_radio.change(
        fn=refresh_records,
        inputs=[user_state, scope_radio],
        outputs=[summary_html, chart_html, list_html],
    )


if __name__ == "__main__":
    print(">>> IWriting 学生端启动中...")
    demo.launch(share=False)
