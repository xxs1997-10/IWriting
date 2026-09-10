"""
IWriting 数据库操作模块
所有与数据库交互的函数都在这里
"""

import sqlite3
import hashlib
import json
from datetime import datetime

DB_PATH = "iwriting.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


# ---------- 登录验证 ----------

def verify_login(username, password):
    """
    验证登录
    成功返回 {'username':..., 'real_name':..., 'role':..., 'class_name':...}
    失败返回 None
    """
    if not username or not password:
        return None

    username = username.strip().lower()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        return None
    if row["password_hash"] != hash_password(password):
        return None

    return {
        "username": row["username"],
        "real_name": row["real_name"],
        "role": row["role"],
        "class_name": row["class_name"],
    }


# ---------- 作业管理 ----------

def create_assignment(title, requirement, hsk_level, deadline, created_by, class_name="默认班级"):
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        """INSERT INTO assignments (title, requirement, hsk_level, deadline, created_by, class_name, is_open, created_at)
           VALUES (?,?,?,?,?,?,1,?)""",
        (title, requirement, hsk_level, deadline, created_by, class_name, now)
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_open_assignments(class_name="默认班级"):
    """学生端用：获取当前开放的作业列表"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM assignments WHERE is_open=1 AND class_name=? ORDER BY created_at DESC",
        (class_name,)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_assignments(class_name="默认班级"):
    """教师端用：获取所有作业，包括已关闭的"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM assignments WHERE class_name=? ORDER BY created_at DESC",
        (class_name,)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def toggle_assignment(assignment_id, is_open):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE assignments SET is_open=? WHERE id=?", (1 if is_open else 0, assignment_id))
    conn.commit()
    conn.close()


# ---------- 提交记录 ----------

def save_submission(username, real_name, assignment_id, essay_title, essay_content,
                    hsk_level, focus_areas, scores, feedback_data, submit_type="练习"):
    """
    保存一次提交
    scores 是五个维度的分数列表 [内容, 结构, 连贯, 词汇, 语法]
    feedback_data 是完整的反馈内容字典，存成 JSON
    """
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    focus_str = "、".join(focus_areas) if isinstance(focus_areas, list) else str(focus_areas or "")

    cur.execute(
        """INSERT INTO submissions
           (username, real_name, assignment_id, essay_title, essay_content, hsk_level, focus_areas,
            score_content, score_structure, score_coherence, score_vocabulary, score_grammar,
            score_total, feedback_json, submit_type, submitted_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (username, real_name, assignment_id, essay_title, essay_content, hsk_level, focus_str,
         float(scores[0]), float(scores[1]), float(scores[2]), float(scores[3]), float(scores[4]),
         float(sum(scores)), json.dumps(feedback_data, ensure_ascii=False), submit_type, now)
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_user_submissions(username, limit=None):
    """获取某个学生的所有提交记录，最新的在前"""
    conn = get_conn()
    cur = conn.cursor()
    sql = "SELECT * FROM submissions WHERE username=? ORDER BY submitted_at DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql, (username,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_homework(username):
    """只取该学生的作业提交，带上作业主题"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.*, a.title AS assignment_title
        FROM submissions s
        LEFT JOIN assignments a ON s.assignment_id = a.id
        WHERE s.username=? AND s.submit_type='作业'
        ORDER BY s.submitted_at DESC
    """, (username,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_practice(username):
    """只取该学生的自由练习"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM submissions
        WHERE username=? AND submit_type='练习'
        ORDER BY submitted_at DESC
    """, (username,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_progress(username, scope="全部"):
    """
    取进步曲线数据，按时间正序
    scope 可以是 全部 / 作业 / 练习
    """
    conn = get_conn()
    cur = conn.cursor()
    if scope == "作业":
        cur.execute("""SELECT * FROM submissions WHERE username=? AND submit_type='作业'
                       ORDER BY submitted_at ASC""", (username,))
    elif scope == "练习":
        cur.execute("""SELECT * FROM submissions WHERE username=? AND submit_type='练习'
                       ORDER BY submitted_at ASC""", (username,))
    else:
        cur.execute("""SELECT * FROM submissions WHERE username=?
                       ORDER BY submitted_at ASC""", (username,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_assignment_by_id(assignment_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM assignments WHERE id=?", (assignment_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_submission_by_id(submission_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM submissions WHERE id=?", (submission_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_submissions(class_name="默认班级"):
    """教师端用：获取全班所有提交记录"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.* FROM submissions s
        JOIN users u ON s.username = u.username
        WHERE u.class_name = ?
        ORDER BY s.submitted_at DESC
    """, (class_name,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_assignment_submissions(assignment_id):
    """获取某次作业的所有提交"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM submissions WHERE assignment_id=? ORDER BY submitted_at DESC",
        (assignment_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- 用户管理 ----------

def get_all_students(class_name="默认班级"):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT username, real_name, class_name FROM users WHERE role='学生' AND class_name=? ORDER BY id",
        (class_name,)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_not_submitted_students(assignment_id, class_name="默认班级"):
    """查出某次作业还没交的学生"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT username, real_name FROM users
        WHERE role='学生' AND class_name=?
          AND username NOT IN (
              SELECT DISTINCT username FROM submissions WHERE assignment_id=?
          )
        ORDER BY id
    """, (class_name, assignment_id))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- 教师端统计 ----------

def query_submissions(username=None, assignment_id=None, class_name="默认班级"):
    """
    教师端统一查询入口
    username 为 None 表示全班；assignment_id 为 None 表示全部作业
    """
    conn = get_conn()
    cur = conn.cursor()

    sql = """
        SELECT s.*, a.title AS assignment_title, u.real_name AS student_name
        FROM submissions s
        LEFT JOIN assignments a ON s.assignment_id = a.id
        JOIN users u ON s.username = u.username
        WHERE u.class_name = ?
    """
    params = [class_name]

    if username:
        sql += " AND s.username = ?"
        params.append(username)
    if assignment_id:
        sql += " AND s.assignment_id = ?"
        params.append(assignment_id)

    sql += " ORDER BY s.submitted_at ASC"
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_per_student(assignment_id, class_name="默认班级"):
    """
    某次作业每个学生的最后一次提交
    用于全班成绩对比，避免重复提交的学生被算多次
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.*, u.real_name AS student_name
        FROM submissions s
        JOIN users u ON s.username = u.username
        WHERE s.assignment_id = ? AND u.class_name = ?
          AND s.id IN (
              SELECT MAX(id) FROM submissions
              WHERE assignment_id = ? GROUP BY username
          )
        ORDER BY u.id
    """, (assignment_id, class_name, assignment_id))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_class_dim_average(assignment_id=None, class_name="默认班级"):
    """全班五维平均分"""
    conn = get_conn()
    cur = conn.cursor()
    sql = """
        SELECT AVG(s.score_content) c, AVG(s.score_structure) st,
               AVG(s.score_coherence) co, AVG(s.score_vocabulary) v,
               AVG(s.score_grammar) g, AVG(s.score_total) t, COUNT(*) n
        FROM submissions s
        JOIN users u ON s.username = u.username
        WHERE u.class_name = ?
    """
    params = [class_name]
    if assignment_id:
        sql += " AND s.assignment_id = ?"
        params.append(assignment_id)
    cur.execute(sql, params)
    row = cur.fetchone()
    conn.close()
    if row is None or row["n"] == 0:
        return None
    return {
        "内容与切题": row["c"] or 0,
        "篇章结构": row["st"] or 0,
        "语篇连贯性": row["co"] or 0,
        "词汇运用": row["v"] or 0,
        "语法准确性": row["g"] or 0,
        "总分": row["t"] or 0,
        "数量": row["n"],
    }


def get_submission_stats(assignment_id, class_name="默认班级"):
    """某次作业的提交情况统计"""
    submitted = get_latest_per_student(assignment_id, class_name)
    not_sub = get_not_submitted_students(assignment_id, class_name)
    totals = [r["score_total"] for r in submitted]
    return {
        "已提交": len(submitted),
        "未提交": len(not_sub),
        "未提交名单": not_sub,
        "平均分": (sum(totals) / len(totals)) if totals else 0,
        "最高分": max(totals) if totals else 0,
        "最低分": min(totals) if totals else 0,
    }
