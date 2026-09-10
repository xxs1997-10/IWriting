"""
IWriting 数据库初始化脚本
运行一次即可创建数据库和所有账号
如果数据库已存在，不会重复创建，也不会覆盖已有数据
"""

import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = "iwriting.db"


def hash_password(password):
    """密码哈希，不存明文"""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def init_database():
    """创建数据库和所有表"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 用户表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            real_name TEXT NOT NULL,
            role TEXT NOT NULL,
            class_name TEXT DEFAULT '默认班级',
            created_at TEXT NOT NULL
        )
    """)

    # 作业表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            requirement TEXT,
            hsk_level TEXT NOT NULL,
            deadline TEXT,
            created_by TEXT NOT NULL,
            class_name TEXT DEFAULT '默认班级',
            is_open INTEGER DEFAULT 1,
            created_at TEXT NOT NULL
        )
    """)

    # 提交记录表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            real_name TEXT NOT NULL,
            assignment_id INTEGER,
            essay_title TEXT NOT NULL,
            essay_content TEXT NOT NULL,
            hsk_level TEXT NOT NULL,
            focus_areas TEXT,
            score_content REAL,
            score_structure REAL,
            score_coherence REAL,
            score_vocabulary REAL,
            score_grammar REAL,
            score_total REAL,
            feedback_json TEXT,
            submit_type TEXT DEFAULT '练习',
            submitted_at TEXT NOT NULL
        )
    """)

    # 教师评语表（预留）
    cur.execute("""
        CREATE TABLE IF NOT EXISTS teacher_comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            submission_id INTEGER NOT NULL,
            teacher_username TEXT NOT NULL,
            comment TEXT,
            adjusted_total REAL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("数据库表创建完成")


def create_users():
    """创建教师和学生账号"""
    students = [
        ("yunliang", "yunliang123", "运良"),
        ("yinbowen", "yinbowen123", "殷博文"),
        ("huangwei", "huangwei123", "黄伟"),
        ("lihaoran", "lihaoran123", "李浩然"),
        ("liuran", "liuran123", "刘冉"),
        ("yangya", "yangya123", "杨雅"),
        ("baixinyi", "baixinyi123", "白心怡"),
        ("wangzhihu", "wangzhihu123", "王志虎"),
        ("jingcheng", "jingcheng123", "景澄"),
        ("wangleyi", "wangleyi123", "王乐一"),
        ("mashuai", "mashuai123", "马帅"),
        ("zixuan", "zixuan123", "子轩"),
        ("linan", "linan123", "林安"),
        ("mile", "mile123", "米乐"),
        ("shianhe", "shianhe123", "史安和"),
        ("mafeilong", "mafeilong123", "马飞龙"),
        ("luonate", "luonate123", "罗纳特"),
        ("liming", "liming123", "李明"),
    ]

    teacher = ("xuxiaoshan", "xuxiaoshan123", "徐晓珊")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 教师账号
    cur.execute(
        "INSERT OR IGNORE INTO users (username, password_hash, real_name, role, class_name, created_at) VALUES (?,?,?,?,?,?)",
        (teacher[0], hash_password(teacher[1]), teacher[2], "教师", "默认班级", now)
    )

    # 学生账号
    for username, password, real_name in students:
        cur.execute(
            "INSERT OR IGNORE INTO users (username, password_hash, real_name, role, class_name, created_at) VALUES (?,?,?,?,?,?)",
            (username, hash_password(password), real_name, "学生", "默认班级", now)
        )

    conn.commit()

    cur.execute("SELECT COUNT(*) FROM users WHERE role='学生'")
    student_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM users WHERE role='教师'")
    teacher_count = cur.fetchone()[0]
    conn.close()

    print(f"账号创建完成：教师 {teacher_count} 个，学生 {student_count} 个")


def show_all_users():
    """列出所有账号，方便核对"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT username, real_name, role FROM users ORDER BY role DESC, id")
    rows = cur.fetchall()
    conn.close()

    print("\n当前所有账号：")
    print("-" * 40)
    for username, real_name, role in rows:
        print(f"{role}  {real_name}  {username}")
    print("-" * 40)


if __name__ == "__main__":
    print(">>> 开始初始化 IWriting 数据库")
    init_database()
    create_users()
    show_all_users()
    print(f"\n数据库文件位置：{os.path.abspath(DB_PATH)}")
    print(">>> 初始化完成")
