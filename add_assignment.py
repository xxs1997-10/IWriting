"""
临时工具：命令行添加作业
第三阶段做完教师端之后，这个脚本就不需要了
"""

import db_helper as db


def add(title, requirement, hsk_level, deadline):
    new_id = db.create_assignment(
        title=title,
        requirement=requirement,
        hsk_level=hsk_level,
        deadline=deadline,
        created_by="xuxiaoshan",
    )
    print(f"已添加作业 #{new_id}：{title}")


def show_all():
    items = db.get_all_assignments()
    if not items:
        print("目前没有任何作业")
        return
    print("\n当前作业列表：")
    print("-" * 50)
    for a in items:
        status = "开放中" if a["is_open"] else "已关闭"
        print(f"#{a['id']}  {a['title']}  {a['hsk_level']}  {status}")
    print("-" * 50)


if __name__ == "__main__":
    # 先建两条示例作业，方便测试
    add("难忘的旅行", "围绕主题自己拟一个题目，不少于300字，注意分段。", "HSK4", "2026-09-20")
    add("我的家乡", "介绍你的家乡，可以写风景、饮食或者人。", "HSK4", "2026-09-30")
    show_all()
