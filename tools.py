# tools.py

tasks = []

def add_task(task):
    tasks.append(task)
    return f"✅ Görev eklendi: {task}"

def delete_task(task):
    if task in tasks:
        tasks.remove(task)
        return f"🗑️ Görev silindi: {task}"
    else:
        return f"⚠️ Görev bulunamadı: {task}"

def list_tasks():
    if not tasks:
        return "📭 Henüz görev yok."
    return "📋 Görevler:\n" + "\n".join([f"- {t}" for t in tasks])
