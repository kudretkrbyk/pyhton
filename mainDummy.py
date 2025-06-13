# mainDummy.py

# -*- coding: utf-8 -*-
import os
import json
import re
import unicodedata
from tools import add_task, delete_task, list_tasks
from openai import OpenAI
from dotenv import load_dotenv
def extract_first_json_block(text):
    """
    Gelen metindeki ilk geçerli JSON objesini çıkartır.
    """
    match = re.search(r'{.*}', text, re.DOTALL)
    return match.group(0) if match else None


# .env dosyasını yükle
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
print("API Key:", api_key)  # Anahtarın doğru yüklendiğini kontrol etmek için
# OpenRouter istemcisi
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# Agent sistemi
SYSTEM_PROMPT = """
Sen bir görev yöneticisi agentsın. Kullanıcıdan gelen doğal dili yorumla ve SADECE AŞAĞIDAKİ MCP FORMATINDA bir yanıt üret:

{
  "name": "add_task" | "delete_task" | "list_tasks",
  "arguments": {
    "task": "<görev açıklaması>" // sadece add_task ve delete_task için
  }
}

KURALLAR:
- Yalnızca geçerli bir JSON nesnesi döndür.
- Yanıtta açıklama, yorum, markdown veya metin kullanma.
- list_tasks komutu için "arguments" boş olmalı: {}

ÖRNEK:

{
  "name": "add_task",
  "arguments": {
    "task": "Annemin doğum günü 24.08.1991"
  }
}
"""




# Araçları haritalayan sözlük
TOOL_MAP = {
    "add_task": lambda args: add_task(args["task"]),
    "delete_task": lambda args: delete_task(args["task"]),
    "list_tasks": lambda args: list_tasks()
}

# Ana LLM çağrısı ve yönlendirme
def run_agent(user_input):
    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
    )

    content_raw = response.choices[0].message.content.strip()
    content_clean = extract_first_json_block(content_raw)  # JSON dışı metinleri ayıkla

    print("🧠 LLM Yanıtı:", content_clean)

    try:
        parsed = json.loads(content_clean)
        tool_name = parsed["name"]
        arguments = parsed.get("arguments", {})

        if tool_name in TOOL_MAP:
            result = TOOL_MAP[tool_name](arguments)
            print(result)
        else:
            print("❌ Bilinmeyen araç:", tool_name)
    except Exception as e:
        print("❌ JSON çözümlenemedi:", e)

# Giriş döngüsü
if __name__ == "__main__":
    while True:
        user_input = input("🗣️ Ne yapmak istersiniz? > ")
        user_input = unicodedata.normalize('NFKD', user_input).encode('ascii', 'ignore').decode('ascii')
        run_agent(user_input)
