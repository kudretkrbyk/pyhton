import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

class AgentState(TypedDict, total=False):
    input: str
    next_step: str
    task: str
    result: str

def planner_llm(state: AgentState) -> AgentState:
    prompt = f"""
Kullanıcının isteği: '{state['input']}'.

Aşağıdaki 3 görevden hangisi yapılmalı?
- add → yeni görev ekle
- list → görevleri göster
- remove → görev sil

Lütfen sadece JSON olarak yanıtla. Örnek:
{{"step": "add", "task": "kitap siparişi ver"}}

Eğer 'list' ya da 'remove' ise 'task' alanını boş bırak.
"""

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )

    import json
    try:
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        return {
            **state,
            "next_step": parsed.get("step", "fallback"),
            "task": parsed.get("task", None)
        }
    except Exception as e:
        return {**state, "next_step": "fallback"}
