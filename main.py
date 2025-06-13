from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from openai import OpenAI

# ------------------------
# STATE TANIMI
# ------------------------
class AgentState(TypedDict, total=False):
    input: str
    next_step: str
    city: str
    weather: str
    currency: str
    final: str

# ------------------------
# OLLAMA LLM PLANNER
# ------------------------
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # sahte key

def planner_llm(state: AgentState) -> AgentState:
    prompt = f"""
Kullanıcının isteği: "{state['input']}".
Aşağıdaki 3 seçenekten hangileri yapılmalı?
- Sadece hava durumu varsa: "weather"
- Sadece döviz varsa: "currency"
- Her ikisi varsa: "weather_then_currency"
Cevap sadece bu üç seçenekten biri olmalı. Açıklama yazma.
"""

    response = client.chat.completions.create(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    import re
    step_raw = response.choices[0].message.content.strip().lower()
    match = re.search(r"(weather_then_currency|weather|currency)", step_raw)
    step = match.group(1) if match else "skip"

    return {**state, "next_step": step, "city": "İstanbul"}

# ------------------------
# TOOL: Weather
# ------------------------
def weather_tool(state: AgentState) -> AgentState:
    city = state["city"]
    return {**state, "weather": f"{city}: Güneşli 28°C ☀️"}

# ------------------------
# TOOL: Currency
# ------------------------
def currency_tool(state: AgentState) -> AgentState:
    return {**state, "currency": "USD/TRY: 32.15 ₺"}

# ------------------------
# AGENT: Summary
# ------------------------
def summarizer(state: AgentState) -> AgentState:
    return {
        "final": f"Hava durumu: {state.get('weather', '-')}, Döviz kuru: {state.get('currency', '-')}"
    }

# ------------------------
# LANGGRAPH YAPISI
# ------------------------
builder = StateGraph(AgentState)

builder.add_node("planner", RunnableLambda(planner_llm))
builder.add_node("call_weather_tool", RunnableLambda(weather_tool))
builder.add_node("call_currency_tool", RunnableLambda(currency_tool))
builder.add_node("summarizer_agent", RunnableLambda(summarizer))

builder.set_entry_point("planner")

builder.add_conditional_edges("planner", lambda x: x["next_step"], {
    "weather": "call_weather_tool",
    "currency": "call_currency_tool",
    "weather_then_currency": "call_weather_tool",
})

builder.add_edge("call_weather_tool", "call_currency_tool")
builder.add_edge("call_currency_tool", "summarizer_agent")
builder.add_edge("call_weather_tool", "summarizer_agent")
builder.add_edge("call_currency_tool", END)
builder.add_edge("summarizer_agent", END)

graph = builder.compile()

# ------------------------
# TEST: METİN GİRİŞİ
# ------------------------
if __name__ == "__main__":
    user_input = input("🗣️ Lütfen ne öğrenmek istediğinizi yazın: ")
    result = graph.invoke({"input": user_input})
    print("🧠 Yanıt:", result["final"])
