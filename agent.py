import os
import time
import itertools
import threading
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from zoneinfo import ZoneInfo

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-3.8-flash"
MAX_STEPS = 5

#------------System Prompt----------

SYSTEM_PROMPT = """You are DocPilot, a friendly and helpful assistant.

Rules:
1. Always use the calculator tool for any arithmetic. Never calculate in your head.
2. Use get_current_time whenever you need today's date or the current time.
3. Keep answers short and clear: 1 to 3 sentences unless the user asks for more.
4. Write in plain text. Do not use Markdown symbols like ** or #.
5. If you don't know something and no tool can help, say so honestly. Never make up facts.
"""

# ---------- Step 1: the tool functions (for Python) ----------

def get_current_time(timezone="local"):
    if timezone == "local":
        now = datetime.now()
    else:
        try:
            now = datetime.now(ZoneInfo(timezone))
        except Exception:
            return f"Error: unknown timezone '{timezone}'. Use names like 'Asia/Kolkata' or 'Europe/London'."
    return now.strftime("%Y-%m-%d %H:%M:%S")


def calculator(expression):
    allowed = "0123456789+-*/(). "
    for character in expression:
        if character not in allowed:
            return "Error: only numbers and + - * / ( ) are allowed."
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

# ---------- Step 2: the tool descriptions (for Gemini) ----------

time_tool = types.FunctionDeclaration(
    name="get_current_time",
    description="Get the current date and time, either locally or in a specific timezone.",
    parameters={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Timezone name like 'Asia/Kolkata' or 'Europe/London'. "
                               "Leave out to get the user's local time.",
            },
        },
    },
)

calculator_tool = types.FunctionDeclaration(
    name="calculator",
    description="Calculate a math expression exactly, e.g. '(12.5 * 4) / 3'. "
                "Use this for any arithmetic instead of calculating yourself.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression, using only numbers and + - * / ( )",
            },
        },
        "required": ["expression"],
    },
)                                                          # NEW

config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[types.Tool(function_declarations=[time_tool, calculator_tool])],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


# ---------- Step 3: the lookup table (for the loop) ----------

TOOLS = {
    "get_current_time": get_current_time,
    "calculator": calculator,                                  # NEW
}


# ---------- Talking to Gemini, with retries ----------

def ask_gemini(messages):
    wait = 2
    for attempt in range(5):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=messages,
                config=config,
            )
        except errors.APIError as e:
            if e.code not in (429, 500, 503) or attempt == 4:
                raise
            print(f"\r [Progressing (error {e.code}), retrying in {wait} seconds...]")
            time.sleep(wait)
            wait = wait * 2

#---------------animation function----------
def spinner(stop_signal):
    for shape in itertools.cycle(["▲", "■", "●"]):
        if stop_signal.is_set():
            break
        print(f"\r  {shape} thinking...", end="", flush=True)
        time.sleep(0.3)
    print("\r                    \r", end="", flush=True)
# ---------- The agent loop ----------

def run_agent(question, messages):
    messages.append(types.Content(role="user", parts=[types.Part(text=question)]))

    for step in range(MAX_STEPS):
        response = ask_gemini(messages)
        messages.append(response.candidates[0].content)

        if not response.function_calls:
            return response.text

        results = []
        for call in response.function_calls:
            print(f"\r  -> using tool: {call.name}")
            print(f"\r     with inputs: {call.args}")
            output = TOOLS[call.name](**dict(call.args or {}))
            results.append(types.Part.from_function_response(
                name=call.name, response={"result": output},
            ))
        messages.append(types.Content(role="user", parts=results))

    return "Sorry, I gave up after too many steps."

print("DocPilot ready. Type 'quit' to exit.\n")
messages = []

while True:
    question = input("You: ")
    if question.lower() in ("quit", "exit"):
        break

    stop_signal = threading.Event()
    animation = threading.Thread(target=spinner, args=(stop_signal,), daemon=True)
    animation.start()
    try:
        answer = run_agent(question, messages)
    finally:
        stop_signal.set()
        animation.join()

    print("DocPilot:", answer, "\n")
