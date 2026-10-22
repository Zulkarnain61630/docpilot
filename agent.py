import os
import time
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-3.8-flash"
MAX_STEPS = 5


# ---------- Step 1: the tool functions (for Python) ----------

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
    description="Get the current local date and time.",
    parameters={"type": "object", "properties": {}},
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
            print(f"  [Progressing (error {e.code}), retrying in {wait} seconds...]")
            time.sleep(wait)
            wait = wait * 2


# ---------- The agent loop ----------

def run_agent(question):
    messages = [types.Content(role="user", parts=[types.Part(text=question)])]

    for step in range(MAX_STEPS):
        response = ask_gemini(messages)
        messages.append(response.candidates[0].content)

        if not response.function_calls:
            return response.text

        results = []
        for call in response.function_calls:
            print(f"  -> using tool: {call.name}")
            print(f"     with inputs: {call.args}")
            output = TOOLS[call.name](**dict(call.args or {}))
            results.append(types.Part.from_function_response(
                name=call.name, response={"result": output},
            ))
        messages.append(types.Content(role="user", parts=results))

    return "Sorry, I gave up after too many steps."


print(run_agent("What is 17.5% of 2480?"))
print(run_agent("What is 48213 times 9177?"))
print(run_agent("How many minutes are left until midnight?"))