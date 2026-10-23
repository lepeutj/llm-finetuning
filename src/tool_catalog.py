"""Tool schemas and deterministic local implementations used by the benchmark."""

TOOLS = [
    {"name": "get_weather", "description": "Get weather for a city on a date.",
     "parameters": {"type": "object", "properties": {
         "city": {"type": "string"}, "date": {"type": "string"}},
         "required": ["city", "date"], "additionalProperties": False}},
    {"name": "convert_currency", "description": "Convert an amount between currencies.",
     "parameters": {"type": "object", "properties": {
         "amount": {"type": "number"}, "from_currency": {"type": "string"},
         "to_currency": {"type": "string"}},
         "required": ["amount", "from_currency", "to_currency"], "additionalProperties": False}},
    {"name": "search_web", "description": "Search the web for current information.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
         "required": ["query"], "additionalProperties": False}},
    {"name": "calculate", "description": "Evaluate a basic arithmetic expression.",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
         "required": ["expression"], "additionalProperties": False}},
    {"name": "find_files", "description": "Find files by name pattern in a directory.",
     "parameters": {"type": "object", "properties": {
         "pattern": {"type": "string"}, "directory": {"type": "string"}},
         "required": ["pattern", "directory"], "additionalProperties": False}},
]

TOOL_MAP = {tool["name"]: tool for tool in TOOLS}


def execute(name, arguments):
    """Return a stable simulated result; no network or filesystem access occurs."""
    if name == "get_weather":
        return {"city": arguments["city"], "date": arguments["date"], "temperature_c": 18,
                "condition": "partly cloudy"}
    if name == "convert_currency":
        return {"amount": round(float(arguments["amount"]) * 0.92, 2),
                "currency": arguments["to_currency"], "rate": 0.92}
    if name == "search_web":
        return {"results": [{"title": "Example result", "url": "https://example.com"}]}
    if name == "calculate":
        allowed = set("0123456789+-*/(). ")
        expression = arguments["expression"]
        if not set(expression) <= allowed:
            return {"error": "unsupported expression"}
        return {"result": eval(expression, {"__builtins__": {}}, {})}  # controlled generated input
    if name == "find_files":
        return {"matches": [f"{arguments['directory'].rstrip('/')}/example.txt"]}
    raise KeyError(f"Unknown tool: {name}")
