# agents.py (Simplified)

# --- CONFIGURATION ---
JUDGE_MODEL = "llama3:8b"

# --- AGENT DEFINITIONS ---
AGENTS = [
    {
        "name": "Expert 1 (Llama3)",
        "model": "llama3:8b",
        "system_prompt": "You are a broad generalist expert. Focus on providing clear, foundational, and accessible information.",
    },
    {
        "name": "Expert 2 (Gemma2)",
        "model": "gemma2:9b",
        "system_prompt": "You are a detailed analyst. Focus on specific insights, deep analysis, and precise reasoning.",
    },
    {
        "name": "Expert 3 (Deepseek-Coder)",
        "model": "deepseek-coder:6.7b",
        "system_prompt": "You are a coding and software structure specialist. Focus on code quality and technical implementation details.",
    },
    {
        "name": "Expert 4 (Llama3)",
        "model": "llama3:8b",
        "system_prompt": "You are an independent validator and general expert. Focus on fact-checking and providing an alternative, balanced perspective.",
    },
    {
        "name": "Expert 5 (Deepseek-Coder)",
        "model": "deepseek-coder:6.7b",
        "system_prompt": "You are a code optimization specialist. Focus on code performance, efficiency, and architectural patterns.",
    },
]
