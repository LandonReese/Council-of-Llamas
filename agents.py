JUDGE_MODEL = "llama3:8b"

AGENTS = [
    {
        "name": "LOGIC-UNIT",
        "model": "llama3:8b",
        "system_prompt": "Logical",
    },
    {
        "name": "RATIONAL-PROC",
        "model": "gemma2:9b",
        "system_prompt": "Rational",
    },
    {
        "name": "ANALYZER-01",
        "model": "deepseek-coder:6.7b",
        "system_prompt": "Analytical",
    },
    {
        "name": "HYPOTHESIS-GEN",
        "model": "llama3:8b",
        "system_prompt": "Speculative",
    },
    {
        "name": "CREATIVE-MODULE",
        "model": "gemma2:9b",
        "system_prompt": "Creative",
    },
    {
        "name": "QUERY-ENGINE",
        "model": "deepseek-coder:6.7b",
        "system_prompt": "Inquisitive",
    },
    {
        "name": "EXPLORE-UNIT",
        "model": "llama3:8b",
        "system_prompt": "Curious",
    },
    {
        "name": "REFLECT-ENGINE",
        "model": "gemma2:9b",
        "system_prompt": "Reflective",
    },
    {
        "name": "PERCEPT-SYS",
        "model": "deepseek-coder:6.7b",
        "system_prompt": "Perceptive",
    },

]
