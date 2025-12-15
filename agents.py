# agents.py (Simplified)
# Intuitive, Logical, Emotional, Rational, Empathetic, Analytical, Creative, Conjectural, Speculative, Inquisitive, Curious, Reflective, Perceptive, Reasonable, Thoughtful.

# --- CONFIGURATION ---
JUDGE_MODEL = "llama3:8b"

# --- AGENT DEFINITIONS ---
AGENTS = [
    {
        "name": "Aurora",
        "model": "llama3:8b",
        "system_prompt": "Logical",
    },
    {
        "name": "Eva",
        "model": "gemma2:9b",
        "system_prompt": "Rational",
    },
    {
        "name": "Rachel",
        "model": "deepseek-coder:6.7b",
        "system_prompt": "Analytical",
    },
    {
        "name": "Willow",
        "model": "llama3:8b",
        "system_prompt": "Speculative",
    },
    {
        "name": "Amanita",
        "model": "gemma2:9b",
        "system_prompt": "Creative",
    },
    {
        "name": "Raven",
        "model": "deepseek-coder:6.7b",
        "system_prompt": "Inquisitive",
    },
    {
        "name": "Emma",
        "model": "llama3:8b",
        "system_prompt": "Curious",
    },
    {
        "name": "Harper",
        "model": "gemma2:9b",
        "system_prompt": "Reflective",
    },
    {
        "name": "Zoe",
        "model": "deepseek-coder:6.7b",
        "system_prompt": "Perceptive",
    },

]

#    {
#        "name": "",
#        "model": "llama3:8b",
#        "system_prompt": "",
#    },
#    {
#        "name": "",
#        "model": "gemma2:9b",
#        "system_prompt": "",
#    },
#    {
#        "name": "",
#        "model": "deepseek-coder:6.7b",
#        "system_prompt": "",
#    },
