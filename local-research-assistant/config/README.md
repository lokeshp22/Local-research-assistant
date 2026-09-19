# config/

- `.env.example` — copy to the project root as `.env` and fill in the two API keys.
  Placeholders only; never commit a real key.
- `settings.py` — every tunable in the project, read from the environment with
  defaults. Nothing else hardcodes a timeout, threshold, model name or path.

Ollama setup and pull commands are in the main [README](../README.md#1-ollama).
