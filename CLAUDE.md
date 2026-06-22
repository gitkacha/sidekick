# Claude.md for Sidekick Project

## 1. Project Overview
Sidekick is a personal co-worker agent built with LangGraph and Gradio. It features:
- A main worker loop that uses tools to complete tasks
- A pre-task clarifier that asks 3 questions before execution
- Integration with OpenAI models via LangChain

## 2. Setup & Dependencies — `uv`-first

### Environment Setup
1. Install uv (if you don't have it already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create a fresh virtual environment **with uv**:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```

3. Install all project dependencies **using uv** (leverages lockfiles and fast installs):
   ```bash
   uv pip install -r requirements.txt
   ```

4. Configure runtime secrets:
   ```bash
   cp .env.example .env
   # Edit .env → set OPENAI_API_KEY, PUSHOVER_TOKEN, etc.
   ```

*All subsequent commands in this project should be executed via `uv` (e.g., `uv run <script>` or `uv pip …`). This guarantees reproducible builds and the speed benefits of uv.*

## 3. Code Structure
```
sidekick/
├── app.py            # Gradio UI entry point
├── sidekick.py       # Core LangGraph logic (state, nodes, builders)
├── sidekick_tools.py # Tool definitions (Playwright, file tools, web search)
└── sidekick_tests.py # Unit and integration tests
```

## 4. Testing
```markdown
# Run tests with uv
uv run python -m pytest sidekick/sidekick_tests.py -v
```
- Keep the same naming conventions (test classes start with **Test**).
- Use mocking (`MagicMock`, `patch`) for LLM calls.

## 5. Development Conventions
- **Naming**
  - Graph builders: `build_<name>_graph()`
  - Node functions: `verb_noun` (e.g., `ask_question`, `format_conversation`)
  - State keys: `snake_case`
- **Imports**
  - Group standard library first, then third-party
  - Avoid unused imports (they will be flagged by linting)
- **Code Style**
  - Prefer early returns over nested conditionals
  - Use `async def` for graph nodes when async operations are needed
  - Always `await` async tool calls
- **Tool Binding**
  - Bind tools to LLMs in `setup()`, not inline
  - Wrap external API calls in `try/except` with user-friendly messages

## 6. Common Patterns
- **Graph Construction**: Always use `StateGraph` with typed dicts
- **Tool Binding**: Bind tools to LLMs in `setup()`, not inline
- **Error Handling**: Wrap external API calls in `try/except` with user-friendly messages

## 7. Key Commands (uv-prefixed)
```markdown
# Run the UI
uv run python sidekick/app.py

# Run tests
uv run python -m pytest sidekick/sidekick_tests.py -v

# Clean up backups (if any)
rm -f sidekick_bkp.py app_bkp.py  # optional, kept for reference
```

## 8. Dockerization & Sandboxing
```markdown
# Build the Docker image
docker build -t sidekick:latest .

# Run the container (mount your env & data dirs)
docker run --rm -it \
  -p 7860:7860 \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/sandbox:/app/sandbox \
  sidekick:latest uv run python sidekick/app.py

# Sandbox notes
- The container runs in a read-only root filesystem except for mounted volumes.
- Secrets are mounted read-only via `.env`.
- Output data should be persisted via the mounted `sandbox` volume.
- Use `--user` or a non-root user inside the Dockerfile for stricter isolation.
```

## 9. Validation & Verification
To ensure the setup works end-to-end, follow these steps:

1. **Environment Setup**
   ```bash
   uv --version
   uv venv .venv && source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

2. **Run the Test Suite**
   ```bash
   uv run python -m pytest sidekick/sidekick_tests.py -v
   # Expected: 16 passed, 2 skipped
   ```

3. **Build & Run Docker Container**
   ```bash
   docker build -t sidekick:latest .
   docker run --rm -it -p 7860:7860 \
     -v $(pwd)/.env:/app/.env:ro \
     -v $(pwd)/sandbox:/app/sandbox \
     sidekick:latest uv run python sidekick/app.py
   ```

4. **End‑to‑End Smoke Test**
   - Open `http://localhost:7860` in a browser.
   - Interact with the Gradio UI: ask a question, answer the three clarifying prompts, and observe the final response.
   - Verify that any data written to `sandbox/` persists across container restarts.

If any step fails, review the error output, adjust the relevant configuration (e.g., missing env variables, incorrect file paths), and repeat until the flow works.

---

### How to Use This File
- **For New Contributors**: Follow the “Setup & Dependencies” section to get a working environment quickly.
- **For CI/CD Pipelines**: The Docker instructions provide a sandboxed way to run the app in a reproducible environment.
- **For Routine Maintenance**: Keep `requirements.txt` and `uv.lock` in sync; run `uv pip compile pyproject.toml -o requirements.txt` whenever you add a new dependency.

--- 

*This document serves as the single source of truth for onboarding, development, and deployment of the Sidekick project.*