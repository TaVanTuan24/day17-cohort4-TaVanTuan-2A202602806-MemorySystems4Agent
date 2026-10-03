# Day 17 Implementation: Memory Systems for AI Agent

This folder contains the complete implementation for the Day 17 Memory Systems lab.

### Key Components:
- **`model_provider.py`**: Chat model initialization and lazy-loading supporting 6 providers (`openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`).
- **`config.py`**: Centralized configuration management with provider-specific API key resolution and compact memory thresholds.
- **`memory_store.py`**: Memory engine featuring token estimation, `UserProfileStore` (`User.md` CRUD), structured entity extraction with confidence scoring, conflict handling, query isolation, and `CompactMemoryManager`.
- **`agent_baseline.py`**: Baseline Agent with within-thread memory and prompt/agent token accounting (forgets across new threads).
- **`agent_advanced.py`**: Advanced Agent with 3-tier memory (short-term thread context, persistent `User.md`, compact memory for long threads).
- **`benchmark.py`**: Rigorous evaluation harness running Standard Benchmark and Long-Context Stress Benchmark across 6 standard metrics.
- **`test_agents.py`**: Test suite covering 12 unit tests verifying memory behaviors, query detection, isolation, corrections, noise resistance, and token savings.

### How to Run:
```bash
pytest src/test_agents.py -v
python src/benchmark.py
```
