from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: Baseline Agent.

    Characteristics:
    - Within-session/within-thread memory only.
    - No persistent User.md storage.
    - Strictly forgets across new threads/sessions.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not self.force_offline:
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Process an incoming turn and update token tracking."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                response = self.langchain_agent.invoke(
                    {"messages": [{"role": "user", "content": message}]},
                    config={"configurable": {"thread_id": thread_id}},
                )
                output_text = (
                    response["messages"][-1].content
                    if isinstance(response, dict) and "messages" in response
                    else str(response)
                )
                session = self.sessions.setdefault(thread_id, SessionState())
                context_messages = session.messages + [{"role": "user", "content": message}]
                turn_prompt_tokens = sum(estimate_tokens(m["content"]) for m in context_messages)
                agent_tokens = estimate_tokens(output_text)
                session.messages.append({"role": "user", "content": message})
                session.messages.append({"role": "assistant", "content": output_text})
                session.token_usage += agent_tokens
                session.prompt_tokens_processed += turn_prompt_tokens
                return {
                    "reply": output_text,
                    "agent_tokens": agent_tokens,
                    "prompt_tokens": turn_prompt_tokens,
                }
            except Exception:
                # Fall back to deterministic offline reply
                pass

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative agent response tokens for a thread or all threads."""
        if thread_id is not None:
            return self.sessions.get(thread_id, SessionState()).token_usage
        return sum(s.token_usage for s in self.sessions.values())

    def prompt_token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative prompt/context tokens processed for a thread or all threads."""
        if thread_id is not None:
            return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed
        return sum(s.prompt_tokens_processed for s in self.sessions.values())

    def compaction_count(self, thread_id: str | None = None) -> int:
        """Baseline agent does not have compact memory."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline reply logic for BaselineAgent."""
        session = self.sessions.setdefault(thread_id, SessionState())

        # Cumulative prompt accounting: all messages currently carried into this turn
        context_messages = list(session.messages)
        context_messages.append({"role": "user", "content": message})
        turn_prompt_tokens = sum(estimate_tokens(m["content"]) for m in context_messages)
        session.prompt_tokens_processed += turn_prompt_tokens

        lower_msg = message.lower()
        is_query = any(
            q in lower_msg
            for q in ["?", "gì", "đâu", "ai", "nhắc lại", "tóm tắt", "style", "ở", "nào"]
        )

        # Baseline has no cross-session knowledge; when queried in a fresh thread, it has no prior facts
        if is_query and len(session.messages) == 0:
            reply_text = (
                "Xin lỗi, mình là Baseline Agent và không có thông tin về bạn trong phiên trò chuyện này."
            )
        elif is_query:
            reply_text = "Mình đã nhận được câu hỏi trong phiên trò chuyện hiện tại."
        else:
            reply_text = "Mình đã nhận được thông tin trong phiên trò chuyện hiện tại."

        agent_tokens = estimate_tokens(reply_text)
        session.token_usage += agent_tokens
        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": reply_text})

        return {
            "reply": reply_text,
            "agent_tokens": agent_tokens,
            "prompt_tokens": turn_prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally build a live LangChain/LangGraph agent with thread checkpointing."""
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            model = build_chat_model(self.config.model)
            checkpointer = MemorySaver()
            return create_react_agent(model, tools=[], checkpointer=checkpointer)
        except Exception:
            return None
