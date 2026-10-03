from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates, is_query_message
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
                pass

        return self._reply_offline(thread_id, message)

    def _session_facts(self, thread_id: str) -> dict[str, str]:
        """Extract structured facts strictly from prior user messages in this thread only."""
        session = self.sessions.get(thread_id)
        if not session:
            return {}
        thread_facts: dict[str, str] = {}
        for m in session.messages:
            if m.get("role") == "user":
                updates = extract_profile_updates(m.get("content", ""))
                thread_facts.update(updates)
        return thread_facts

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

        # Cumulative prompt accounting: all messages carried into this turn
        context_messages = list(session.messages)
        context_messages.append({"role": "user", "content": message})
        turn_prompt_tokens = sum(estimate_tokens(m["content"]) for m in context_messages)
        session.prompt_tokens_processed += turn_prompt_tokens

        is_query = is_query_message(message)
        prior_facts = self._session_facts(thread_id)
        lower_msg = message.lower()

        if is_query and prior_facts:
            # Baseline remembers facts that were stated earlier in this exact thread
            lines = ["Trong phiên trò chuyện này, mình ghi nhận:"]
            matched = False

            if "name" in prior_facts and any(k in lower_msg for k in ["tên", "ai", "mình là ai", "tóm tắt"]):
                lines.append(f"- Tên: {prior_facts['name']}")
                matched = True
            if "location" in prior_facts and any(k in lower_msg for k in ["ở đâu", "nơi ở", "tóm tắt"]):
                lines.append(f"- Nơi ở: {prior_facts['location']}")
                matched = True
            if "profession" in prior_facts and any(k in lower_msg for k in ["nghề", "công việc", "làm gì", "tóm tắt"]):
                lines.append(f"- Nghề nghiệp: {prior_facts['profession']}")
                matched = True
            if "favorite_drink" in prior_facts and any(k in lower_msg for k in ["đồ uống", "uống", "tóm tắt"]):
                lines.append(f"- Đồ uống yêu thích: {prior_facts['favorite_drink']}")
                matched = True
            if "favorite_food" in prior_facts and any(k in lower_msg for k in ["món ăn", "ăn", "tóm tắt"]):
                lines.append(f"- Món ăn yêu thích: {prior_facts['favorite_food']}")
                matched = True
            if "pet" in prior_facts and any(k in lower_msg for k in ["nuôi", "con gì", "corgi", "bơ", "tóm tắt"]):
                lines.append(f"- Thú cưng: {prior_facts['pet']}")
                matched = True
            if "response_style" in prior_facts and any(k in lower_msg for k in ["style", "kiểu", "trả lời", "tóm tắt"]):
                lines.append(f"- Phong cách trả lời: {prior_facts['response_style']}")
                matched = True
            if "interests" in prior_facts and any(k in lower_msg for k in ["quan tâm", "thích", "kỹ thuật", "tóm tắt"]):
                lines.append(f"- Mối quan tâm: {prior_facts['interests']}")
                matched = True

            if not matched:
                for k, v in prior_facts.items():
                    lines.append(f"- {k}: {v}")

            reply_text = "\n".join(lines)
        elif is_query:
            # In a fresh thread or when no facts were stated yet in this thread
            reply_text = "Xin lỗi, mình là Baseline Agent và không có thông tin về bạn trong phiên trò chuyện này."
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
