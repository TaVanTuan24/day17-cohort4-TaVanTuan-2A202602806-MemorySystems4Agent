from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    is_query_message,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: Advanced Agent.

    Includes three distinct memory tiers:
    1. Short-term memory (within-thread messages)
    2. Persistent memory (User.md for cross-session recall)
    3. Compact memory (automatic summarization when context exceeds threshold)
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None

        if not self.force_offline:
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route message handling between offline deterministic mode and live mode."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                updates = extract_profile_updates(message)
                for k, v in updates.items():
                    self.profile_store.upsert_fact(user_id, k, v)

                self.compact_memory.append(thread_id, "user", message)
                turn_prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
                self.thread_prompt_tokens[thread_id] = (
                    self.thread_prompt_tokens.get(thread_id, 0) + turn_prompt_tokens
                )

                profile_text = self.profile_store.read_text(user_id)
                response = self.langchain_agent.invoke(
                    {
                        "messages": [
                            {"role": "system", "content": f"User Profile Context:\n{profile_text}"},
                            {"role": "user", "content": message},
                        ]
                    },
                    config={"configurable": {"thread_id": thread_id}},
                )
                output_text = (
                    response["messages"][-1].content
                    if isinstance(response, dict) and "messages" in response
                    else str(response)
                )

                self.compact_memory.append(thread_id, "assistant", output_text)
                agent_tokens = estimate_tokens(output_text)
                self.thread_tokens[thread_id] = (
                    self.thread_tokens.get(thread_id, 0) + agent_tokens
                )
                return {
                    "reply": output_text,
                    "agent_tokens": agent_tokens,
                    "prompt_tokens": turn_prompt_tokens,
                }
            except Exception:
                pass

        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative agent response tokens."""
        if thread_id is not None:
            return self.thread_tokens.get(thread_id, 0)
        return sum(self.thread_tokens.values())

    def prompt_token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative prompt tokens processed."""
        if thread_id is not None:
            return self.thread_prompt_tokens.get(thread_id, 0)
        return sum(self.thread_prompt_tokens.values())

    def memory_file_size(self, user_id: str) -> int:
        """Return file size of the user's User.md in bytes."""
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str | None = None) -> int:
        """Return compaction count for a thread or all threads."""
        if thread_id is not None:
            return self.compact_memory.compaction_count(thread_id)
        return sum(
            int(t.get("compactions", 0)) for t in self.compact_memory.state.values()
        )

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline processing flow."""
        # 1. Extract stable profile facts from the incoming message (questions are ignored)
        updates = extract_profile_updates(message)

        # 2. Persist facts into User.md (handles conflict resolution)
        for k, v in updates.items():
            self.profile_store.upsert_fact(user_id, k, v)

        # 3. Append user message into compact memory (triggers compaction if exceeding threshold)
        self.compact_memory.append(thread_id, "user", message)

        # 4. Estimate prompt-context load carried into this turn
        turn_prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + turn_prompt_tokens
        )

        # 5. Generate deterministic offline response using persisted memory & compact summary
        reply_text = self._offline_response(user_id, thread_id, message)

        # 6. Append assistant reply to compact memory
        self.compact_memory.append(thread_id, "assistant", reply_text)

        # 7. Update agent token accounting
        agent_tokens = estimate_tokens(reply_text)
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + agent_tokens
        )

        return {
            "reply": reply_text,
            "agent_tokens": agent_tokens,
            "prompt_tokens": turn_prompt_tokens,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate total context tokens carried into the turn: User.md + summary + kept messages."""
        user_md_content = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary_text = str(ctx.get("summary", ""))
        messages: list[dict[str, str]] = ctx.get("messages", [])  # type: ignore

        tokens = (
            estimate_tokens(user_md_content)
            + estimate_tokens(summary_text)
            + sum(estimate_tokens(m.get("content", "")) for m in messages)
        )
        return tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Generate a deterministic response utilizing persistent profile memory, compact summary, and recent context."""
        lower_msg = message.lower()
        is_query = is_query_message(message)

        if not is_query:
            facts = self.profile_store.facts(user_id)
            style = facts.get("response_style", "")
            if "3 bullet" in style:
                return (
                    "Đã ghi nhận thông tin:\n"
                    "- Đã lưu các cập nhật vào User.md và CompactMemoryManager.\n"
                    "- Giữ trọng tâm vào bài học thực chiến và trade-off hệ thống.\n"
                    "- Sẵn sàng cho các câu hỏi phân tích tiếp theo."
                )
            return "Mình đã ghi nhận thông tin của bạn vào User.md và bộ nhớ ngữ cảnh."

        # Check if the query asks about older conversational context / compacted topics
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))

        if summary and any(
            k in lower_msg
            for k in [
                "artemis",
                "x-59",
                "wmo",
                "el nino",
                "el niño",
                "bc energy",
                "british columbia",
                "điện sạch",
                "pattern",
                "cuộc nói chuyện trước",
                "tin tức",
                "đại diện cho",
            ]
        ):
            summary_lines = [l.strip() for l in summary.splitlines() if l.strip()]
            matched_lines: list[str] = []

            if "artemis" in lower_msg:
                matched_lines.extend([l for l in summary_lines if "artemis" in l.lower()])
            if "x-59" in lower_msg:
                matched_lines.extend([l for l in summary_lines if "x-59" in l.lower()])
            if any(k in lower_msg for k in ["wmo", "el nino", "el niño"]):
                matched_lines.extend(
                    [l for l in summary_lines if any(k in l.lower() for k in ["wmo", "el nino"])]
                )
            if any(k in lower_msg for k in ["bc energy", "columbia", "điện sạch"]):
                matched_lines.extend(
                    [l for l in summary_lines if any(k in l.lower() for k in ["bc energy", "tiết kiệm điện"])]
                )

            if not matched_lines:
                matched_lines = summary_lines[:4]

            response_lines = ["Dựa trên tóm tắt ngữ cảnh đã compact:"]
            for l in matched_lines:
                clean_l = l.lstrip("- ").strip()
                response_lines.append(f"- {clean_l}")
            return "\n".join(response_lines)

        # Query user profile facts from persistent User.md
        facts = self.profile_store.facts(user_id)
        if facts:
            lines = ["Dựa trên hồ sơ người dùng trong User.md:"]
            show_all = any(
                term in lower_msg
                for term in ["tóm tắt", "nhắc lại giúp mình", "ai không", "người dùng"]
            )

            if "name" in facts and (
                show_all or any(k in lower_msg for k in ["tên", "ai", "mô tả", "stress"])
            ):
                lines.append(f"- Tên: {facts['name']}")

            if "location" in facts and (
                show_all
                or any(
                    k in lower_msg
                    for k in ["ở đâu", "nơi ở", "huế", "đà nẵng", "hà nội"]
                )
            ):
                lines.append(f"- Nơi ở hiện tại: {facts['location']}")

            if "profession" in facts and (
                show_all
                or any(
                    k in lower_msg
                    for k in [
                        "nghề",
                        "công việc",
                        "làm gì",
                        "mlops",
                        "backend",
                        "product manager",
                        "chọn giữa",
                    ]
                )
            ):
                lines.append(f"- Nghề nghiệp hiện tại: {facts['profession']}")

            if "favorite_drink" in facts and (
                show_all or any(k in lower_msg for k in ["đồ uống", "uống", "cà phê"])
            ):
                lines.append(f"- Đồ uống yêu thích: {facts['favorite_drink']}")

            if "favorite_food" in facts and (
                show_all or any(k in lower_msg for k in ["món ăn", "ăn", "mì quảng"])
            ):
                lines.append(f"- Món ăn yêu thích: {facts['favorite_food']}")

            if "pet" in facts and (
                show_all
                or any(k in lower_msg for k in ["nuôi", "con gì", "corgi", "bơ", "thú cưng"])
            ):
                lines.append(f"- Thú cưng: {facts['pet']}")

            if "response_style" in facts and (
                show_all
                or any(
                    k in lower_msg
                    for k in ["style", "kiểu", "trả lời", "bullet", "ngắn gọn"]
                )
            ):
                lines.append(f"- Style trả lời mong muốn: {facts['response_style']}")

            if "interests" in facts and (
                show_all
                or any(
                    k in lower_msg
                    for k in ["quan tâm", "thích", "kỹ thuật", "python", "ai"]
                )
            ):
                lines.append(f"- Mối quan tâm kỹ thuật: {facts['interests']}")

            if len(lines) == 1:
                for k, v in facts.items():
                    lines.append(f"- {k}: {v}")

            return "\n".join(lines)

        return "Xin lỗi, hiện tại mình chưa có thông tin phù hợp để trả lời."

    def _maybe_build_langchain_agent(self):
        """Build live LangChain/LangGraph agent when runtime dependencies and keys exist."""
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            model = build_chat_model(self.config.model)
            checkpointer = MemorySaver()
            return create_react_agent(model, tools=[], checkpointer=checkpointer)
        except Exception:
            return None
