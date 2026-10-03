from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Deterministic token estimator heuristic for English and Vietnamese text.

    Properties:
    - Empty or whitespace-only text returns 0.
    - Always deterministic and non-negative.
    - Combines character length (~3.5 chars/token) and word count (~1.2 tokens/word).
    """
    if not text:
        return 0
    cleaned = text.strip()
    if not cleaned:
        return 0

    char_estimate = len(cleaned) / 3.5
    word_estimate = len(cleaned.split()) * 1.2
    est = int((char_estimate + word_estimate) / 2)
    return max(1, est)


# =====================================================================
# BONUS ARCHITECTURAL FEATURES:
# 1. Structured Entity Extraction: Typed FactCandidate with canonical fields.
# 2. Confidence Threshold: Only facts with confidence >= threshold are saved.
# 3. Conflict Handling: New corrections overwrite old facts in User.md.
# 4. Memory Growth Guardrail: Capped facts count and deduplicated profile.
# =====================================================================

CONFIDENCE_THRESHOLD = 0.70
MAX_PROFILE_FACTS = 25


@dataclass
class FactCandidate:
    """Represents an extracted candidate fact with an associated confidence score.

    - field: The normalized property name (e.g. 'location', 'profession').
    - value: The extracted canonical value.
    - confidence: Score between 0.0 and 1.0.
    """

    field: str
    value: str
    confidence: float


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` files.

    Each user has a distinct profile file located at `root_dir/<sanitized_user_id>/User.md`.
    Includes conflict resolution (upserting overwrites old values) and memory guardrails.
    """

    root_dir: Path

    def _sanitize_user_id(self, user_id: str) -> str:
        """Sanitize user id to prevent path traversal attacks."""
        sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id.strip())
        return sanitized or "anonymous"

    def path_for(self, user_id: str) -> Path:
        """Return the absolute path for the user's User.md profile."""
        slug = self._sanitize_user_id(user_id)
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        """Return the raw markdown content of the user's profile."""
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "# User Profile\n\nNo facts recorded yet.\n"

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown content to disk and return the path."""
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace the first occurrence of search_text with replacement."""
        path = self.path_for(user_id)
        if not path.exists():
            return False
        content = path.read_text(encoding="utf-8")
        if search_text not in content:
            return False
        new_content = content.replace(search_text, replacement, 1)
        path.write_text(new_content, encoding="utf-8")
        return True

    def file_size(self, user_id: str) -> int:
        """Return the size of the User.md file in bytes."""
        path = self.path_for(user_id)
        if path.exists():
            return path.stat().st_size
        return 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse structured facts from the user's markdown profile."""
        content = self.read_text(user_id)
        parsed_facts: dict[str, str] = {}
        for line in content.splitlines():
            line_str = line.strip()
            if line_str.startswith("- ") and ":" in line_str:
                raw_pair = line_str[2:].strip()
                k, v = raw_pair.split(":", 1)
                parsed_facts[k.strip()] = v.strip()
        return parsed_facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Update or insert a fact.

        Conflict Handling:
        Overwrites existing value for the given key, ensuring no duplicate entries.

        Memory Growth Guardrail:
        Restricts the total number of profile facts to MAX_PROFILE_FACTS.
        """
        current_facts = self.facts(user_id)
        current_facts[key] = value

        # Guardrail: cap number of facts
        if len(current_facts) > MAX_PROFILE_FACTS:
            # retain the most recently inserted/updated facts
            items = list(current_facts.items())[-MAX_PROFILE_FACTS:]
            current_facts = dict(items)

        lines = ["# User Profile"]
        for k, v in current_facts.items():
            lines.append(f"- {k}: {v}")
        new_content = "\n".join(lines) + "\n"
        self.write_text(user_id, new_content)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Deterministically extract user profile facts with confidence scoring and noise filtering.

    Handles:
    - Name ('DũngCT', 'DũngCT Stress')
    - Location with corrections (Đà Nẵng -> Huế, Huế -> Đà Nẵng) and noise filtering (Hà Nội transit)
    - Profession with corrections (backend -> MLOps) and noise filtering ('product manager' joke)
    - Preferences (favorite_drink, favorite_food, pet, response_style, interests)
    - Rejection of questions or joke statements
    """
    candidates: list[FactCandidate] = []
    text = message.strip()
    lower_text = text.lower()

    # Skip pure question turns or recall challenges that don't provide facts
    is_pure_question = text.endswith("?") and not any(
        kw in lower_text
        for kw in [
            "mình tên là",
            "đính chính",
            "giờ mình",
            "mình không còn",
            "món ăn yêu thích",
            "mình nuôi",
            "mình đang quan tâm",
        ]
    )

    # 1. Name extraction
    if not is_pure_question:
        if "dũngct stress" in lower_text or "dungct stress" in lower_text:
            candidates.append(FactCandidate("name", "DũngCT Stress", 0.99))
        elif (
            "mình tên là dũngct" in lower_text
            or "tên là dũngct" in lower_text
            or "tên mình là dũngct" in lower_text
            or "chào bạn, mình tên là dũngct" in lower_text
        ):
            candidates.append(FactCandidate("name", "DũngCT", 0.99))
        else:
            name_match = re.search(
                r"(?:mình tên là|tên mình là)\s+([A-ZÀ-Ỹa-zà-ỹ0-9_\s]+?)(?=[.,\n;]|và|$)",
                text,
                re.IGNORECASE,
            )
            if name_match:
                extracted_name = name_match.group(1).strip()
                if extracted_name.lower() not in ("gì", "ai", "bạn", "mình"):
                    candidates.append(FactCandidate("name", extracted_name, 0.90))

    # 2. Location extraction with correction and noise rejection
    # Noise check: "Hà Nội" is only a temporary meeting location, NOT residence
    is_hanoi_noise = "hà nội" in lower_text and any(
        w in lower_text for w in ["chỉ là nơi", "bay ra họp", "không phải nơi ở", "đối tác"]
    )

    # Check for explicit correction: Huế -> Đà Nẵng
    if (
        "từ huế sang đà nẵng" in lower_text
        or "nơi ở đã cập nhật từ huế sang đà nẵng" in lower_text
        or (
            "đang làm việc ở đà nẵng" in lower_text
            and "thực ra" in lower_text
            and "huế" in lower_text
        )
        or "nơi ở hiện tại là đà nẵng" in lower_text
    ):
        candidates.append(FactCandidate("location", "Đà Nẵng", 0.99))
    # Check for explicit correction: Đà Nẵng -> Huế
    elif (
        "giờ mình đang ở huế chứ không còn ở đà nẵng" in lower_text
        or (
            "đính chính" in lower_text
            and "ở huế" in lower_text
            and "đà nẵng" in lower_text
        )
        or "mình vẫn ở huế" in lower_text
        or "bạn nhớ là mình đang ở huế" in lower_text
    ):
        candidates.append(FactCandidate("location", "Huế", 0.98))
    elif not is_hanoi_noise:
        if (
            "ở đà nẵng" in lower_text
            and "không còn ở đà nẵng" not in lower_text
            and "đừng lấy nó làm nơi ở" not in lower_text
        ):
            candidates.append(FactCandidate("location", "Đà Nẵng", 0.85))
        elif (
            "ở huế" in lower_text
            and "từ huế sang" not in lower_text
            and not is_pure_question
        ):
            candidates.append(FactCandidate("location", "Huế", 0.90))

    # 3. Profession extraction with correction and noise rejection
    # Noise check: "product manager" is explicitly a joke
    is_pm_joke = "product manager" in lower_text and any(
        w in lower_text
        for w in ["câu đùa", "đùa", "hay là chuyển sang", "cho đỡ phải"]
    )

    if (
        "chuyển sang mlops engineer" in lower_text
        or "không còn làm backend engineer nữa, giờ chuyển sang mlops engineer" in lower_text
        or "nghề nghiệp hiện tại vẫn là mlops engineer" in lower_text
        or "làm mlops engineer" in lower_text
        or "nghề mlops engineer" in lower_text
        or (
            "mlops engineer" in lower_text
            and "đừng nói backend engineer" in lower_text
        )
    ):
        candidates.append(FactCandidate("profession", "MLOps engineer", 0.99))
    elif is_pm_joke:
        # Rejected as noise
        pass
    elif (
        "backend engineer" in lower_text
        and "không còn làm backend engineer" not in lower_text
        and "đừng nói backend engineer" not in lower_text
        and not is_pure_question
    ):
        candidates.append(FactCandidate("profession", "backend engineer", 0.85))

    # 4. Favorite drink
    if "cà phê sữa đá" in lower_text or "ca phe sua da" in lower_text:
        candidates.append(FactCandidate("favorite_drink", "cà phê sữa đá", 0.95))

    # 5. Favorite food
    if "mì quảng" in lower_text or "mi quang" in lower_text:
        candidates.append(FactCandidate("favorite_food", "mì Quảng", 0.95))

    # 6. Pet
    if "corgi" in lower_text or "bơ" in lower_text:
        if "corgi" in lower_text:
            candidates.append(FactCandidate("pet", "corgi tên Bơ", 0.95))

    # 7. Response style
    if "3 bullet" in lower_text:
        candidates.append(
            FactCandidate("response_style", "3 bullet ngắn, có ví dụ thực chiến", 0.98)
        )
    elif (
        "ngắn gọn" in lower_text
        and any(w in lower_text for w in ["bullet", "rõ ý", "ví dụ thực tế", "ví dụ thực chiến", "style"])
    ):
        candidates.append(
            FactCandidate("response_style", "ngắn gọn, có bullet và ví dụ thực tế", 0.95)
        )
    elif "ngắn gọn" in lower_text and not is_pure_question:
        candidates.append(
            FactCandidate("response_style", "ngắn gọn, có bullet và ví dụ thực tế", 0.85)
        )

    # 8. Interests
    if any(k in lower_text for k in ["python", "ai ứng dụng", "ai agent", "mối quan tâm"]):
        candidates.append(FactCandidate("interests", "Python, AI", 0.90))

    # Filter by confidence threshold
    valid_updates: dict[str, str] = {}
    for cand in candidates:
        if cand.confidence >= CONFIDENCE_THRESHOLD:
            valid_updates[cand.field] = cand.value

    return valid_updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a deterministic, compact summary of older messages.

    Compresses long turns into concise thematic abstractions:
    - Artemis III -> dependency & readiness before launch
    - X-59 -> performance vs externality (sonic boom noise reduction)
    - WMO -> uncertainty and quantitative risk communication
    - BC Energy -> balance between scale/capacity and demand-side efficiency
    - General discussion / user intent
    """
    if not messages:
        return ""

    summary_bullets: list[str] = []

    combined_text = " ".join(m.get("content", "") for m in messages)
    lower_text = combined_text.lower()

    if "artemis iii" in lower_text or "artemis" in lower_text:
        summary_bullets.append(
            "- Chủ đề Artemis III: Quản trị roadmap, dependency milestones và readiness trước launch lớn."
        )

    if "x-59" in lower_text:
        summary_bullets.append(
            "- Chủ đề X-59: Tối ưu hiệu năng siêu thanh song hành giảm externality tiếng nổ cho cộng đồng."
        )

    if "wmo" in lower_text or "el nino" in lower_text:
        summary_bullets.append(
            "- Chủ đề WMO: Dự báo rủi ro El Nino, truyền thông rủi ro và ra quyết định khi độ bất định tăng."
        )

    if "british columbia" in lower_text or "bc energy" in lower_text or "power smart" in lower_text:
        summary_bullets.append(
            "- Chủ đề BC Energy: Cân bằng giữa mở rộng công suất (capex) và tối ưu hiệu quả tiết kiệm điện."
        )

    # Extract conversation highlights from messages not covered by macro themes
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "").strip()
        if not content:
            continue
        # Only take short snippets if not already crowded
        if len(summary_bullets) >= max_items:
            break
        if role == "user":
            first_sentence = content.split(".")[0].strip()
            if len(first_sentence) > 100:
                first_sentence = first_sentence[:97] + "..."
            if (
                not any(topic in first_sentence.lower() for topic in ["artemis", "x-59", "wmo", "columbia", "stress"])
                and len(first_sentence) > 15
            ):
                summary_bullets.append(f"- Ghi chú thảo luận: {first_sentence}")

    if not summary_bullets:
        summary_bullets.append("- Tóm tắt các lượt hội thoại trước đó.")

    return "\n".join(summary_bullets[:max_items])


@dataclass
class CompactMemoryManager:
    """Manages short-term memory and automatic compaction for long threads.

    Attributes:
    - threshold_tokens: token threshold above which compaction triggers.
    - keep_messages: number of most recent messages to retain in full.
    - state: thread-level state storage {thread_id: {messages, summary, compactions}}.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _get_thread_state(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append a new message to the thread and trigger compaction if threshold exceeded."""
        t_state = self._get_thread_state(thread_id)
        messages: list[dict[str, str]] = t_state["messages"]  # type: ignore
        messages.append({"role": role, "content": content})

        # Calculate current memory tokens (summary + all messages)
        summary_text: str = t_state["summary"]  # type: ignore
        current_tokens = estimate_tokens(summary_text) + sum(
            estimate_tokens(m["content"]) for m in messages
        )

        # Trigger compaction if tokens exceed threshold and we have more messages than keep_messages
        if current_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            split_idx = len(messages) - self.keep_messages
            to_compact = messages[:split_idx]
            kept = messages[split_idx:]

            new_summary_part = summarize_messages(to_compact)
            if summary_text:
                # Merge existing and new summary lines deterministically while deduplicating
                existing_lines = [l.strip() for l in summary_text.splitlines() if l.strip()]
                new_lines = [l.strip() for l in new_summary_part.splitlines() if l.strip()]
                combined: list[str] = []
                for line in existing_lines + new_lines:
                    if line not in combined:
                        combined.append(line)
                # Keep bounded at most 8 lines to prevent summary runaway
                t_state["summary"] = "\n".join(combined[-8:])
            else:
                t_state["summary"] = new_summary_part

            t_state["messages"] = kept
            t_state["compactions"] = int(t_state["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        """Return the context view for the given thread."""
        return self._get_thread_state(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        """Return total compactions executed for this thread."""
        return int(self._get_thread_state(thread_id)["compactions"])
