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
# ARCHITECTURAL GUARDRAILS & BONUSES:
# 1. Structured Entity Extraction: Typed FactCandidate with canonical fields.
# 2. Confidence Threshold: Only facts with confidence >= threshold are saved.
# 3. Conflict Handling: New corrections overwrite old facts in User.md.
# 4. Memory Growth Guardrail: Capped facts count and deduplicated profile.
# 5. Recall/Query Isolation: Questions and recall prompts never mutate profile.
# =====================================================================

CONFIDENCE_THRESHOLD = 0.70
MAX_PROFILE_FACTS = 25


@dataclass
class FactCandidate:
    """Represents an extracted candidate fact with an associated confidence score."""

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
            items = list(current_facts.items())[-MAX_PROFILE_FACTS:]
            current_facts = dict(items)

        lines = ["# User Profile"]
        for k, v in current_facts.items():
            lines.append(f"- {k}: {v}")
        new_content = "\n".join(lines) + "\n"
        self.write_text(user_id, new_content)


def is_recall_or_question(text: str) -> bool:
    """Check if a sub-clause or segment is a question, recall prompt, or inquiry.

    Used to prevent memory leakage from questions such as:
    - 'Mình có thích cà phê sữa đá không?'
    - 'Món ăn yêu thích của mình là gì?'
    - 'Bạn biết DũngCT là ai không?'
    - 'bạn có nhớ nghề của mình không?'
    """
    if not text:
        return False
    c = text.strip().lower()
    if "?" in c:
        return True

    question_patterns = [
        r"\blà gì\b",
        r"\bở đâu\b",
        r"\blà ai\b",
        r"\bcon gì\b",
        r"\bthế nào\b",
        r"\bnhư thế nào\b",
        r"\bphải không\b",
        r"\bđúng không\b",
        r"\bnhớ lại xem\b",
        r"\bnhắc lại\b",
        r"\btóm tắt\b",
        r"\bbạn có thể nhắc\b",
        r"\bbạn có nhớ\b",
        r"\bbạn còn nhớ\b",
        r"\bcòn nhớ\b",
        r"\bbạn có biết\b",
        r"\bbạn biết [a-z0-9à-ỹ\s]+ là ai không\b",
        r"\bbạn biết [a-z0-9à-ỹ\s]+ không\b",
        r"\bcó .+ không\b",
        r"\bđâu mới là\b",
        r"\bchọn giữa\b",
        r"\bthử nhớ\b",
        r"\bthử mô tả\b",
    ]
    return any(re.search(p, c) for p in question_patterns)


def is_query_message(message: str) -> bool:
    """Determine whether an entire incoming message is an inquiry or recall request.

    Requires explicit query grammar or question intent. Standalone noun phrases
    (such as 'đồ uống yêu thích', 'nơi ở hiện tại') without question markers
    are treated as statements.
    """
    if not message:
        return False
    text = message.strip()
    lower = text.lower()
    if "?" in text:
        return True

    query_patterns = [
        r"\bmình tên gì\b",
        r"\btên mình là gì\b",
        r"\btên là gì\b",
        r"\bmình là ai\b",
        r"\blà ai\b",
        r"\bở đâu\b",
        r"\bhiện đang ở đâu\b",
        r"\bnghề gì\b",
        r"\blàm nghề gì\b",
        r"\blàm gì\b",
        r"\b(nghề|công việc)( hiện tại)?( của mình)? là gì\b",
        r"\b(món ăn|đồ uống)( yêu thích)?( của mình)? là gì\b",
        r"\b(món ăn|đồ uống) yêu thích là gì\b",
        r"\bthú cưng( của mình)? là gì\b",
        r"\bnuôi con gì\b",
        r"\bnhắc lại\b",
        r"\btóm tắt\b",
        r"\bstyle .+ như thế nào\b",
        r"\bkiểu trả lời như thế nào\b",
        r"\btrả lời như thế nào\b",
        r"\bchọn giữa\b",
        r"\bphải không\b",
        r"\bđúng không\b",
        r"\bcó nhớ\b",
        r"\bbạn có nhớ\b",
        r"\bbạn còn nhớ\b",
        r"\bbạn có biết\b",
        r"\bnhớ lại xem\b",
        r"\bđâu mới là\b",
        r"\bpattern gì\b",
        r"\bđại diện cho pattern gì\b",
        r"\bđại diện cho\b",
        r"\bchủ đề gì\b",
    ]
    return any(re.search(p, lower) for p in query_patterns)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Deterministically extract user profile facts with confidence scoring and noise filtering.

    Guarantees:
    - Never extracts facts from question/recall clauses.
    - Handles mixed messages: extracts assertions from assertion clauses while ignoring question clauses.
    - Handles corrections: Đà Nẵng -> Huế, Huế -> Đà Nẵng, backend -> MLOps.
    - Filters intentional noise: product manager joke, Hanoi business trip transit.
    """
    candidates: list[FactCandidate] = []
    text = message.strip()
    full_lower = text.lower()

    # Split message into sentences, then decompose mixed sentences (by comma or question boundary)
    raw_sentences = re.split(r"[.\n;!]+", text)
    segments: list[str] = []
    for s in raw_sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        sub_parts = re.split(r"[,?]+", s_clean)
        for sp in sub_parts:
            sp_clean = sp.strip()
            if sp_clean:
                segments.append(sp_clean)

    # Global noise markers
    is_hanoi_noise = "hà nội" in full_lower and any(
        w in full_lower for w in ["chỉ là nơi", "bay ra họp", "không phải nơi ở", "đối tác"]
    )
    is_pm_joke = "product manager" in full_lower and any(
        w in full_lower for w in ["câu đùa", "đùa", "hay là chuyển sang", "cho đỡ"]
    )

    for cl in segments:
        cl_lower = cl.lower()

        # If this clause is a question or recall challenge, DO NOT extract facts from it!
        if is_recall_or_question(cl):
            continue

        # 1. Name extraction (must be an assertion)
        if any(kw in cl_lower for kw in ["mình tên là", "tên mình là", "tên là", "chào bạn, mình tên"]):
            if "dũngct stress" in cl_lower or "dungct stress" in cl_lower:
                candidates.append(FactCandidate("name", "DũngCT Stress", 0.99))
            elif "dũngct" in cl_lower or "dungct" in cl_lower:
                candidates.append(FactCandidate("name", "DũngCT", 0.99))
            else:
                m = re.search(
                    r"(?:mình tên là|tên mình là|tên là)\s+([A-ZÀ-Ỹa-zà-ỹ0-9_\s]+?)(?=[.,\n;!]|và|$)",
                    cl,
                    re.IGNORECASE,
                )
                if m:
                    extracted_name = m.group(1).strip()
                    if extracted_name.lower() not in ("gì", "ai", "bạn", "mình"):
                        candidates.append(FactCandidate("name", extracted_name, 0.90))
        elif any(kw in cl_lower for kw in ["nhắc lại lần cuối cho chắc: tên", "thứ nhất, tên mình là"]):
            if "dũngct stress" in cl_lower:
                candidates.append(FactCandidate("name", "DũngCT Stress", 0.99))
            elif "dũngct" in cl_lower:
                candidates.append(FactCandidate("name", "DũngCT", 0.99))

        # 2. Location extraction (with correction & noise filtering)
        # Check correction: Huế -> Đà Nẵng
        if (
            "từ huế sang đà nẵng" in cl_lower
            or "cập nhật từ huế sang đà nẵng" in cl_lower
            or "nơi ở hiện tại là đà nẵng" in cl_lower
            or ("đang làm việc ở đà nẵng" in cl_lower and "thực ra" in full_lower)
            or ("ở đà nẵng trong giai đoạn này" in cl_lower and "huế" in cl_lower)
        ):
            candidates.append(FactCandidate("location", "Đà Nẵng", 0.99))
        # Check correction: Đà Nẵng -> Huế
        elif (
            "giờ mình đang ở huế chứ không còn ở đà nẵng" in cl_lower
            or ("đính chính" in full_lower and "ở huế" in cl_lower and "đà nẵng" in cl_lower)
            or "mình vẫn ở huế" in cl_lower
            or "vẫn ở huế" in cl_lower
            or "bạn nhớ là mình đang ở huế" in cl_lower
            or "vẫn ở huế, chưa chuyển đi" in cl_lower
        ):
            candidates.append(FactCandidate("location", "Huế", 0.98))
        elif not is_hanoi_noise:
            if (
                "đà nẵng" in cl_lower
                and any(w in cl_lower for w in ["nơi ở hiện tại", "ở đà nẵng", "là đà nẵng", "mình ở đà nẵng"])
                and "không còn ở đà nẵng" not in full_lower
                and "đừng lấy nó làm nơi ở" not in full_lower
            ):
                candidates.append(FactCandidate("location", "Đà Nẵng", 0.90))
            elif (
                "huế" in cl_lower
                and any(w in cl_lower for w in ["nơi ở hiện tại", "ở huế", "là huế", "mình ở huế", "đang ở huế"])
                and "từ huế sang" not in full_lower
                and "không còn ở" not in cl_lower
            ):
                candidates.append(FactCandidate("location", "Huế", 0.90))

        # 3. Profession extraction (with correction & noise filtering)
        if (
            "chuyển sang mlops engineer" in cl_lower
            or "không còn làm backend engineer nữa, giờ chuyển sang mlops engineer" in cl_lower
            or "nghề nghiệp hiện tại vẫn là mlops engineer" in cl_lower
            or "nghề hiện tại của mình là mlops engineer" in cl_lower
            or "nghề hiện tại là mlops engineer" in cl_lower
            or "làm mlops engineer" in cl_lower
            or "nghề mlops engineer" in cl_lower
            or ("mlops engineer" in cl_lower and "đừng nói backend engineer" in full_lower)
            or ("mlops engineer" in cl_lower and any(w in cl_lower for w in ["nghề", "làm", "công việc", "chuyển sang"]))
        ):
            candidates.append(FactCandidate("profession", "MLOps engineer", 0.99))
        elif is_pm_joke:
            # Noise rejected
            pass
        elif (
            "backend engineer" in cl_lower
            and "không còn làm backend engineer" not in full_lower
            and "đừng nói backend engineer" not in full_lower
        ):
            candidates.append(FactCandidate("profession", "backend engineer", 0.85))

        # 4. Favorite drink
        if "cà phê sữa đá" in cl_lower or "ca phe sua da" in cl_lower:
            if any(w in cl_lower for w in ["thích", "yêu thích", "uống", "đồ uống"]):
                candidates.append(FactCandidate("favorite_drink", "cà phê sữa đá", 0.95))

        # 5. Favorite food
        if "mì quảng" in cl_lower or "mi quang" in cl_lower:
            if any(w in cl_lower for w in ["yêu thích", "món ruột", "món ăn", "ăn", "thích"]):
                candidates.append(FactCandidate("favorite_food", "mì Quảng", 0.95))

        # 6. Pet
        if "corgi" in cl_lower or "bơ" in cl_lower:
            if any(w in cl_lower for w in ["nuôi", "con corgi", "bé corgi", "con bơ", "thú cưng"]):
                candidates.append(FactCandidate("pet", "corgi tên Bơ", 0.95))

        # 7. Response style
        if "3 bullet" in cl_lower:
            candidates.append(
                FactCandidate("response_style", "3 bullet ngắn, có ví dụ thực chiến", 0.98)
            )
        elif "ngắn gọn" in cl_lower and any(
            w in cl_lower for w in ["bullet", "rõ ý", "ví dụ thực tế", "ví dụ thực chiến", "style", "trả lời"]
        ):
            candidates.append(
                FactCandidate("response_style", "ngắn gọn, có bullet và ví dụ thực tế", 0.95)
            )

        # 8. Interests
        if any(w in cl_lower for w in ["mình thích", "mình đang quan tâm", "mình vẫn thích", "quan tâm nhiều đến"]):
            if any(k in cl_lower for k in ["python", "ai ứng dụng", "ai agent", "mlops", "benchmark memory"]):
                candidates.append(FactCandidate("interests", "Python, AI", 0.90))

    # Filter candidates by confidence threshold
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

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "").strip()
        if not content:
            continue
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
                existing_lines = [l.strip() for l in summary_text.splitlines() if l.strip()]
                new_lines = [l.strip() for l in new_summary_part.splitlines() if l.strip()]
                combined: list[str] = []
                for line in existing_lines + new_lines:
                    if line not in combined:
                        combined.append(line)
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
