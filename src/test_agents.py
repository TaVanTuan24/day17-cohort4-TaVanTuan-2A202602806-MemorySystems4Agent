from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for testing without network dependencies."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    provider_cfg = ProviderConfig(
        provider="custom",
        model_name="mock-model",
        temperature=0.0,
    )
    return LabConfig(
        base_dir=tmp_path,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=150,
        compact_keep_messages=2,
        model=provider_cfg,
        judge_model=provider_cfg,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, read, written, and edited with file size tracking."""
    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / "profiles")
    user_id = "dungct_test"

    # 1. Read default when file does not exist yet
    initial_text = store.read_text(user_id)
    assert "# User Profile" in initial_text

    # 2. Write new content
    content = "# User Profile\n- name: DũngCT\n- location: Đà Nẵng\n"
    written_path = store.write_text(user_id, content)
    assert written_path.exists()
    assert store.read_text(user_id) == content

    # 3. File size should be positive
    assert store.file_size(user_id) > 0

    # 4. Edit content
    edited = store.edit_text(user_id, "location: Đà Nẵng", "location: Huế")
    assert edited is True
    updated_text = store.read_text(user_id)
    assert "location: Huế" in updated_text
    assert "location: Đà Nẵng" not in updated_text


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction, generate summary, and retain recent messages."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    thread_id = "test-compact-thread"
    user_id = "dungct_compact"

    long_text = (
        "Đây là một lượt tin tức dài về dự án khoa học và công nghệ vũ trụ nhằm "
        "tạo áp lực ngữ cảnh vượt ngưỡng compact threshold trong bộ nhớ."
    )
    for i in range(8):
        agent.reply(user_id, thread_id, f"{long_text} (Lượt {i})")

    ctx = agent.compact_memory.context(thread_id)
    summary = str(ctx.get("summary", ""))
    kept_messages = ctx.get("messages", [])

    assert agent.compaction_count(thread_id) > 0
    assert len(summary) > 0
    # Kept messages should be bounded to recent window + assistant response
    assert len(kept_messages) <= config.compact_keep_messages + 2


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify Advanced remembers across sessions/threads via User.md while Baseline forgets."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)
    user_id = "dungct"

    # Thread A: introduce user profile facts
    intro_message = "Chào bạn, mình tên là DũngCT và đồ uống yêu thích là cà phê sữa đá."
    baseline.reply(user_id, "thread-A", intro_message)
    advanced.reply(user_id, "thread-A", intro_message)

    # Thread B: fresh thread, ask recall question
    recall_question = "Mình tên gì và đồ uống yêu thích là gì?"
    res_baseline = baseline.reply(user_id, "thread-B", recall_question)
    res_advanced = advanced.reply(user_id, "thread-B", recall_question)

    # Advanced agent recalls persistent facts
    assert "DũngCT" in res_advanced["reply"]
    assert "cà phê sữa đá" in res_advanced["reply"]

    # Baseline agent has strictly within-thread memory and forgets across threads
    assert "DũngCT" not in res_baseline["reply"]
    assert "cà phê sữa đá" not in res_baseline["reply"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Verify that compact memory reduces cumulative prompt token load on a long conversation."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)
    user_id = "dungct_stress"
    thread_id = "thread-long-benchmark"

    long_message_template = (
        "Nhiệm vụ Artemis III của NASA đặt mục tiêu đổ bộ Mặt Trăng và đòi hỏi hàng loạt kiểm thử "
        "tích hợp hệ thống hỗ trợ sự sống và module kết nối quỹ đạo để đảm bảo tính sẵn sàng cao."
    )
    for i in range(12):
        m = f"{long_message_template} (Lượt {i})"
        baseline.reply(user_id, thread_id, m)
        advanced.reply(user_id, thread_id, m)

    # Advanced triggered compaction
    assert advanced.compaction_count(thread_id) > 0

    # Advanced cumulative prompt tokens processed MUST be significantly less than Baseline
    baseline_prompt_tokens = baseline.prompt_token_usage(thread_id)
    advanced_prompt_tokens = advanced.prompt_token_usage(thread_id)
    assert advanced_prompt_tokens < baseline_prompt_tokens
