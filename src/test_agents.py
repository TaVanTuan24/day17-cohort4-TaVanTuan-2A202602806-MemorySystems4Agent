from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import UserProfileStore, is_query_message
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

    assert advanced.compaction_count(thread_id) > 0
    assert advanced.prompt_token_usage(thread_id) < baseline.prompt_token_usage(thread_id)


def test_baseline_recalls_within_same_thread(tmp_path: Path) -> None:
    """Verify Baseline remembers facts within the same thread, but forgets in a new thread."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    user_id = "dungct"

    # Thread A: User asserts name, then asks for name in the same thread
    baseline.reply(user_id, "thread-A", "Chào bạn, mình tên là DũngCT.")
    res_same = baseline.reply(user_id, "thread-A", "Mình tên gì?")
    assert "DũngCT" in res_same["reply"]

    # Thread B: Ask in a fresh thread; Baseline MUST forget
    res_fresh = baseline.reply(user_id, "thread-B", "Mình tên gì?")
    assert "DũngCT" not in res_fresh["reply"]
    assert "không có thông tin" in res_fresh["reply"]


def test_recall_question_does_not_write_profile(tmp_path: Path) -> None:
    """Verify queries and recall questions do not mutate User.md or create false facts."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    user_id = "test_no_leak"

    # Query 1: Question about coffee should NOT create favorite_drink
    agent.reply(user_id, "thread-1", "Mình có thích cà phê sữa đá không?")
    facts1 = agent.profile_store.facts(user_id)
    assert "favorite_drink" not in facts1

    # Query 2: Question about food should NOT create favorite_food
    agent.reply(user_id, "thread-2", "Món ăn yêu thích của mình là gì?")
    facts2 = agent.profile_store.facts(user_id)
    assert "favorite_food" not in facts2

    # Query 3: Question about identity should NOT create name
    agent.reply(user_id, "thread-3", "Bạn biết DũngCT là ai không?")
    facts3 = agent.profile_store.facts(user_id)
    assert "name" not in facts3


def test_profile_correction_overwrites_old_fact(tmp_path: Path) -> None:
    """Verify corrections overwrite old facts in User.md without duplicate keys."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    user_id = "test_correction"

    # Location correction: Đà Nẵng -> Huế
    agent.reply(user_id, "thread-1", "Mình ở Đà Nẵng.")
    agent.reply(user_id, "thread-1", "Giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.")
    facts = agent.profile_store.facts(user_id)
    assert facts.get("location") == "Huế"
    profile_text = agent.profile_store.read_text(user_id)
    assert "location: Huế" in profile_text
    assert "location: Đà Nẵng" not in profile_text

    # Profession correction: backend engineer -> MLOps engineer
    agent.reply(user_id, "thread-1", "Mình đang làm backend engineer.")
    agent.reply(
        user_id,
        "thread-1",
        "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.",
    )
    facts = agent.profile_store.facts(user_id)
    assert facts.get("profession") == "MLOps engineer"
    profile_text = agent.profile_store.read_text(user_id)
    assert "profession: MLOps engineer" in profile_text
    assert "profession: backend engineer" not in profile_text


def test_noise_does_not_overwrite_stable_profile(tmp_path: Path) -> None:
    """Verify jokes and temporary transit locations do not corrupt stable facts."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    user_id = "test_noise"

    # Pre-populate valid profile
    agent.profile_store.upsert_fact(user_id, "location", "Đà Nẵng")
    agent.profile_store.upsert_fact(user_id, "profession", "MLOps engineer")

    # Noise 1: Product manager joke
    agent.reply(
        user_id,
        "thread-noise",
        "Có lúc mình đùa là hay chuyển sang product manager cho đỡ canh pipeline, nhưng đó chỉ là câu đùa.",
    )
    facts = agent.profile_store.facts(user_id)
    assert facts.get("profession") == "MLOps engineer"

    # Noise 2: Hanoi business trip
    agent.reply(
        user_id,
        "thread-noise",
        "Hà Nội chỉ là nơi mình bay ra họp hai ngày, không phải nơi ở hiện tại.",
    )
    facts = agent.profile_store.facts(user_id)
    assert facts.get("location") == "Đà Nẵng"


def test_compact_summary_supports_old_context_recall(tmp_path: Path) -> None:
    """Verify that compact summary provides relevant context to answer questions on compacted themes."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    user_id = "dungct_summary_test"
    thread_id = "thread-summary"

    # Provide news turns regarding Artemis III and X-59 to trigger compaction
    turns = [
        (
            "Tin NASA về Artemis III: công bố crew cho nhiệm vụ bay quanh Mặt Trăng năm 2027 "
            "như bước đệm trước Artemis IV 2028. Bài học quản trị sản phẩm là lộ trình phải neo "
            "bằng dependency milestones và readiness có thể kiểm chứng."
        ),
        (
            "Tin NASA về X-59: máy bay lần đầu bay siêu thanh Mach 1.1, hướng tới giảm tiếng nổ "
            "xuống mức tiếng thump nhẹ để cộng đồng chấp nhận, tối ưu hiệu năng song hành giảm externality."
        ),
        "Đây là turn dài thứ ba để tiếp tục đẩy số lượng token trong thread lên cao vượt ngưỡng compaction.",
        "Đây là turn dài thứ tư nhằm chắc chắn kích hoạt compact memory cho thread này.",
    ]
    for turn in turns:
        agent.reply(user_id, thread_id, turn)

    assert agent.compaction_count(thread_id) > 0

    # Ask question regarding the compacted Artemis III topic
    query = "Artemis III trong cuộc nói chuyện trước đại diện cho pattern gì?"
    res = agent.reply(user_id, thread_id, query)
    reply_lower = res["reply"].lower()

    assert any(term in reply_lower for term in ["readiness", "dependency", "artemis"])


def test_statements_are_not_misclassified_as_queries(tmp_path: Path) -> None:
    """Verify affirmative user statements are not misclassified as queries and update memory."""
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)
    user_id = "test_statements"

    statements = [
        "Đồ uống yêu thích của mình là cà phê sữa đá.",
        "Món ăn yêu thích của mình là mì Quảng.",
        "Nghề hiện tại của mình là MLOps engineer.",
        "Nơi ở hiện tại của mình là Huế.",
    ]

    for stmt in statements:
        # 1. Statement must NOT be flagged as a query
        assert is_query_message(stmt) is False

        # 2. Baseline should acknowledge rather than say 'không có thông tin'
        base_res = baseline.reply(user_id, "thread-base", stmt)
        assert "không có thông tin" not in base_res["reply"]
        assert "đã nhận được" in base_res["reply"]

        # 3. Advanced should extract and store the facts
        advanced.reply(user_id, "thread-adv", stmt)

    facts = advanced.profile_store.facts(user_id)
    assert facts.get("favorite_drink") == "cà phê sữa đá"
    assert facts.get("favorite_food") == "mì Quảng"
    assert facts.get("profession") == "MLOps engineer"
    assert facts.get("location") == "Huế"


def test_mixed_assertion_and_question_preserves_assertion(tmp_path: Path) -> None:
    """Verify statements containing both an assertion and a question preserve the assertion without leaking question content."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    user_id = "test_mixed"

    # Message 1: Contains location assertion and question about profession
    agent.reply(
        user_id,
        "thread-mixed-1",
        "Mình vẫn ở Huế nhé, bạn có nhớ nghề của mình không?",
    )
    facts1 = agent.profile_store.facts(user_id)
    assert facts1.get("location") == "Huế"
    assert "profession" not in facts1

    # Message 2: Contains profession assertion and question about location
    agent.reply(
        user_id,
        "thread-mixed-2",
        "Mình vẫn làm MLOps engineer nhé, bạn còn nhớ mình ở đâu không?",
    )
    facts2 = agent.profile_store.facts(user_id)
    assert facts2.get("profession") == "MLOps engineer"


def test_provider_api_key_resolution(monkeypatch, tmp_path: Path) -> None:
    """Verify provider-specific environment variables map to the correct model config."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-secret")
    monkeypatch.setenv("JUDGE_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key-secret")

    cfg = load_config(tmp_path)

    assert cfg.model.provider == "gemini"
    assert cfg.model.api_key == "gemini-key-secret"
    assert cfg.judge_model.provider == "anthropic"
    assert cfg.judge_model.api_key == "anthropic-key-secret"
