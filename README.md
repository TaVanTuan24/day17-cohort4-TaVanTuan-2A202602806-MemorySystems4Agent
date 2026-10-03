# Phase 2, Track 3, Day 17: Memory Systems for AI Agent

Trong Day 17 này, các bạn sẽ tập trung vào một câu hỏi rất thực tế: làm sao để AI agent **không chỉ trả lời tốt trong một lượt chat**, mà còn **nhớ đúng thông tin quan trọng qua nhiều phiên làm việc** mà vẫn kiểm soát được chi phí token.

Trong bài lab này, các bạn sẽ xây dựng và so sánh hai agent:

- `Baseline Agent`: chỉ có short-term memory trong cùng một thread
- `Advanced Agent`: có short-term memory, `User.md` bền vững, và compact memory để nén hội thoại dài

Mục tiêu cuối cùng không phải chỉ là “agent nhớ nhiều hơn”, mà là hiểu rõ trade-off giữa:

- độ nhớ dài hạn
- chất lượng phản hồi
- chi phí token
- độ phức tạp của hệ thống memory

## Các bạn sẽ làm gì trong track này?

Sau khi hoàn thành, các bạn cần có khả năng:

- phân biệt `short-term memory`, `persistent memory`, và `compact memory`
- xây dựng agent baseline và advanced trên cùng một benchmark
- lưu hồ sơ người dùng bằng `User.md`
- kích hoạt compact memory khi hội thoại dài vượt ngưỡng
- benchmark hai agent bằng cùng một bộ dữ liệu tiếng Việt
- đọc kết quả benchmark theo các chỉ số recall, token, memory growth, chất lượng phản hồi

## Cấu trúc codebase

```
.
├── README.md        # giới thiệu track (file này)
├── Guide.md         # hướng dẫn từng bước
├── Rubric.md        # tiêu chí chấm điểm
├── data/            # dữ liệu benchmark dùng chung
│   ├── conversations.json
│   └── advanced_long_context.json
└── src/             # bản scaffold dành cho sinh viên (pseudocode + TODO)
    ├── model_provider.py
    ├── config.py
    ├── memory_store.py
    ├── agent_baseline.py
    ├── agent_advanced.py
    ├── benchmark.py
    └── test_agents.py
```

Khi chạy, agent sẽ ghi trạng thái (ví dụ `state/profiles/<user>/User.md`) vào thư mục `state/`. Thư mục này đã nằm trong `.gitignore`.

### Vai trò từng file trong `src/`

Các file được liệt kê theo thứ tự nên triển khai:

| File | Vai trò | Thành phần chính |
|---|---|---|
| `model_provider.py` | Khởi tạo chat model cho từng provider | `ProviderConfig`, `normalize_provider()`, `build_chat_model()` |
| `config.py` | Cấu hình chung của lab | `LabConfig` (đường dẫn, ngưỡng compact, model chính + judge), `load_config()` |
| `memory_store.py` | Lõi memory layer | `estimate_tokens()`, `UserProfileStore` (read/write/edit `User.md`), `extract_profile_updates()`, `summarize_messages()`, `CompactMemoryManager` |
| `agent_baseline.py` | Agent A: chỉ nhớ trong cùng thread | `BaselineAgent.reply()`, `token_usage()`, `prompt_token_usage()` |
| `agent_advanced.py` | Agent B: short-term + `User.md` + compact | `AdvancedAgent.reply()`, `_reply_offline()`, `_estimate_prompt_context_tokens()`, `_offline_response()` |
| `benchmark.py` | So sánh hai agent trên hai bộ dữ liệu | `run_agent_benchmark()`, `recall_points()`, `heuristic_quality()`, `format_rows()` |
| `test_agents.py` | Kiểm chứng hành vi memory | test `User.md`, compact trigger, cross-session recall, giảm prompt load |

### Luồng xử lý một lượt của Advanced Agent

```
message người dùng
  → extract_profile_updates()      # trích fact ổn định: tên, nơi ở, nghề, style...
  → ghi vào User.md                # persistent memory
  → CompactMemoryManager.append()  # short-term memory, tự compact khi vượt ngưỡng
  → prompt = User.md + summary + recent messages
  → sinh câu trả lời → cập nhật bộ đếm token
```

Baseline Agent chỉ giữ danh sách message theo `thread_id`. Sang thread mới, nó **phải quên** toàn bộ fact cũ.

Cả hai agent nên có **chế độ offline** cho ra kết quả lặp lại được, để benchmark và test chạy được mà không cần API key. Chế độ live (LangChain/LangGraph) là phần mở rộng.

## Dữ liệu benchmark

| File | Nội dung | Mục tiêu |
|---|---|---|
| `data/conversations.json` | 10 hội thoại khoảng 10 lượt, user `dungct`, kèm `recall_questions` | Standard benchmark: đo recall qua nhiều phiên bình thường |
| `data/advanced_long_context.json` | 1 hội thoại 16 lượt rất dài, user `dungct_stress` | Long-context stress benchmark: ép compact xảy ra nhiều lần |

Mỗi hội thoại có dạng:

```json
{
  "id": "conv-01",
  "user_id": "dungct",
  "turns": ["...", "..."],
  "recall_questions": [
    { "question": "...", "expected_contains": ["DũngCT", "cà phê sữa đá"] }
  ]
}
```

`recall_questions` được hỏi ở **thread mới**. Điểm recall dựa trên số chuỗi trong `expected_contains` xuất hiện trong câu trả lời.

Dữ liệu cố tình chứa các tình huống khó:

- **correction**: nơi ở đổi giữa Đà Nẵng và Huế, agent phải giữ fact mới nhất
- **nhiễu**: "Hà Nội" chỉ là nơi đi họp, "product manager" chỉ là câu đùa
- **ngữ cảnh dài**: nhiều đoạn tin tức dài trong stress test để làm lộ chi phí prompt của baseline

## Provider hỗ trợ

Trong bản solved lab, runtime hỗ trợ các provider sau:

- `openai`
- `custom` (OpenAI-compatible base URL)
- `gemini`
- `anthropic`
- `ollama`
- `openrouter`

Điều này quan trọng vì memory system không nên bị khóa vào một provider duy nhất.

## Chỉ số benchmark cần hiểu

Khi hoàn thiện bài, benchmark nên cho các cột sau:

- `Agent tokens only`: token sinh ra trực tiếp trong hội thoại của agent
- `Prompt tokens processed`: lượng ngữ cảnh agent phải kéo theo qua các lượt
- `Cross-session recall`: khả năng nhớ facts qua thread hoặc session mới
- `Response quality`: chất lượng phản hồi
- `Memory growth (bytes)`: tốc độ phình của file memory
- `Compactions`: số lần compact memory đã nén lịch sử cũ

Điểm quan trọng nhất của track này là:

- ở hội thoại ngắn, `Advanced` có thể tốn hơn `Baseline` về token usage
- ở hội thoại rất dài, compact memory nên giúp `Advanced` xử lý ngữ cảnh hiệu quả hơn đáng kể + tiết kiệm usage.

## Setup môi trường

Các bạn cần chuẩn bị môi trường Python `>= 3.11` và cài các package cần thiết cho LangChain, LangGraph, provider SDK, `python-dotenv`, `tabulate`, và `pytest`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest
```

Nếu muốn chạy chế độ live với LLM thật, hãy tạo file `.env` ở root repo (đã nằm trong `.gitignore`). Tên biến môi trường do các bạn quyết định khi viết `load_config()`. Ví dụ:

```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=...
```

## Chạy benchmark và test

Sau khi hoàn thiện `src/`, chạy từ root repo:

```bash
python src/benchmark.py
```

```bash
pytest src/test_agents.py -v
```

Benchmark cần in ra hai bảng: **Standard Benchmark** và **Long-Context Stress Benchmark**. Mỗi bảng so sánh Baseline với Advanced theo đủ 6 cột trong phần "Chỉ số benchmark cần hiểu".

## Cách dùng repo này

Nếu các bạn là sinh viên:

- làm bài trong `src/`
- dùng `data/` làm benchmark input

Nếu các bạn là giảng viên hoặc reviewer:

- dùng `src/` để đánh giá scaffold giao cho sinh viên và kết quả hoàn thiện cuối cùng

## Tài liệu nên đọc tiếp

- `Guide.md`: hướng dẫn từng bước để hoàn thành lab
- `Rubric.md`: tiêu chí chấm điểm và bonus

Track này được thiết kế để các bạn không chỉ “dùng agent”, mà còn bắt đầu nghĩ như một người thiết kế **memory system** cho agent production.

---

## Kết quả Thực nghiệm & Phân tích Hệ thống Memory (RESULTS & ANALYSIS)

### 1. Bảng số liệu Benchmark thực tế

#### Standard Benchmark (`data/conversations.json` - 10 hội thoại, user `dungct`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| **Baseline** | 1,853 | 15,767 | 0.00 | 0.25 | 0 | 0 |
| **Advanced** | 2,932 | 25,519 | 1.00 | 1.00 | 268 | 0 |

#### Long-Context Stress Benchmark (`data/advanced_long_context.json` - 16 turns dài, user `dungct_stress`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| **Baseline** | 317 | 24,197 | 0.00 | 0.25 | 0 | 0 |
| **Advanced** | 883 | 10,142 | 1.00 | 1.00 | 184 | 9 |

---

### 2. Phân tích chi tiết hành vi và Trade-offs

#### 2.1. Khả năng nhớ Cross-Session & Vai trò của `User.md`
- **Baseline Agent (Recall = 0.00)**: Chỉ quản lý bộ nhớ cục bộ theo `thread_id` (`within-session memory`). Khi bước sang một `fresh_thread_id` để đánh giá recall, Baseline không có quyền truy cập vào các ngữ cảnh lịch sử trước đó và buộc phải trả lời rằng không có thông tin người dùng.
- **Advanced Agent (Recall = 1.00)**: Sở hữu tầng lưu trữ bền vững (`persistent memory`) tách biệt tại `state/profiles/<user_id>/User.md`. Dù bước sang bất kỳ thread mới nào, agent đều tự động nạp `User.md` vào prompt context, cho phép truy xuất chính xác 100% các thực thể thông tin cốt lõi (tên, nơi ở, nghề nghiệp, đồ uống/món ăn yêu thích, thú cưng, phong cách phản hồi).

#### 2.2. Chi phí ngữ cảnh ở hội thoại ngắn (Overhead Trade-off)
- Ở bộ **Standard Benchmark** (các hội thoại ngắn ~10 lượt), Advanced Agent tốn nhiều `Prompt tokens processed` hơn Baseline (25,519 so với 15,767).
- **Lý do**: Ở mỗi lượt trò chuyện, Advanced Agent chủ động nạp thêm cấu trúc markdown `User.md` vào prompt context. Với hội thoại ngắn, overhead của persistent profile chiếm tỷ trọng đáng kể so với dung lượng tin nhắn ngắn, dẫn đến chi phí prompt cao hơn ~61%. Đây là chi phí đánh đổi tất yếu (trade-off) để đạt được độ chính xác recall tuyệt đối qua các phiên làm việc.

#### 2.3. Tác động của Compact Memory trong Long-Context Stress
- Ở bộ **Long-Context Stress Benchmark**, dữ liệu gồm 16 lượt trao đổi chuyên sâu với dung lượng ngữ cảnh rất lớn (các bài báo NASA, WMO, BC Energy).
- **Baseline Agent**: Không nén lịch sử, kéo theo toàn bộ tin nhắn từ đầu đến cuối qua từng lượt. Hệ quả là `Prompt tokens processed` tăng theo cấp số cộng lũy tiến, chạm mốc **24,197 tokens**.
- **Advanced Agent**: `CompactMemoryManager` tự động kích hoạt **9 lần compaction** khi tổng token vượt ngưỡng `compact_threshold_tokens` (800 tokens). Toàn bộ tin nhắn cũ được tóm tắt thành các trừu tượng khái quát (abstractions: Artemis III readiness, X-59 externality, WMO risk communication, BC Energy demand-side efficiency) và chỉ giữ lại `compact_keep_messages` (4 tin nhắn) gần nhất.
- **Kết quả**: `Prompt tokens processed` giảm từ 24,197 xuống chỉ còn **10,142 tokens** (tiết kiệm **~58.1% chi phí prompt**), chứng minh rõ ràng: *compact memory giải quyết triệt để bài toán phình to ngữ cảnh trong hội thoại dài*.

---

### 3. Rủi ro của Persistent Memory & Giải pháp Thiết kế (Bonus Features)

Việc duy trì bộ nhớ dài hạn tiềm ẩn nhiều rủi ro trong môi trường production:
1. **Stale facts (Thông tin lỗi thời)**: Người dùng chuyển nơi ở hoặc đổi nghề nghiệp nhưng hệ thống vẫn lưu thông tin cũ.
2. **Wrong extraction / Noise (Lưu sai do nhiễu)**: Người dùng nói đùa ("hay là làm product manager") hoặc nhắc đến địa điểm đi công tác tạm thời ("Hà Nội họp 2 ngày"), nếu trích xuất máy móc sẽ làm bẩn hồ sơ.
3. **Memory growth phình to**: File `User.md` phình to vô hạn nếu mọi câu nói vụn vặt đều được lưu lại, làm tăng latency và chi phí token.
4. **Conflicting facts**: Lưu đồng thời hai giá trị mâu thuẫn cho cùng một thuộc tính (vừa ở Huế vừa ở Đà Nẵng).

#### Các cơ chế bảo vệ đã triển khai (Bonus Features):
- **Bonus A - Confidence Threshold (`CONFIDENCE_THRESHOLD = 0.70`)**: Chỉ các ứng viên fact có độ tin cậy cao từ các mẫu câu khẳng định mạnh (`mình tên là`, `đính chính`, `giờ mình đang`, `không còn làm... nữa`) mới được lưu vào `User.md`. Các câu hỏi, giả định, hoặc câu nói đùa bị loại bỏ triệt để.
- **Bonus B - Conflict Handling**: Tự động ghi đè giá trị mới nhất lên trường dữ liệu tương ứng khi có đính chính (Đà Nẵng -> Huế ở Standard; Huế -> Đà Nẵng ở Stress; backend engineer -> MLOps engineer), đảm bảo không bao giờ tồn tại đồng thời hai giá trị mâu thuẫn.
- **Bonus C - Structured Entity Extraction**: Chuẩn hóa thông tin thành các thực thể có cấu trúc định danh rõ ràng (`name`, `location`, `profession`, `favorite_drink`, `favorite_food`, `pet`, `response_style`, `interests`).
- **Bonus D - Memory Growth Guardrail**: Thiết lập giới hạn trần `MAX_PROFILE_FACTS = 25` cho `User.md` và deduplicate nội dung tóm tắt trong `CompactMemoryManager` (tối đa 8 dòng abstraction), ngăn chặn triệt để tình trạng memory leak hoặc phình to không kiểm soát.
