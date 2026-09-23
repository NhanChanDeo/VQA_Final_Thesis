Viewed ThesisPremise.md:1-61

Chào bạn, với vai trò là một **giảng viên hướng dẫn (chuyên môn AI, Computer Vision và Recommender Systems/Information Retrieval)**, tôi nhận định ý tưởng đề tài **CrossVQA** của bạn rất có tiềm năng nghiên cứu, đánh trúng bài toán giao thoa đa phương thức (Vision-Language) có tính thời sự cao. 

Tuy nhiên, bản thảo hiện tại (`ThesisPremise.md`) mới dừng ở mức **ý tưởng sơ khởi (idea pitch)** và **đặc tả dữ liệu thô (raw schema specification)**. Để nâng tầm thành một **đề cương khóa luận/luận văn tốt nghiệp (Thesis Proposal / Outline)** chuẩn mực, chặt chẽ về mặt học thuật và có tính khả thi cao, chúng ta cần tái cấu trúc và mở rộng theo **5 trụ cột chính** dưới đây.

---

### 1. Phản biện & Làm sắc nét Động lực nghiên cứu (Research Motivation & Problem Formulation)

Một hội đồng chấm luận văn sẽ đặt ngay các câu hỏi phản biện cốt lõi:
1. **Tại sao lại chuyển giao từ DocVQA $\rightarrow$ TextVQA mà không phải ngược lại, hoặc học đồng thời (Joint Training)?**
   - *Góc nhìn chuyên môn:* Ảnh tài liệu (DocVQA) có mật độ chữ dày, cấu trúc quan hệ ngữ cảnh phân cấp (layout/key-value/table), phông chữ chuẩn. Ngược lại, ảnh tự nhiên (TextVQA) có hình học 3D phức tạp, biến dạng phối cảnh, nhiễu ánh sáng, nghệ thuật hóa phông chữ, chữ thường thưa thớt nhưng gắn chặt với bối cảnh vật thể (scene context).
   - *Cần làm rõ:* Tri thức nào từ DocVQA có thể giúp TextVQA? (Khả năng suy luận ngữ nghĩa văn bản dày, quan hệ không gian giữa các token chữ, hay khả năng đọc hiểu câu hỏi phức tạp?). Cần đề phòng **Negative Transfer** (chuyển giao tiêu cực làm giảm hiệu năng).
2. **Cụ thể hóa thành 3 Câu hỏi Nghiên cứu (Research Questions - RQs):**
   - **RQ1 (Domain Gap Analysis):** Khoảng cách biểu diễn giữa văn bản 2D phẳng và văn bản cảnh 3D biểu hiện như thế nào trong không gian đặc trưng đa phương thức (multimodal feature space)?
   - **RQ2 (Cross-Domain Knowledge Transfer):** Làm thế nào để điều phối biểu diễn (representation alignment) hoặc sử dụng cơ chế thích ứng tham số (parameter-efficient domain adaptation) để chuyển giao năng lực đọc hiểu từ Doc sang Text mà không làm suy giảm khả năng nhận diện hình ảnh tự nhiên?
   - **RQ3 (Ablation & Trade-offs):** Phương pháp đề xuất giải quyết bài toán suy giảm tri thức (catastrophic forgetting) và cân bằng giữa tính chính xác hình học (bounding box/spatial) với suy luận ngữ nghĩa ra sao?

---

### 2. Gợi ý Kiến trúc Kỹ thuật & Góc nhìn liên ngành (CV + AI + RecSys/IR)

Ở đề tài này, bạn nên kết hợp tư duy giữa **Computer Vision**, **Vision-Language Models (VLMs)** và **Information Retrieval/Ranking**:

#### A. Lựa chọn Paradigm & Backbone Model
Bạn cần xác định rõ ràng hướng tiếp cận mô hình:
- **Hướng 1: OCR-based Pipeline (Truyền thống nhưng dễ mổ xẻ cơ chế):** Sử dụng OCR engine bên ngoài (PaddleOCR / Google Cloud Vision) trích xuất tokens + bounding boxes, sau đó đưa vào Multimodal Transformer (như M4C, TAPAS, LayoutLMv3, LaTr).
- **Hướng 2: OCR-free End-to-End VLM (Xu hướng hiện đại):** Donut, Pix2Struct, Florence-2, PaliGemma, hoặc Qwen2-VL.
  > *Khuyến nghị:* Nếu tài nguyên tính toán ở mức trung bình (1-2 GPU T4/RTX 3090/A5000), bạn nên chọn **PaliGemma-3B**, **Florence-2** hoặc **Donut** để tinh chỉnh dạng PEFT (LoRA/QLoRA), hoặc dùng mô hình backbone đã có OCR embeddings để phân tích sâu cơ chế cross-attention.

#### B. Cơ chế Domain Adaptation (Điểm nhấn sáng tạo - Contribution)
Thay vì chỉ "fine-tune nối tiếp", một luận văn tốt cần một giải pháp kỹ thuật cụ thể:
- **Feature-Level Alignment (Đối sánh đặc trưng):** Sử dụng Contrastive Learning hoặc Maximum Mean Discrepancy (MMD) / Adversarial Discriminator giữa các visual tokens có chứa chữ của 2 miền dữ liệu.
- **Cross-Domain Prompt Tuning / Adapter Routing:** Thiết kế các Domain-Specific Adapters hoặc Visual Prompts (ví dụ: một nhánh Adapter học đặc thù 3D Scene Distortion, một nhánh giữ nguyên Core Reasoning từ Document).
- **Góc nhìn RecSys / IR (Candidate Re-ranking):** 
  - Trong TextVQA, câu trả lời thường là việc *chọn/trích xuất* đúng từ trong ảnh hoặc sinh từ mới. Ta có thể mô hình hóa bài toán trả lời câu hỏi như một bài toán **Ranker/Pointer Network**: xếp hạng xác suất các token OCR ứng viên dựa trên độ tương đồng ngữ cảnh truy vấn (query-to-candidate relevance matching).

---

### 3. Chuẩn hóa Quy trình Kỹ thuật Dữ liệu (Data Engineering Pipeline)

Bản thảo của bạn đã chỉ ra sự khác nhau giữa schema của TextVQA và DocVQA. Đây là điểm khởi đầu tốt, nhưng cần mở rộng thành quy trình chuẩn hóa:
1. **Unified Schema:** Xây dựng một Abstract Schema chung (ví dụ: `UnifiedVQAExample` gồm `id`, `image`, `question`, `ground_truth_answers`, `domain_tag`, `ocr_tokens_normalized`, `spatial_boxes_norm_0_1000`).
2. **Data Harmonization:**
   - Chuẩn hóa tọa độ Bounding Box về hệ quy chiếu $[ymin, xmin, ymax, xmax]$ tỷ lệ $[0, 1000]$.
   - Xử lý câu trả lời: TextVQA có 10 người gán nhãn cho mỗi ảnh (dùng VQA accuracy formula), trong khi DocVQA dùng danh sách câu trả lời chấp nhận được (dùng ANLS). Cần thiết kế hàm đánh giá thống nhất hoặc tách biệt nhưng tương thích.
3. **Synthetic Intermediate Domain (Nâng cao):** Tạo tập dữ liệu trung gian (Pseudo-Domain) bằng cách áp dụng các phép biến đổi hình học (affine transformations, 3D perspective warp, noise, shadows, homography) lên ảnh DocVQA để thu hẹp dần domain gap trước khi chuyển sang TextVQA.

---

### 4. Khung Đề cương Chi tiết 5 Chương (Chuẩn Luận văn Tốt nghiệp ĐH/ThS)

Bạn có thể mở rộng file đề cương theo cấu trúc chương hồi chuẩn sau:

```markdown
CHƯƠNG 1: MỞ ĐẦU (INTRODUCTION)
1.1. Bối cảnh và Tầm quan trọng của bài toán Visual Question Answering (VQA)
1.2. Thách thức về khoảng cách miền (Domain Gap) giữa Document VQA và Scene Text VQA
1.3. Mục tiêu nghiên cứu và Các câu hỏi nghiên cứu (Research Questions)
1.4. Đóng góp chính của đề tài (Contributions)
1.5. Bố cục của luận văn

CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ TỔNG QUAN NGHIÊN CỨU (LITERATURE REVIEW)
2.1. Tổng quan về Xử lý ảnh văn bản: Scene Text Detection & Recognition (STR) và Document Layout Analysis
2.2. Các kiến trúc Vision-Language Models (VLM) cho bài toán VQA (M4C, LayoutLM, Donut, PaliGemma)
2.3. Các phương pháp Thích nghi miền (Domain Adaptation) và Chuyển giao tri thức (Transfer Learning)
2.4. Phân tích các bộ dữ liệu chuẩn: TextVQA và DocVQA (Cấu trúc, đặc trưng, sự khác biệt)
2.5. Các thước đo đánh giá: VQA Accuracy, ANLS, và các độ đo khoảng cách miền (FID, MMD)

CHƯƠNG 3: PHƯƠNG PHÁP ĐỀ XUẤT (PROPOSED METHODOLOGY: CROSSVQA)
3.1. Kiến trúc tổng thể hệ thống CrossVQA
3.2. Tiền xử lý và Chuẩn hóa biểu diễn dữ liệu liên miền
3.3. Module trích xuất đặc trưng đa phương thức (Vision-Language Feature Extraction)
3.4. Chiến lược Chuyển giao tri thức và Thích nghi miền:
     - 3.4.1. Cơ chế căn chỉnh không gian biểu diễn (Feature Alignment / Contrastive Objective)
     - 3.4.2. Kỹ thuật thích ứng tham số hiệu quả (PEFT / Domain-specific Adapters)
3.5. Hàm mục tiêu và Quy trình huấn luyện hai giai đoạn (Curriculum / Phased Training)

CHƯƠNG 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ (EXPERIMENTS & EVALUATION)
4.1. Môi trường thực nghiệm, siêu tham số và kịch bản huấn luyện
4.2. Các mô hình cơ sở so sánh (Baselines: Source-only, Target-only, Naive Joint Training)
4.3. Kết quả định lượng trên tập TextVQA và DocVQA (Accuracy, ANLS)
4.4. Phân tích thành phần loại trừ (Ablation Studies: tác động của từng module chuyển giao)
4.5. Phân tích định tính và trường hợp lỗi (Qualitative & Error Analysis: phân loại các ca thất bại do góc nhìn, ánh sáng, hay do OCR)

CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (CONCLUSION & FUTURE WORK)
5.1. Tóm tắt kết quả đạt được
5.2. Hạn chế của nghiên cứu
5.3. Hướng nghiên cứu mở rộng (Mở rộng sang InfographicVQA, Video Text VQA, hoặc triển khai ứng dụng thực tế)
```

---

### 5. Kế hoạch Thực nghiệm Khuyến nghị (Ablation & Baselines Matrix)

Để hội đồng đánh giá cao tính khoa học, bạn cần chuẩn bị một ma trận thực nghiệm rõ ràng:

| Kịch bản Thực nghiệm (Experiment) | Dữ liệu Train | Kỹ thuật áp dụng | Mục đích chứng minh |
| :--- | :--- | :--- | :--- |
| **B1 (Target Only)** | Chỉ TextVQA | Standard Fine-tuning | Mức chuẩn cơ bản của target domain |
| **B2 (Source Only - Zero-shot Target)** | Chỉ DocVQA | Standard Fine-tuning | Đánh giá khả năng chuyển giao tự nhiên (out-of-domain) |
| **B3 (Naive Joint Training)** | DocVQA + TextVQA | Gộp chung train | Kiểm tra xem việc học đồng thời có gây nhiễu/negative transfer không |
| **B4 (Sequential Fine-tuning)** | DocVQA $\rightarrow$ TextVQA | Fine-tune tuần tự | Đánh giá hiện tượng quên tri thức (catastrophic forgetting) |
| **M1 (Proposed CrossVQA)** | DocVQA $\rightarrow$ TextVQA | Chuyển giao có Domain Adaptation (Adapter / Alignment) | Chứng minh vượt trội hơn B1-B4 về cả Accuracy và ANLS |

---

### Lời khuyên tiếp theo cho bạn:
1. Bạn nên cập nhật file `ThesisPremise.md` thành một bản **Đề cương nghiên cứu hoàn chỉnh** (Research Proposal) bao gồm các mục trên.
2. Kiểm tra tài nguyên tính toán khả dụng (GPU VRAM) để chốt kiến trúc mô hình Backbone phù hợp trước khi code pipeline.
3. Nếu cần, tôi có thể hỗ trợ bạn viết ngay mã nguồn khung (code skeleton) cho Data Loader hợp nhất giữa DocVQA và TextVQA hoặc xây dựng chi tiết file `ThesisPremise.md` mới. Bạn muốn bắt đầu mở rộng phần nào trước?