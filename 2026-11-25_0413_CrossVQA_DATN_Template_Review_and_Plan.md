# Kế hoạch Rà soát Cấu trúc & Xây dựng Nội dung Báo cáo Đồ án Tốt nghiệp CrossVQA

> **Tài liệu mẫu được đánh giá:** `OFFLIMIT/DATN-template.docx` (Đồ án Tốt nghiệp Đại học — Theo định hướng nghiên cứu, PTIT)  
> **Tài liệu tham khảo văn phong tiếng Việt:** `OFFLIMIT/N16.docx` (Báo cáo cuối kỳ PTIT — phân loại bình luận độc hại ViHSD/ViHOS)  
> **Phạm vi đối chiếu:** Toàn bộ mã nguồn, dữ liệu, tài liệu toán học và kế hoạch hiện tại của dự án CrossVQA (`PLAN/`, `DATASETS SCHEMA NORMALIZATION/`, `ANALYSIS/`, `DATA_PIPELINE/`, `TRAIN/`, `MATHS/`, `EVALUATION/`, `DEMO/`).

---

## 1. Mục tiêu & Đánh giá Tổng quan

`OFFLIMIT/DATN-template.docx` đã được phác thảo sẵn theo **khung 3 chương chuẩn PTIT định hướng nghiên cứu** và bám rất sát chủ đề CrossVQA. Tuy nhiên, quá trình kiểm toán đối chiếu với thực trạng dự án cho thấy **4 độ lệch quan trọng** cần chỉnh sửa trước khi viết nội dung:

1. **Thiếu các thành phần đã hoàn thiện trong mã nguồn `TRAIN/`:** Mẫu báo cáo hiện đang bỏ sót **M3** (`CurriculumPseudoDomainTransform` trong `TRAIN/src/augmentations.py`), baseline **B5** (Reverse Sequential `Text -> Doc` trong `TRAIN/ALT_THREE_SPLITS/configs/baselines_B1_B5.yaml`) và backbone thứ 3 **`HuggingFaceTB/SmolVLM-500M-Base`** (`TRAIN/ALT_THREE_SPLITS/configs/backbones.yaml`).
2. **Một số mục trong mẫu chưa có hiện thực thực nghiệm:** `TRAIN/VALIDATION_AS_TEST/` (phục vụ mục `3.2.4`), `EVALUATION/` (phục vụ kiểm định thống kê Bootstrap CI & phân loại lỗi `3.3.1`), và `DEMO/` (phục vụ mục `3.3.2`) hiện chỉ chứa `.gitkeep`. Ngoài ra, các kết quả huấn luyện `TRAIN/ALT_THREE_SPLITS/runs/` chưa được chạy xong.
3. **Bẫy nhất quán số liệu giữa hai giao thức chia tập (`2.2.2` vs `2.2.3`):** Tài liệu cũ `DATA_PIPELINE/Verdict on ds splitting decision.md` ghi nhận 0 tài liệu trùng lặp giữa `train` và `validation` gốc, trong khi kiểm toán mới hơn tại `DATA_PIPELINE/ALT_THREE_SPLITS/Verdict on ds splitting decision (Pooled).md` phát hiện **523 tài liệu cha (`ucsf_document_id`, 13.798 câu hỏi) vắt ngang giữa `train` và `validation` gốc** của DocVQA. Cần kể lại mạch phát hiện khoa học này một cách nhất quán.
4. **Lỗi trình bày nhỏ trên chính file mẫu:** Trang bìa trong (`OFFLIMIT/DATN-template.docx`) có lỗi dính chữ `"TÀI LIỆUSANG"` (thiếu dấu cách); đồng thời bản sao ở thư mục gốc `DATN-CrossVQA-template.docx` đã lược mất mục `"7. Nội dung chính của đề án"` và đổi `"DANH MỤC KHÁI NIỆM..."` thành `"DANH SÁCH TỪ KHÓA"` — nên lấy `OFFLIMIT/DATN-template.docx` làm gốc chuẩn PTIT hơn.

---

## 2. Các Quyết định Cần Người dùng Xác nhận (User Review Required)

> [!IMPORTANT]
> Trước khi cập nhật lại đề cương `.docx` và bắt đầu viết nội dung, cần chốt **4 quyết định phạm vi** sau để tránh viết thừa hoặc hứa hẹn thực nghiệm không có số liệu:

1. **Mục `3.2.4` — Thực nghiệm đối chứng `VALIDATION_AS_TEST` vs `ALT_THREE_SPLITS`:**
   - Hiện `TRAIN/VALIDATION_AS_TEST/` chưa có script/config/run nào.
   - *Phương án A (Khuyên dùng nếu giới hạn GPU):* **Thu gọn / Cắt bỏ khỏi Chương 3** việc huấn luyện lặp lại toàn bộ ma trận trên `VALIDATION_AS_TEST`; thay vào đó chỉ đối chiếu **định lượng mức độ rò rỉ dữ liệu** ở mục `2.2.2`–`2.2.3` (523 tài liệu cầu nối + 85 cặp gần trùng `pHash`), hoặc chỉ chạy đối chứng tối thiểu 1 cặp (`B2` hoặc `B4` trên `SmolVLM-500M-Base`) cho mục `3.2.4`.
   - *Phương án B:* Bổ sung runner `TRAIN/VALIDATION_AS_TEST/` và chạy đầy đủ để giữ nguyên mục `3.2.4`.
2. **Mục `3.3.2` — Xây dựng và triển khai ứng dụng minh họa (`DEMO/`):**
   - Với PTIT **định hướng nghiên cứu**, demo ứng dụng không bắt buộc nặng về hệ thống sản phẩm như định hướng kỹ thuật, nhưng một giao diện Gradio/Streamlit nhẹ (giống mục `4.5` trong `OFFLIMIT/N16.docx`) giúp tăng sức thuyết phục khi bảo vệ.
   - *Phương án A (Khuyên dùng):* Giữ mục `3.3.2` ở mức **Demo Gradio nhẹ** (tải checkpoint `DualLoRA`, hiển thị câu trả lời + trọng số định tuyến $w_{\text{doc}}, w_{\text{scene}}$ + lớp phủ hộp bao OCR).
   - *Phương án B:* **Cắt bỏ (Truncate)** mục `3.3.2` để tập trung 100% nguồn lực vào phân tích định tính & lỗi (`3.3.1`), đổi tên Chương 3 thành *"THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ"*.
3. **Bổ sung `M3`, `B5` và `SmolVLM-500M-Base` vào đề cương:**
   - Xác nhận có giữ đầy đủ cả 3 tầng đề xuất (`M1`: Dual LoRA + Router, `M2`: + MMD/InfoNCE Alignment, `M3`: + Curriculum Pseudo-Domain) và đầy đủ `B1–B5` + `SmolVLM-500M-Base` (fast pilot) trong Chương 2 & 3 như thiết kế tại `TRAIN/ALT_THREE_SPLITS/configs/` hay không.
4. **Vị trí trình bày kết quả đo khoảng cách miền không gian tiềm ẩn (UMAP, RKHS MMD, Proxy $\mathcal{A}$-distance từ `00_Z0_ZeroShot_and_DomainGap.ipynb`):**
   - *Phương án A (Khuyên dùng):* Đặt tại **Mục `3.2.1`** cùng với điểm neo Zero-Shot (`Z0`) để tránh việc phải giới thiệu kiến trúc `Qwen2-VL`/`SmolVLM` trước Chương 2, đồng thời chỉ phân tích khoảng cách miền ở tầng dữ liệu (hình học, OCR, ngôn ngữ) tại **Mục `1.4`**.
   - *Phương án B:* Tách thêm mục `1.4.3` cho phân tích không gian tiềm ẩn (khi đó phải giới thiệu sơ lược bộ trích xuất đặc trưng trước).

---

## 3. Ma trận Kiểm toán Từng Phần & Tiêu đề của `OFFLIMIT/DATN-template.docx`

> **Quy ước trạng thái:**
> - **KEEP**: Phù hợp tốt với dự án hiện tại, giữ nguyên cấu trúc.
> - **MODIFY**: Giữ lại nhưng cần sửa tiêu đề, bổ sung mục con, hoặc điều chỉnh phạm vi cho khớp mã nguồn/dữ liệu thực tế.
> - **WATCH OUT / TRUNCATE**: Tiềm ẩn rủi ro thiếu kết quả thực nghiệm hoặc trùng lặp; cân nhắc thu gọn hoặc cắt bỏ tùy quyết định ở Mục 2.
> - **Sẵn sàng dữ liệu:** `🟢 Sẵn sàng viết ngay` | `🟡 Chờ hoàn thiện EDA/số liệu phụ` | `🔴 Chờ chạy huấn luyện/đánh giá (TRAIN/EVAL/DEMO)`

### 3.1 Phần Mở đầu & Thủ tục (Front Matter)

| Mục trong `DATN-template.docx` | Đánh giá | Trạng thái | Ghi chú chỉnh sửa & cảnh báo |
| :--- | :---: | :---: | :--- |
| Trang bìa ngoài & Trang bìa trong | **MODIFY** | 🟢 | Sửa lỗi dính chữ `"TÀI LIỆUSANG"` $\rightarrow$ `"TÀI LIỆU SANG"`; điền họ tên, mã SV, ngành, GVHD. |
| LỜI CAM ĐOAN / LỜI CẢM ƠN | **KEEP** | 🟢 | Thủ tục bắt buộc chuẩn PTIT. |
| MỤC LỤC | **KEEP** | 🟢 | Tự động cập nhật bằng Word Field sau khi chốt đề mục. |
| DANH MỤC KHÁI NIỆM VÀ CÁC CHỮ CÁI VIẾT TẮT | **KEEP** | 🟢 | Giữ bảng chuẩn của `OFFLIMIT/DATN-template.docx` (không đổi thành "Danh sách từ khóa"). Bổ sung đầy đủ: `VLM`, `VQA`, `ANLS`, `OCR`, `PEFT`, `LoRA`, `QLoRA`, `NF4`, `MMD`, `InfoNCE`, `M-RoPE`, `pHash`, `DSU`, `MoE`, `TG`, `FR`, `OOD`. |
| DANH MỤC BẢNG BIỂU / DANH MỤC HÌNH VẼ | **KEEP** | 🟢 | Tự động sinh từ Caption (Bảng 1.1, Hình 2.1...). |
| **MỞ ĐẦU** (1. Lý do chọn đề tài; 2. Tổng quan vấn đề; 3. Mục đích; 4. Đối tượng & phạm vi; 5. Phương pháp; 7. Nội dung chính) | **KEEP** | 🟢 | Bám sát chuẩn PTIT định hướng nghiên cứu; đưa trực tiếp 3 câu hỏi nghiên cứu (**RQ1–RQ3**) và đóng góp chính vào mục 3 & 7. |
| **MỞ ĐẦU — 6. Những nghiên cứu liên quan** | **WATCH OUT / MODIFY** | 🟢 | **Cảnh báo trùng lặp với Mục 1.2 & 1.3.** Chỉ viết tóm tắt 1–1.5 trang theo hướng định vị khoảng trống nghiên cứu (research gap), không lặp lại mô tả chi tiết kiến trúc ở Chương 1. |

### 3.2 CHƯƠNG 1: CƠ SỞ LÝ THUYẾT VÀ PHÂN TÍCH KHOẢNG CÁCH MIỀN DỮ LIỆU

| Mục trong `DATN-template.docx` | Đánh giá | Trạng thái | Ghi chú chỉnh sửa & cảnh báo |
| :--- | :---: | :---: | :--- |
| **1.1** Tổng quan về bài toán Hỏi đáp Trực quan dựa trên Văn bản (Text-Centric VQA) | **KEEP** | 🟢 | Rất phù hợp, định vị rõ bối cảnh nghiên cứu. |
| 1.1.1 Bài toán Hỏi đáp trên Ảnh Tài liệu (DocVQA) và Ảnh Cảnh Tự nhiên (TextVQA) | **KEEP** | 🟢 | Trích dẫn chuẩn Singh et al. (2019) & Mathew et al. (2021). |
| 1.1.2 Các thước đo đánh giá chuẩn: ANLS và VQA Accuracy | **MODIFY** | 🟢 | Nên bổ sung thêm định nghĩa ngắn gọn về **độ đo chuyển giao liên miền** dùng xuyên suốt luận văn: **Transfer Gain (TG)**, **Forgetting Rate (FR)**, và **Harmonic Mean** (từ `TRAIN/src/metrics.py`), hoặc ghi rõ các độ đo chuyển giao sẽ trình bày ở 1.3.1. |
| **1.2** Kiến trúc các Mô hình Thị giác – Ngôn ngữ (Vision-Language Models) | **KEEP** | 🟢 | Bố cục tiến hóa lịch sử rất mạch lạc (OCR-Grounded $\rightarrow$ OCR-Free $\rightarrow$ Modern Dynamic VLM). |
| 1.2.1 Trường phái phụ thuộc OCR (OCR-Grounded: M4C, LayoutLMv3, LaTr) | **KEEP** | 🟢 | Có sẵn đầy đủ paper PDF & BibTeX trong `PLAN/Papers/` (STT 3, 4, 9, 10, 15). |
| 1.2.2 Trường phái End-to-End không phụ thuộc OCR (OCR-Free: Donut, Pix2Struct) | **KEEP** | 🟢 | Có sẵn paper (STT 5, 8, 14 — Donut, Pix2Struct, UReader). |
| 1.2.3 Kiến trúc VLM phân giải động hiện đại (Qwen2-VL, SmolVLM / Idefics3, PaliGemma) | **MODIFY** | 🟢 | **Watch-out về PaliGemma:** `PaliGemma` dùng độ phân giải vuông cố định (`224/448/896px`) và không nằm trong nhóm backbone huấn luyện cuối cùng. Nên sửa tiêu đề thành *"Kiến trúc VLM hiện đại xử lý độ phân giải linh hoạt (Qwen2-VL, SmolVLM / Idefics3 và đối chiếu với PaliGemma)"* để tạo luận cứ tự nhiên cho việc chọn Qwen2-VL & SmolVLM ở 2.3.1. |
| **1.3** Chuyển giao Tri thức Liên miền và Tinh chỉnh Tham số Hiệu quả (PEFT) | **KEEP** | 🟢 | Đúng trọng tâm lý thuyết của đề tài. |
| 1.3.1 Bài toán thích nghi miền (Domain Adaptation) và hiện tượng quên tri thức (Catastrophic Forgetting) | **MODIFY** | 🟢 | Bổ sung nền tảng toán học về **đo khoảng cách phân phối miền** (**RKHS MMD**, **Proxy $\mathcal{A}$-distance** $\hat{d}_{\mathcal{A}} = 2(1-2\epsilon)$) và **Contrastive Alignment (InfoNCE)** từ `MATHS/0716_22112026_MMD_vs_InfoNCE_alignment_loss.md` để chuẩn bị cho 2.3.2 & 3.2.1. |
| 1.3.2 Tinh chỉnh hạng thấp (LoRA) và lượng tử hóa 4-bit (QLoRA / NF4) | **KEEP** | 🟢 | Có sẵn paper Hu et al. (2022) & Dettmers et al. (2023) (STT 28, 29). |
| **1.4** Phân tích thực nghiệm khoảng cách miền giữa DocVQA và TextVQA | **KEEP** | 🟢 | Điểm sáng rất mạnh của đề tài (có sẵn số liệu định lượng EDA thực tế). |
| 1.4.1 Sự khác biệt về hình học ảnh, độ phân giải và tỷ lệ khung hình | **KEEP** | 🟢 | Sẵn số liệu chênh lệch $4.8\times$ độ phân giải, tỉ lệ khung hình, màu sắc (`L` vs `RGB/CMYK`). |
| 1.4.2 Sự khác biệt về mật độ token OCR, bố cục không gian và phân phối câu hỏi – câu trả lời | **KEEP** | 🟢 | Sẵn số liệu mật độ OCR ($15\times$), bản đồ nhiệt không gian, phân loại ngữ nghĩa **Extractive vs. Non-extractive** & tỷ lệ đồng thuận annotator từ `ANALYSIS/ALT_THREE_SPLITS/analyze_NEW_splits.ipynb`. |
| **1.5** Tiểu kết chương 1 | **KEEP** | 🟢 | Chốt lại 3 nút thắt kỹ thuật dẫn sang Chương 2. |

### 3.3 CHƯƠNG 2: CHUẨN HÓA DỮ LIỆU CHỐNG RÒ RỈ VÀ THIẾT KẾ KIẾN TRÚC CROSSVQA

| Mục trong `DATN-template.docx` | Đánh giá | Trạng thái | Ghi chú chỉnh sửa & cảnh báo |
| :--- | :---: | :---: | :--- |
| **2.1** Chuẩn hóa hợp đồng dữ liệu đa miền (Unified Canonical Data Schema) | **KEEP** | 🟢 | Phản ánh chính xác Milestones 0–4 đã hoàn thành 100%. |
| 2.1.1 Phân tích bất đồng bộ cấu trúc giữa nguồn dữ liệu gốc DocVQA và TextVQA | **KEEP** | 🟢 | Map 1-1 từ `PLAN/02-15-00_2026-10-05_Data_Schema_Discrepancies_and_Handling_Strategies.md`. |
| 2.1.2 Thiết kế lược đồ dữ liệu hợp nhất và chuẩn hóa tọa độ hộp bao OCR về hệ quy chiếu `[0, 1000]` | **KEEP** | 🟢 | Trích dẫn LayoutLM, Pix2seq, PaLI (STT 15–17) + bảng 15 cột `canonical_schema.py`. |
| 2.1.3 Bảo toàn chất lượng ảnh không tổn hao và phân mảnh luồng Parquet trên Hugging Face | **KEEP** | 🟢 | Trình bày True Lossless PNG (DocVQA), Native JPEG + `exif_transpose` (TextVQA), kiến trúc 3 nhánh (`lossless`, `main`, `three-splits`), ZSTD Parquet + `write_page_index=True`. |
| **2.2** Kiểm toán phân phối quan hệ và giao thức phân tách dữ liệu chống rò rỉ | **KEEP** | 🟢 | Một trong những đóng góp phương pháp luận chặt chẽ nhất của đồ án. |
| 2.2.1 Kiểm toán tính bội quan hệ đa cấp (`ucsf_document_id`, `image_id`, `question_id`) | **KEEP** | 🟢 | Dựa trực tiếp trên `DATA_PIPELINE/Verdict on data distribution.md` + chứng minh toán học rò rỉ khi chia ngẫu nhiên (>53% TextVQA, >85% DocVQA) + Kapoor & Narayanan (2023), Agrawal et al. (2018). |
| 2.2.2 Giao thức phân tách bảo toàn tập kiểm thử chuẩn (`VALIDATION_AS_TEST`) và cơ chế ghim ngoại lai (`+3.0σ`) | **MODIFY** | 🟢 | **Watch-out quan trọng:** Trình bày giao thức này như **Bước thiết kế cơ sở (Baseline Protocol)** (công thức $\text{MoE}_{95\%}$ giải thích mốc 8% validation, quy tắc ghim 66 tài liệu ngoại lai $|z|>3.0$), sau đó **chỉ rõ hạn chế phát hiện được** (523 tài liệu cha vắt ngang giữa `train` và `validation` gốc của DocVQA) để dẫn dắt hợp lý sang 2.2.3! |
| 2.2.3 Giao thức phân tách gộp 3 tập độc lập (`ALT_THREE_SPLITS`) và xử lý 523 tài liệu cầu nối | **KEEP** | 🟢 | Dựa trên `DATA_PIPELINE/ALT_THREE_SPLITS/Verdict on ds splitting decision (Pooled).md` (81 mega-docs được ghim trong tập gộp, 2-stage `GroupShuffleSplit`, phân tách Few-Shot lồng nhau 1%/5%/10%/25%/100% theo `image_id`). |
| 2.2.4 Cổng kiểm soát tương đồng thị giác bằng băm nhận thức 64-bit (`pHash`) và hợp nhất Disjoint Set Union | **KEEP** | 🟢 | Dựa trên `MATHS/0158_11112026_pHash_explanation.md` & `Audit_Visual_Similarity_Pooled.ipynb` (36.156 pHash, $d_H \le 3$, 85 cặp trùng, 76/85 cách ly vào `NEW_train`, 0 rò rỉ liên tập, kiểm định 10/10 tiêu chí). |
| **2.3** Thiết kế kiến trúc chuyển giao tri thức liên miền CrossVQA | **MODIFY** | 🟢 | Cần cập nhật đầy đủ **3 cấu hình đề xuất M1–M3** và **3 backbone khóa cứng** theo `TRAIN/0805_22112026_CrossVQA_Training_Pipeline_Documentation.md`. |
| 2.3.1 Lựa chọn mô hình nền tảng (`Qwen2-VL-2B Base`, `SmolVLM-Base`) và kiểm toán chống ô nhiễm dữ liệu | **MODIFY** | 🟢 | Bổ sung thêm **`SmolVLM-500M-Base`** (vai trò Fast Pilot & Scale Ablation) vào tiêu đề và bảng so sánh kiến trúc (`TRAIN/VLM_Backbones_Technical_Reference.md`), cùng trích dẫn kiểm toán ô nhiễm Wang et al. (2024) (STT 19). |
| 2.3.2 Thiết kế nhánh LoRA Adapter chuyên biệt theo miền và cơ chế điều hướng | **MODIFY** | 🟢 | Hiện có `2.3.2.1` (Adapter $\mathcal{A}_{\text{doc}}, \mathcal{A}_{\text{scene}}$ + Router — **M1**) và `2.3.2.2` (Hàm mục tiêu kết hợp + MMD/InfoNCE + FIFO Queue — **M2**). **Bắt buộc bổ sung mục con `2.3.2.3`**: *"Cầu nối miền giả lập theo giáo trình tăng dần độ khó (Curriculum Pseudo-Domain Bridge — M3)"* (biến đổi phối cảnh 3D homography, bóng đổ ánh sáng, mờ quang học/JPEG ramp $0.2 \rightarrow 1.0$ trong `TRAIN/src/augmentations.py`). |
| 2.3.3 Chiến lược điều kiện hóa đầu vào: Chế độ Không OCR (OCR-Free) và Có hỗ trợ OCR không gian (Given-OCR) | **KEEP** | 🟢 | Trình bày Track A (OCR-Free mặc định), Track B (Given-OCR `max_ocr_tokens=50` cho ablation trên B1/M2), cùng kỹ thuật **Answer-Only Label Masking (`-100`) & EOS discipline** trong `TRAIN/src/collator.py`. |
| 2.3.4 Xây dựng thư viện huấn luyện và đánh giá hợp nhất (`TRAIN/src`) | **MODIFY** | 🟢 | Giữ ngắn gọn ở mức **kiến trúc mô-đun phần mềm** (sơ đồ luồng dữ liệu qua 8 module `dataset`, `collator`, `models`, `adapters`, `losses`, `augmentations`, `trainer`, `metrics` + cơ chế `CheckpointManager` chống ngắt Colab), không chép mã nguồn thô dài dòng như `N16.docx`. |
| **2.4** Tiểu kết chương 2 | **KEEP** | 🟢 | Tổng hợp pipeline khép kín từ Dữ liệu sạch $\rightarrow$ Kiến trúc CrossVQA. |

### 3.4 CHƯƠNG 3: THỰC NGHIỆM, ĐÁNH GIÁ KẾT QUẢ VÀ XÂY DỰNG ỨNG DỤNG MINH HỌA

| Mục trong `DATN-template.docx` | Đánh giá | Trạng thái | Ghi chú chỉnh sửa & cảnh báo |
| :--- | :---: | :---: | :--- |
| **3.1** Môi trường thực nghiệm, siêu tham số và tối ưu hóa bộ nhớ GPU | **KEEP** | 🟢 | Thông số đã được khóa cứng trong `TRAIN/ALT_THREE_SPLITS/configs/*.yaml` — có thể viết ngay! |
| 3.1.1 Thiết lập môi trường huấn luyện và giới hạn ngân sách điểm ảnh động (`min_pixels` / `max_pixels`) | **MODIFY** | 🟢 | Trình bày bảng 4 hồ sơ phần cứng tự động (`l4_24gb` chính, `a100_40gb`, `g4_96gb`, `t4_16gb` dự phòng), công thức token thị giác $N_{\text{vis}} \approx \lceil H/28 \rceil \times \lceil W/28 \rceil$ và siêu tham số LoRA ($r=16, \alpha=32, \text{lr}=2\times 10^{-4}$, 3 seed `42, 123, 999`). |
| 3.1.2 Lượng tử hóa 4-bit NF4 chọn lọc (Selective Vision Exemption) và tối ưu hóa Paged AdamW 8-bit | **KEEP** | 🟢 | Giải thích rõ cơ chế `llm_int8_skip_modules=["visual", "merger"]` / `["vision_model", "connector"]` để bảo toàn khả năng đọc chữ 6–8pt trên DocVQA. |
| **3.2** Kịch bản thực nghiệm và kết quả đánh giá định lượng | **MODIFY** | 🔴 | Chờ kết quả chạy từ 6 notebook `TRAIN/ALT_THREE_SPLITS/notebooks/00..05`. |
| 3.2.1 Kiểm chứng điểm neo Zero-Shot (`Z0`) và độ tinh khiết của mô hình nền tảng | **MODIFY** | 🔴 | Chạy `00_Z0_ZeroShot_and_DomainGap.ipynb`: báo cáo điểm `Z0` + trực quan hóa **UMAP (A8)** + giá trị **RKHS MMD & Proxy $\mathcal{A}$-distance (A9)** kiểm chứng **Giả thuyết H1**. |
| 3.2.2 Kết quả các mô hình cơ sở (Target-Only B1, Source-Only B2, Joint B3, Sequential B4) | **MODIFY** | 🔴 | **Sửa tiêu đề để bổ sung `B5` (Reverse Sequential `Text -> Doc`)** kiểm chứng tính bất đối xứng chiều chuyển giao (`01..03` notebooks). |
| 3.2.3 Kết quả phương pháp đề xuất CrossVQA trên tập phân tách chuẩn (`ALT_THREE_SPLITS`) | **KEEP** | 🔴 | So sánh đầy đủ `M1`, `M2`, `M3` với `Z0`, `B1–B5`. |
| 3.2.3.1 Đánh giá độ chính xác chuyển giao (VQA Accuracy, ANLS) và tỷ lệ chống quên tri thức | **KEEP** | 🔴 | Bổ sung khoảng tin cậy **95% Bootstrap CI ($B=1.000$)** và kiểm định ý nghĩa thống kê (paired permutation $p < 0.05$) từ Phase D (`EVALUATION/`), báo cáo cả `TG`, `FR`, `Harmonic Mean`. |
| 3.2.3.2 Đánh giá hiệu quả trên kịch bản ít mẫu (Few-Shot Learning Curve) | **KEEP** | 🔴 | Chạy `05_FewShot_Scaling_Matrix.ipynb` trên `1%, 5%, 10%, 25%, 100%` (328 $\rightarrow$ 32.762 câu hỏi trong `textvqa_few_shot_summary.json`) so sánh `B1`, `B4`, `M2` (**Giả thuyết H4**). |
| 3.2.3.3 So sánh giữa hai kiến trúc nền tảng `Qwen2-VL-2B Base` và `SmolVLM-Base` | **MODIFY** | 🔴 | Bổ sung cột **`SmolVLM-500M-Base`** để phân tích mở rộng theo quy mô tham số (500M vs 2B) và so sánh cơ chế thị giác (M-RoPE Dynamic Resolution vs Idefics3 Tile Splitting + Pixel Shuffle). |
| 3.2.4 Thực nghiệm đối chứng tác động rò rỉ dữ liệu giữa `VALIDATION_AS_TEST` và `ALT_THREE_SPLITS` | **WATCH OUT / TRUNCATE** | 🔴 | **Cảnh báo rủi ro cao nhất:** `TRAIN/VALIDATION_AS_TEST/` hiện trống (0%). Nếu không có đủ giờ GPU để chạy đối chứng thêm, nên **cắt bỏ mục 3.2.4** (hoặc chỉ chạy 1 cặp nhỏ `B2`/`B4` trên `SmolVLM-500M-Base` để minh chứng mức độ ảo hóa điểm ANLS do 523 tài liệu cầu nối). |
| 3.2.5 Nghiên cứu loại trừ thành phần (Ablation Study): So sánh chế độ OCR-Free và Given-OCR | **MODIFY** | 🔴 | Mở rộng nhẹ tiêu đề thành *"Nghiên cứu loại trừ thành phần (Ablation Study)"* để gộp cả 3 lát cắt ablation có sẵn trong mã nguồn: **(1)** `M1` vs `M2` vs `M3` (tác động của Router, MMD, Curriculum), **(2)** `MMD` vs `InfoNCE` (`--align_type infonce`), và **(3)** `Track A (OCR-Free)` vs `Track B (Given-OCR)` trên `B1` và `M2`. |
| **3.3** Phân tích định tính, phân loại lỗi và xây dựng ứng dụng minh họa | **WATCH OUT** | 🔴 | Phụ thuộc vào Phase D (`EVALUATION/`) và Phase E (`DEMO/`). |
| 3.3.1 Phân tích các trường hợp lỗi điển hình theo khung phân loại lỗi | **KEEP** | 🔴 | Áp dụng **khung phân loại lỗi 7 lớp (`E1–E7`)** trên $\ge 200$ mẫu dự đoán sai (từ `preds_test_*.jsonl`), kết hợp biểu đồ tỷ lệ lỗi và hình ảnh minh họa trực quan (theo phong cách phân tích sâu nguyên nhân lỗi của `N16.docx` mục `4.4.6`). |
| 3.3.2 Xây dựng và triển khai ứng dụng minh họa hỏi đáp trực quan liên miền (`DEMO`) | **WATCH OUT / TRUNCATE** | 🔴 | `DEMO/` hiện trống. Chỉ giữ mục này nếu bạn triển khai ít nhất một ứng dụng **Gradio demo cục bộ / HF Spaces** có chụp màn hình kết quả thực tế (tương tự `N16.docx` mục `4.5`). Nếu không triển khai, **cắt bỏ `3.3.2`** và rút gọn tên Chương 3. |
| **3.4** Tiểu kết chương 3 | **KEEP** | 🔴 | Tổng kết kiểm chứng các giả thuyết H1–H5. |

### 3.5 Phần Kết luận & Tài liệu Tham khảo

| Mục trong `DATN-template.docx` | Đánh giá | Trạng thái | Ghi chú chỉnh sửa & cảnh báo |
| :--- | :---: | :---: | :--- |
| **KẾT QUẢ ĐẠT ĐƯỢC** (A. Các kết quả; B. Nhận xét, đề xuất, khuyến nghị; C. Hướng nghiên cứu tiếp theo) | **KEEP** | 🟡 | Chuẩn cấu trúc PTIT. Mục B cần trình bày thẳng thắn **Mối đe dọa tính hợp lệ (Threats to Validity)** và hạn chế (giới hạn `max_pixels`, phạm vi 2 miền Anh ngữ, chi phí tính toán). Mục C mở rộng sang InfographicVQA, ChartQA, ViDocVQA/tiếng Việt, triển khai FastAPI lượng tử hóa. |
| **DANH MỤC CÁC TÀI LIỆU THAM KHẢO** | **KEEP** | 🟢 | Đã có sẵn **29 bài báo học thuật cốt lõi kèm BibTeX** tại `PLAN/Papers_List.md` (cần bổ sung thêm 2–3 tài liệu nền về MMD — Gretton et al. 2012, Proxy $\mathcal{A}$-distance — Ben-David et al. 2010, InfoNCE — van den Oord et al. 2018, và pHash/DCT). |

---

## 4. Bản đồ Dữ liệu & Nội dung Điền vào Từng Mục Sống sót (Content Population Blueprint)

Dưới đây là danh mục chi tiết **chính xác thông tin, bảng biểu, công thức và hình vẽ** cần điền cho từng mục, gắn trực tiếp với tệp nguồn trong dự án:

### MỞ ĐẦU
- **1. Lý do chọn đề tài & 2. Tổng quan:** Nêu mâu thuẫn giữa nhu cầu VLM đọc hiểu văn bản tổng quát và khoảng cách miền giữa tài liệu 2D phẳng (DocVQA) với ảnh cảnh 3D tự nhiên (TextVQA); vấn nạn rò rỉ dữ liệu & quên tri thức (`PLAN/ThesisPremise.md`, `PLAN/feedback.md`).
- **3. Mục đích nghiên cứu:** Trình bày rõ **3 Câu hỏi Nghiên cứu (RQ1–RQ3)** từ `README.md` và **5 Giả thuyết (H1–H5)**.
- **4. Đối tượng & phạm vi:** Tập dữ liệu `gamusa/docvqa-canonical` (44.812 câu hỏi có nhãn) & `gamusa/textvqa-canonical` (39.602 câu hỏi có nhãn); 3 mô hình nền tảng Base (`Qwen2-VL-2B`, `SmolVLM-Base`, `SmolVLM-500M-Base`); phần cứng GPU học thuật 16–24 GB VRAM (Colab L4/T4).
- **5. Phương pháp nghiên cứu:** Kết hợp phân tích thống kê EDA, kỹ thuật dữ liệu chống rò rỉ (Group-First + pHash/DSU), thiết kế kiến trúc PEFT (Dual-LoRA + Soft Router + MMD + Curriculum), và thực nghiệm định lượng đa seed + kiểm định thống kê Bootstrap.
- **6. Những nghiên cứu liên quan (rút gọn):** 1 bảng tóm tắt định vị 4 hướng (OCR-based, OCR-free, Modern VLM, Cross-Domain/Data Leakage) chỉ rõ khoảng trống mà CrossVQA lấp đầy.
- **7. Nội dung chính của đề án:** Sơ lược 3 chương.

### CHƯƠNG 1: CƠ SỞ LÝ THUYẾT VÀ PHÂN TÍCH KHOẢNG CÁCH MIỀN DỮ LIỆU
- **1.1.1:** Định nghĩa bài toán VQA, đặc trưng DocVQA (`UCSF Industry Documents Library`, 50.000 Q / 12.767 ảnh gốc; 44.812 Q có nhãn) và TextVQA (`OpenImages`, 45.336 Q / 28.408 ảnh gốc; 39.602 Q có nhãn). Nguồn: `PLAN/ThesisPremise.md`, `PLAN/Papers_List.md` (STT 1, 2).
- **1.1.2:** Công thức toán học chuẩn:
  - **VQA Accuracy:** $\text{Acc}(\hat{y}) = \min\left(\frac{\sum_{j=1}^{10} \mathbb{I}(\hat{y} = y_j)}{3}, 1\right)$
  - **ANLS:** $\text{NLD}(s, r) = \frac{d_L(s, r)}{\max(|s|, |r|)}$, ngưỡng $\tau = 0.5$
  - **Độ đo chuyển giao:** $\text{TG} = \text{Score}_{\text{Model}} - \text{Score}_{\text{B1}}$, $\text{FR} = \text{ANLS}_{\text{B2}}(\text{Doc}) - \text{ANLS}_{\text{Model}}(\text{Doc})$, và Harmonic Mean. Nguồn: `TRAIN/src/metrics.py`.
- **1.2.1–1.2.3:** Tổng hợp có hệ thống từ `PLAN/Papers_List.md` (STT 3–17, 22–27) và `TRAIN/VLM_Backbones_Technical_Reference.md`. Kèm **Bảng 1.1: So sánh các trường phái kiến trúc VLM cho Text-Centric VQA** (độ phân giải, phụ thuộc OCR, cơ chế không gian, ưu/nhược điểm).
- **1.3.1–1.3.2:** Lý thuyết Domain Adaptation, Negative Transfer, Catastrophic Forgetting; công thức **Multi-Kernel RBF MMD**, **Proxy $\mathcal{A}$-distance** $\hat{d}_{\mathcal{A}} = 2(1 - 2\epsilon)$, **InfoNCE** (`MATHS/0716_22112026_MMD_vs_InfoNCE_alignment_loss.md`); công thức phân rã hạng thấp **LoRA** $h = W_0 x + \frac{\alpha}{r} BAx$ và lượng tử hóa **QLoRA / NF4 + Double Quantization** (`PLAN/Papers_List.md` STT 28, 29).
- **1.4.1 & 1.4.2 (Số liệu thực nghiệm sẵn có):**
  - *Hình học & độ phân giải:* DocVQA ~3.7 MP (trung vị $1698 \times 2214$ px, cực đại tới $6921 \times 7184$ px, chủ yếu ảnh xám mode `L`) vs TextVQA ~0.78 MP (trung vị $1024 \times 768$ px, chủ yếu `RGB` + một phần `L`/`CMYK`) $\rightarrow$ **chênh lệch độ phân giải $4.8\times$** (`ANALYSIS/reports/gamusa__docvqa-canonical.md`, `ANALYSIS/reports/gamusa__textvqa-canonical.md`, `ANALYSIS/CrossVQA_Comparative.ipynb`).
  - *Mật độ OCR & bố cục:* DocVQA trung vị ~164–174 token OCR/trang (TB ~180–191, tối đa 1.257) vs TextVQA trung vị 8–12 token/ảnh (TB ~12.9, tối đa 100, tỷ lệ Zero-OCR ~2.4%) $\rightarrow$ **chênh lệch mật độ chữ $\approx 15\times$**; phân bố không gian dạng lưới đều (DocVQA) vs cụm trung tâm (TextVQA).
  - *Ngữ nghĩa câu hỏi – câu trả lời:* Tỷ lệ câu trả lời trích xuất được từ OCR (**Extractive vs. Non-extractive** — sử dụng thuật ngữ mới `"Non-extractive"` đã cập nhật trong `ANALYSIS/ALT_THREE_SPLITS/analyze_NEW_splits.ipynb`), hình `ANALYSIS/figures/OCR grounding - overlap between OCR tokens and answer tokens.png`, độ đồng thuận annotator (10 nhãn TextVQA vs 1–6 nhãn DocVQA).

### CHƯƠNG 2: CHUẨN HÓA DỮ LIỆU CHỐNG RÒ RỈ VÀ THIẾT KẾ KIẾN TRÚC CROSSVQA
- **2.1.1–2.1.3:**
  - **Bảng 2.1: Ma trận đối chiếu 11 chiều bất đồng cấu trúc** giữa DocVQA và TextVQA (từ `PLAN/02-15-00_2026-10-05_Data_Schema_Discrepancies_and_Handling_Strategies.md`).
  - **Bảng 2.2: Lược đồ 15 cột hợp nhất (`canonical_schema.py`)** + công thức chuẩn hóa hộp bao Rosetta $[x,y,w,h]$ và MS Azure 8-điểm về $[y_{\min}, x_{\min}, y_{\max}, x_{\max}] \in [0, 1000]$ (trích dẫn LayoutLM, Pix2seq, PaLI).
  - Quy trình **True Lossless PNG** (DocVQA), **Native JPEG bit-for-bit + `ImageOps.exif_transpose()`** (TextVQA), phân mảnh ZSTD Parquet (`write_page_index=True`), và **kiến trúc nhánh trên Hugging Face Hub** (`lossless`: 95.336 mẫu / 17.00 GB $\rightarrow$ `main`: 84.414 mẫu / 14.99 GB $\rightarrow$ `three-splits`: phân tách 3 tập chống rò rỉ).
- **2.2.1–2.2.4:**
  - **Hình 2.x:** Sơ đồ phân cấp quan hệ 3 tầng DocVQA (`ucsf_document_id` 5.687 $\rightarrow$ `image_id` 11.480 $\rightarrow$ `question_id` 44.812; 31.69% tài liệu đa trang, tối đa 140 trang & 762 câu hỏi) vs 2 tầng TextVQA (`image_id` 25.119 $\rightarrow$ `question_id` 39.602, tối đa 2 câu/ảnh). Nguồn: `DATA_PIPELINE/Verdict on data distribution.md` & các hình trong `DATA_PIPELINE/figures/`.
  - Chứng minh xác suất rò rỉ khi chia ngẫu nhiên (>53% TextVQA, >85% DocVQA) + công thức $\text{MoE}_{95\%} \le 0.98 / \sqrt{N}$ (`MATHS/0026_11112026_explain_MoE_formulas.md`) + hình `DATA_PIPELINE/figures/Split decision - NEW_validation size vs Margin of Error (95% confidence).png`.
  - Cơ chế ghim ngoại lai $|z| > 3.0$ (66 mega-docs ở `VALIDATION_AS_TEST` $\rightarrow$ 81 mega-docs / 7.767 câu hỏi ở `ALT_THREE_SPLITS`).
  - **Phát hiện 523 tài liệu cầu nối** (13.798 câu hỏi, 9.20% tổng số tài liệu cha DocVQA) vắt ngang `train`–`validation` gốc $\rightarrow$ động lực thiết kế `ALT_THREE_SPLITS`.
  - Thuật toán **64-bit DCT `pHash`** + khoảng cách Hamming $d_H \le 3$ + hợp nhất đồ thị **Disjoint Set Union (DSU)** (`MATHS/0158_11112026_pHash_explanation.md`): 36.156 mã băm, rà soát >118 triệu cặp ảnh, phát hiện 85 cặp nhóm gần trùng, cách ly 76/85 vào `NEW_train`, **0 rò rỉ liên tập**, đạt **10/10 tiêu chí kiểm định độc lập**.
  - **Bảng 2.3: Ma trận phân tách cuối cùng `ALT_THREE_SPLITS`** (DocVQA: 38.000 / 3.141 / 3.671; TextVQA: 32.762 / 2.860 / 3.980; 95% MoE $\le \pm 1.83\%$) + Hình `DATA_PIPELINE/ALT_THREE_SPLITS/figures/polars_final_partition_summary.png` + bảng phân tách Few-Shot lồng nhau (`textvqa_few_shot_summary.json`).
- **2.3.1–2.3.4:**
  - **Bảng 2.4:** So sánh 3 backbone khóa cứng (`Qwen/Qwen2-VL-2B`, `HuggingFaceTB/SmolVLM-Base`, `HuggingFaceTB/SmolVLM-500M-Base`) về số lớp ViT, Connector (2×2 Merger + M-RoPE vs Pixel Shuffle + Tiling), LLM decoder, và vệ sinh chống ô nhiễm (dùng bản Base thay vì Instruct).
  - **Sơ đồ kiến trúc tổng thể CrossVQA** (chuyển thể từ sơ đồ Mermaid trong `TRAIN/0805_22112026_CrossVQA_Training_Pipeline_Documentation.md`).
  - **2.3.2.1 (M1 — DualLoRALinear & VisualSoftDomainRouter):** Công thức phối trộn nhánh theo mẫu:
    $$y = W_0 x + w_{\text{doc}}(h_{\text{vis}})\frac{\alpha}{r}B_{\text{doc}}A_{\text{doc}}(\text{drop}(x)) + w_{\text{scene}}(h_{\text{vis}})\frac{\alpha}{r}B_{\text{scene}}A_{\text{scene}}(\text{drop}(x))$$
    cùng chiến lược huấn luyện 2 pha đóng băng $\mathcal{A}_{\text{doc}}$ ở Pha 2 để chặn quên tri thức.
  - **2.3.2.2 (M2 — Hàm mục tiêu kết hợp & Căn chỉnh đặc trưng):**
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{gen}} + \lambda_{\text{align}}\mathcal{L}_{\text{align}} + \lambda_{\text{gate}}\mathcal{L}_{\text{gate}}$$
    với Multi-Kernel RBF MMD ($\sigma \in \{0.5, 1, 2, 4, 8\}$) + hàng đợi **FIFO `FeatureQueue` (`size=256`)** khắc phục batch size nhỏ trên GPU 16–24 GB, và biến thể đối chứng **InfoNCE** ($\tau=0.07$).
  - **2.3.2.3 (M3 — Curriculum Pseudo-Domain Bridge):** Chèn pha trung gian áp dụng `CurriculumPseudoDomainTransform` lên ảnh DocVQA (biến đổi phối cảnh 3D homography, gradient chiếu sáng/bóng đổ, mờ quang học + nhiễu nén JPEG) với độ khó tăng tuyến tính $0.2 \rightarrow 1.0$.
  - **2.3.3:** Hai luồng điều kiện hóa **Track A (OCR-Free)** & **Track B (Given-OCR, top 50 token)** + cơ chế **Answer-Only Label Masking (`-100`)** đảm bảo hàm loss chỉ tối ưu token câu trả lời và `EOS`.
  - **2.3.4:** Sơ đồ kiến trúc 8 module của `TRAIN/src/` và cơ chế chịu lỗi Colab (`CheckpointManager` đồng bộ HF Hub private + Google Drive).

### CHƯƠNG 3: THỰC NGHIỆM, ĐÁNH GIÁ KẾT QUẢ (VÀ ỨNG DỤNG MINH HỌA)
- **3.1.1 & 3.1.2 (Sẵn sàng viết ngay):** Bảng hồ sơ phần cứng (`l4_24gb`, `a100_40gb`, `g4_96gb`, `t4_16gb`), công thức ngân sách token ảnh $N_{\text{vis}} \approx \lceil H/28 \rceil \times \lceil W/28 \rceil$, cấu hình `min_pixels=200,704`, `max_pixels=1,310,720` (L4) / `1,048,576` (T4), bảng siêu tham số chuẩn (`lora_r=16`, `lora_alpha=32`, `lr=2e-4`, `effective_batch=16`, 3 seeds `42, 123, 999`), và luận giải kỹ thuật Selective 4-bit NF4 (`llm_int8_skip_modules`).
- **3.2.1–3.2.5 (Điền sau khi chạy `notebooks/00..05` & `EVALUATION/`):**
  - *3.2.1:* Bảng điểm `Z0` (DocVQA ANLS, TextVQA VQA Acc) + hình UMAP 2D (`A8`) + bảng chỉ số **RKHS MMD** và **Proxy $\mathcal{A}$-distance** (`A9`).
  - *3.2.2 & 3.2.3:* Bảng tổng hợp toàn ma trận thực nghiệm **`Z0`, `B1`, `B2`, `B3`, `B4`, `B5`, `M1`, `M2`, `M3`** (mean $\pm$ std qua 3 seed + 95% Bootstrap CI, chỉ số `ANLS`, `VQA Acc`, `Harmonic Mean`, `TG`, `FR`); biểu đồ đường cong Few-Shot (`1%–100%`) so sánh `B1`, `B4`, `M2`; bảng so sánh 3 backbone (`Qwen2-VL-2B`, `SmolVLM-Base`, `SmolVLM-500M-Base`).
  - *3.2.4:* (Tùy quyết định mục 2) Đối chứng rò rỉ hoặc lược bỏ.
  - *3.2.5:* Bảng Ablation 3 chiều: đóng góp từng module (`M1` $\rightarrow$ `M2` $\rightarrow$ `M3`), hàm căn chỉnh (`MMD` vs `InfoNCE`), và điều kiện đầu vào (`OCR-Free` vs `Given-OCR`).
- **3.3.1 (Điền sau Phase D):** Phân tích định tính + thống kê phân bố **7 nhóm lỗi (`E1–E7`)** trên tập mẫu lỗi ($\ge 200$ mẫu), kèm 4–6 ví dụ hình ảnh minh họa trước/sau chuyển giao và trọng số định tuyến $(w_{\text{doc}}, w_{\text{scene}})$.
- **3.3.2 (Tùy quyết định mục 2):** Kiến trúc & ảnh chụp giao diện Demo Gradio tương tác.

---

## 5. Quy chuẩn Văn phong Tiếng Việt Học thuật (Rút kinh nghiệm từ `OFFLIMIT/N16.docx`)

Đối chiếu `OFFLIMIT/N16.docx` (báo cáo học phần tiếng Việt tại PTIT) giúp định hình tiêu chuẩn viết tự nhiên nhưng nâng lên tầm **Đồ án Tốt nghiệp Định hướng Nghiên cứu**:

| Khía cạnh | Điểm mạnh cần kế thừa từ `OFFLIMIT/N16.docx` | Điểm cần tránh / Nâng cấp cho ĐATN PTIT |
| :--- | :--- | :--- |
| **Thuật ngữ song ngữ** | Giữ văn phong tiếng Việt tự nhiên, chú thích thuật ngữ tiếng Anh trong ngoặc ở lần xuất hiện đầu tiên (vd: *"quên tri thức thảm khốc (catastrophic forgetting)"*, *"thích nghi miền (domain adaptation)"*). | Không lạm dụng từ ghép nửa Anh nửa Việt thiếu nhất quán; lập bảng quy ước thuật ngữ ngay từ đầu. |
| **Cấu trúc Bảng $\rightarrow$ Biện luận** | Mọi bảng số liệu đều có phần phân tích sâu ngay phía dưới; đặc biệt học hỏi phong cách mục `4.4.6` (*mổ xẻ 5 nguyên nhân vì sao mô hình A kém mô hình B*). | Với CrossVQA, áp dụng lối biện luận này để giải thích *vì sao* `B3` bị chuyển giao tiêu cực, *vì sao* `B4` suy giảm ANLS mạnh, và *vì sao* `M2`/`M3` khắc phục được. |
| **Trình bày kỹ thuật** | Liên kết chặt chẽ giữa kết quả EDA và lựa chọn siêu tham số (như N16 dùng P99=60 token để chọn `max_length=128`). | **Tuyệt đối không dán khối mã Python thô dài dòng** như N16 (mục 4.3.1–4.3.3). Thay bằng **công thức toán học, sơ đồ khối kiến trúc, thuật toán giả mã (pseudocode), và bảng siêu tham số**. |
| **Chuẩn định dạng số & trích dẫn** | — | Sửa triệt để lỗi của N16: (1) Thống nhất 1 chuẩn dấu thập phân (nên dùng chuẩn tiếng Việt `0,85` hoặc nhất quán dấu chấm `.` kiểu khoa học trong toàn bài), (2) Rà soát đánh số đề mục tự động, (3) Trích dẫn đầy đủ $\ge 30$ tài liệu học thuật chuẩn IEEE/APA từ `PLAN/Papers_List.md`. |

---

## 6. Kế hoạch Triển khai & Kiểm chứng (Sau khi Người dùng Phê duyệt)

### Giai đoạn 1 — Chuẩn hóa Đề cương & Bản thảo Sẵn sàng (Làm được ngay)
1. Cập nhật lại cấu trúc đề mục trên bản sao mới của mẫu báo cáo (giữ nguyên `OFFLIMIT/DATN-template.docx` nguyên trạng; sửa trên tệp làm việc như `DATN-CrossVQA-template.docx` hoặc bản thảo Markdown/Word riêng theo chỉ định của người dùng), sửa lỗi `"TÀI LIỆUSANG"`, thêm `M3`, `B5`, `SmolVLM-500M-Base`, và điều chỉnh `3.2.4` / `3.3.2` theo lựa chọn đã chốt.
2. Soạn thảo hoàn chỉnh các phần đã đủ 100% dữ liệu thực tế: **MỞ ĐẦU**, **CHƯƠNG 1**, **CHƯƠNG 2**, **Mục 3.1**, **Danh mục chữ viết tắt**, và **Danh mục tài liệu tham khảo**.

### Giai đoạn 2 — Hoàn thiện Chương 3 & Kết luận (Sau khi có kết quả `TRAIN/` & `EVALUATION/`)
1. Thu thập toàn bộ tệp `val_metrics.json`, `run_card.yaml`, `preds_*.jsonl` và biểu đồ UMAP từ `TRAIN/ALT_THREE_SPLITS/runs/`.
2. Chạy kiểm định thống kê (95% Bootstrap CI, permutation test) và gán nhãn phân loại lỗi `E1–E7` cho `3.3.1`.
3. Điền hoàn thiện `3.2`, `3.3`, `3.4` và phần **KẾT QUẢ ĐẠT ĐƯỢC**.

### Kế hoạch Kiểm chứng (Verification Plan)
- **Kiểm tra chéo số liệu tự động:** Đối chiếu mọi con số trong Chương 1 & Chương 2 (số mẫu 84.414, 38.000/3.141/3.671, 32.762/2.860/3.980, 523 tài liệu cầu nối, 85 cặp pHash, 15 cột schema, siêu tham số YAML) khớp 100% với `DATA_PIPELINE/ALT_THREE_SPLITS/splits/crossvqa_split_manifest_pooled.parquet` và `TRAIN/ALT_THREE_SPLITS/configs/*.yaml`.
- **Kiểm tra cấu trúc & hình thức thủ công:** Xác nhận bảng Mục lục (TOC), Danh mục Bảng biểu, Danh mục Hình vẽ, thứ tự đánh số đề mục, chuẩn trích dẫn tài liệu tham khảo và quy ước dấu thập phân đồng nhất trên toàn văn bản.
