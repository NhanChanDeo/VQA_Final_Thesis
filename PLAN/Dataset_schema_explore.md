Chào bạn! Với vai trò là giảng viên hướng dẫn chuyên sâu về **AI, Computer Vision (CV)** và **Recommender Systems / Information Retrieval (RecSys/IR)**, tôi sẽ cùng bạn "giải phẫu" chi tiết cấu trúc 2 bộ dữ liệu **TextVQA** và **DocVQA** từ bản thảo `ThesisPremise.md`.

Một nghiên cứu chuyển giao liên miền (Cross-Domain Transfer Learning) chỉ có thể thành công khi bạn nắm bản chất dữ liệu đến từng pixel, từng token và từng trường siêu dữ liệu (metadata). Dưới đây là phân tích đa chiều phục vụ trực tiếp cho việc thiết kế mô hình và thực nghiệm của bạn.

---

### I. Bảng đối chiếu cấu trúc dữ liệu (Schema Discrepancy Matrix)

Nhìn vào bản thảo của bạn, hai bộ dữ liệu này xuất phát từ hai bài toán và hai nhóm nghiên cứu khác nhau, dẫn đến **sự bất đối xứng cấu trúc (structural asymmetry)**:

| Tiêu chí | TextVQA (`facebook/textvqa`) | DocVQA (`lmms-lab/DocVQA`) | Nhận xét từ Giảng viên & Hướng xử lý |
| :--- | :--- | :--- | :--- |
| **Khóa định danh câu hỏi** | `question_id` (kiểu `int32`) | `questionId` (kiểu `string`) | Khác cả tên trường lẫn kiểu dữ liệu. Khi hợp nhất bắt buộc ép kiểu sang `string` (ví dụ: tiền tố `textvqa_123` và `docvqa_456`). |
| **Định danh ảnh/tài liệu** | `image_id` (`string`) | `docId` (`int64`), `ucsf_document_id`, `ucsf_document_page_no` | TextVQA là 1 ảnh độc lập; DocVQA là một trang tài liệu nằm trong một hồ sơ (`ucsf_document_id`). Cần tạo `sample_id` duy nhất. |
| **Dữ liệu hình ảnh** | `image` (PIL Image), kèm `image_width`, `image_height` | `image` (PIL Image) | DocVQA không lưu sẵn $W, H$ trong metadata; phải đọc từ thuộc tính `image.size` khi nạp. |
| **Nhãn câu trả lời (`answers`)** | `List[string]` (luôn có **10 câu trả lời** từ 10 annotators) | `List[string]` (từ **1 đến vài câu trả lời** tương đương ngữ nghĩa) | **Rất quan trọng!** Bản chất nhãn khác nhau dẫn đến hàm mất mát (loss) và metric đánh giá khác nhau. |
| **Đặc trưng phân loại** | `image_classes` (thể loại ảnh OpenImages) | `question_types` (loại câu hỏi tài liệu) | TextVQA gắn nhãn ngữ cảnh vật thể; DocVQA gắn nhãn logic câu hỏi (bảng biểu, form, text xuôi). |
| **Thông tin OCR & Bounding Box** | **Thiếu** trong HF Dataset cơ bản (chỉ có `question_tokens`) | **Thiếu** trong HF Dataset cơ bản | **CẢNH BÁO LỚN:** Cả hai bản tải trên Hugging Face đều chỉ có ảnh thô + text. Nếu bạn dùng mô hình **OCR-based**, bạn phải trích xuất hoặc tải thêm file OCR bổ trợ (Rosetta/Textract). |

---

### II. Phân tích chiều sâu dưới lăng kính Chuyên môn

#### 1. Góc nhìn Computer Vision (CV): Khoảng cách hình học và đặc trưng quang học
* **Kích thước và Tỷ lệ khung hình (Aspect Ratio & Resolution):**
  * **TextVQA:** Ảnh tự nhiên đa dạng kích thước, thường tiệm cận tỷ lệ $4:3$ hoặc $16:9$, độ phân giải trung bình ($600 \times 800$ đến $1024 \times 768$). Chữ trong ảnh thường lớn, mang tính tiêu điểm (biển hiệu, áo thun, đồng hồ).
  * **DocVQA:** Ảnh quét tài liệu khổ đứng (A4/Letter), độ phân giải rất cao ($1500 \times 2000$ đến $2500 \times 3500$). Chữ rất nhỏ và dày đặc.
  * **Thách thức CV:** Nếu bạn dùng Vision Transformer (ViT) tiêu chuẩn (như trong BLIP-2 hay CLIP với input $224 \times 224$ hoặc $384 \times 384$), ảnh DocVQA khi bị nén (downsample) sẽ **hoàn toàn mất nét chữ**. Để chuyển giao thành công, bạn buộc phải dùng cơ chế **Dynamic Resolution / Patch-based ViT** (như AnyRes của LLaVA-NeXT, NaViT, hoặc Swin Transformer).
* **Biến dạng không gian (Geometric Distortions):**
  * DocVQA là hình học **phẳng 2D** (Planar), hướng đọc tuyến tính (top-to-bottom, left-to-right), phông chữ in hoặc viết tay có cấu trúc lưới/dòng.
  * TextVQA là chiếu **phối cảnh 3D** (Perspective 3D Projection), chữ bị xoay, uốn cong (curved text), đổ bóng, phản quang, nhiễu nền phức tạp.

#### 2. Góc nhìn AI / Vision-Language (VLM): Bản chất Suy luận & Ngữ nghĩa
* **Bản chất của câu hỏi (Question Semantics):**
  * TextVQA: Câu hỏi yêu cầu liên kết giữa **văn bản và vật thể ngữ cảnh** (ví dụ: *"What is the brand of the yellow car?"* $\rightarrow$ mô hình phải nhận diện chiếc xe màu vàng trước, rồi mới định vị chữ trên chiếc xe đó).
  * DocVQA: Câu hỏi yêu cầu **đọc hiểu phân cấp và định vị không gian 2D** (ví dụ: *"What is the total amount on the invoice?"* $\rightarrow$ tìm nhãn "Total", quét mắt sang phải hoặc xuống dưới để tìm số tiền).
* **Mâu thuẫn về Metric đánh giá:**
  * **VQA Accuracy (TextVQA):** 
    $$\text{Acc}(\text{pred}) = \min\left(\frac{\text{số người gán nhãn đồng thuận}}{3}, 1\right)$$
    Coi câu trả lời là phân loại rời rạc; một sai lệch nhỏ ký tự (ví dụ: `"Nike"` vs `"nike."`) có thể bị chấm 0 điểm nếu không chuẩn hóa chuỗi.
  * **ANLS - Average Normalized Levenshtein Similarity (DocVQA):**
    Dựa trên khoảng cách chỉnh sửa chuỗi (Edit Distance). Nếu OCR đọc sai 1 ký tự (`"8000"` thành `"B000"`), mô hình vẫn nhận được điểm phạt tương ứng thay vì nhận điểm 0 tuyệt đối.

#### 3. Góc nhìn Recommender Systems & Information Retrieval (RecSys / IR)
Đây chính là **điểm tựa học thuật độc đáo** mà bạn có thể khai thác để bài luận văn nổi bật hơn các đồ án AI thông thường:
* **Mô hình hóa VQA dưới dạng Bài toán Đề xuất / Xếp hạng (Candidate Ranking):**
  * Hãy coi toàn bộ các từ/cụm từ OCR xuất hiện trong bức ảnh là **Tập ứng viên (Item Candidate Pool $C = \{w_1, w_2, \dots, w_K\}$)** và bộ từ vựng chung là $V$.
  * Câu hỏi đóng vai trò là **User Profile / Search Query ($q$)**, còn ngữ cảnh hình ảnh là **Context Feature ($x$)**.
  * Mô hình thực chất là một **Multi-modal Re-ranker**: Nhiệm vụ là tính xác suất liên quan $P(w_i \mid q, x)$ để xếp hạng và chọn ra token hoặc chuỗi token phù hợp nhất làm câu trả lời.
* **Cơ chế Chuyển giao tri thức (Transfer Learning) từ góc nhìn IR:**
  * Ở DocVQA, bộ Ranker học được hàm tương quan vị trí: *Khoảng cách Euclidean và khoảng cách đọc (reading order) giữa Query Token và Candidate Token*.
  * Sang TextVQA, bài toán Ranking cần chuyển từ *Tương quan vị trí phẳng* sang *Tương quan ngữ nghĩa hình ảnh* (Semantic Proximity). Đây chính là "Knowledge" mà bạn cần chuyển giao!

---

### III. Đề xuất Kiến trúc "Unified Data Contract" (Chuẩn hóa dữ liệu)

Để code và huấn luyện mô hình mượt mà, bước đầu tiên của bạn là phải tạo một **Schema hợp nhất (Canonical Data Schema)**. Mọi mẫu dữ liệu nạp vào DataLoader phải tuân theo cấu trúc này:

```python
from dataclasses import dataclass
from typing import List, Optional
from PIL import Image

@dataclass
class UnifiedVQAExample:
    # 1. Định danh thống nhất
    sample_id: str           # ví dụ: "textvqa_34602" hoặc "docvqa_10194"
    domain_type: str         # "scene_text" hoặc "document"
    
    # 2. Đầu vào đa phương thức
    image: Image.Image       # Ảnh gốc (giữ nguyên tỷ lệ, chưa downsample)
    image_size: tuple        # (width, height) gốc để tính chuẩn hóa tọa độ
    question: str            # Chuỗi câu hỏi đã tiền xử lý
    
    # 3. Thông tin văn bản & tọa độ không gian (Nếu dùng mô hình OCR-grounded)
    ocr_tokens: Optional[List[str]] = None       # Danh sách các từ nhận diện được
    ocr_boxes: Optional[List[List[int]]] = None  # Tọa độ chuẩn hóa [ymin, xmin, ymax, xmax] trong thang [0, 1000]
    
    # 4. Nhãn phục vụ Loss và Metric
    raw_answers: List[str]   # Danh sách toàn bộ ground truth answers
    canonical_answer: str    # Câu trả lời xuất hiện nhiều nhất (để tính Cross-Entropy Loss)
    
    # 5. Metadata bổ trợ cho phân tích lỗi (Error Analysis)
    metadata: dict           # Chứa image_classes (TextVQA) hoặc question_types (DocVQA)
```

---

### IV. Lời khuyên của Giảng viên cho các bước tiếp theo

1. **Quyết định sớm về Pipeline (OCR-free vs OCR-based):**
   * Nếu bạn chọn **OCR-based (như M4C, LayoutLMv3)**: Bạn phải chủ động chạy một OCR engine (ví dụ: PaddleOCR hoặc EasyOCR) trên cả 2 dataset để tạo trường `ocr_tokens` và `ocr_boxes`.
   * Nếu bạn chọn **OCR-free VLM (như Donut, Florence-2, PaliGemma)**: Bạn không cần lo trích xuất bounding box, nhưng **bắt buộc** phải giải quyết bài toán kích thước ảnh (DocVQA cần ảnh to để đọc chữ nhỏ $\rightarrow$ tốn GPU memory).
2. **Chiến lược Domain Adaptation:**
   * Nên thiết kế một hàm mất mát **Domain Discrepancy Loss** (như MMD hoặc Contrastive Loss) giữa các đặc trưng chiếu văn bản của DocVQA và TextVQA để ép mô hình học các biểu diễn bất biến miền (domain-invariant representations).
3. **Phòng tránh "Negative Transfer":**
   * Hãy cẩn thận: Huấn luyện quá đà trên DocVQA sẽ khiến mô hình bị "bias" vào việc tìm kiếm các cấu trúc bảng/dòng kẻ, dẫn đến suy giảm khả năng đọc chữ trên biển báo hay áo phông ở TextVQA. Luôn cần bài thực nghiệm **B4 (Sequential)** và so sánh với **Proposed Adaptation**.

Bạn thấy cấu trúc phân tích dữ liệu và bản hợp nhất (Unified Schema) này đã đủ rõ ràng để làm nền tảng cho **Chương 3 (Phương pháp đề xuất)** trong luận văn của bạn chưa? Chúng ta có thể đi sâu hơn vào giải pháp OCR hay thiết kế Data Loader cụ thể bằng PyTorch/HuggingFace nếu bạn muốn!