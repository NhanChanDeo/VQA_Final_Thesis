# Danh Mục Bài Báo Nghiên Cứu (Downloaded Papers for CrossVQA)

> **Thư mục lưu trữ:** [`PLAN/Papers`](file:///c:/Users/tungb/OneDrive%20-%20ptit.edu.vn/PTIT_BuuChinhVienThong/SOTSUGYOU%20pending/PLAN/Papers)  
> **Đề tài:** CrossVQA: Nghiên cứu chuyển giao tri thức liên miền từ ảnh tài liệu sang ảnh cảnh tự nhiên cho mô hình Thị giác–Ngôn ngữ  
> **Bộ dữ liệu chính:** TextVQA (`facebook/textvqa`) & DocVQA (`lmms-lab/DocVQA`)  
> **Thời điểm cập nhật:** 24/09/2026  

---

## 1. Bảng Tổng Hợp Các Bài Báo Đã Tải

| STT | Tên Tệp PDF | Tiêu Đề Bài Báo | Tác Giả & Năm | Nguồn / Hội Nghị | arXiv ID | Vai Trò trong Luận Văn |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | [`Singh_2019_TextVQA.pdf`](file:///c:/Users/tungb/OneDrive%20-%20ptit.edu.vn/PTIT_BuuChinhVienThong/SOTSUGYOU%20pending/PLAN/Papers/Singh_2019_TextVQA.pdf) | Towards VQA Models That Can Read | Amanpreet Singh et al. (2019) | CVPR 2019 | [`1904.08920`](https://arxiv.org/abs/1904.08920) | **Target Domain**: Bộ dữ liệu & chuẩn đánh giá TextVQA, công thức VQA Accuracy |
| **2** | [`Mathew_2021_DocVQA.pdf`](file:///c:/Users/tungb/OneDrive%20-%20ptit.edu.vn/PTIT_BuuChinhVienThong/SOTSUGYOU%20pending/PLAN/Papers/Mathew_2021_DocVQA.pdf) | DocVQA: A Dataset for VQA on Document Images | Minesh Mathew et al. (2021) | WACV 2021 | [`2007.07394`](https://arxiv.org/abs/2007.07394) | **Source Domain**: Bộ dữ liệu DocVQA văn bản phẳng 2D, công thức đo ANLS |
| **3** | [`Hu_2020_M4C.pdf`](file:///c:/Users/tungb/OneDrive%20-%20ptit.edu.vn/PTIT_BuuChinhVienThong/SOTSUGYOU%20pending/PLAN/Papers/Hu_2020_M4C.pdf) | Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA (M4C) | Ronghang Hu et al. (2020) | CVPR 2020 | [`1911.06258`](https://arxiv.org/abs/1911.06258) | **OCR-based Baseline**: Kiến trúc Pointer Network xếp hạng / sao chép từ OCR |
| **4** | [`Biten_2022_LaTr.pdf`](file:///c:/Users/tungb/OneDrive%20-%20ptit.edu.vn/PTIT_BuuChinhVienThong/SOTSUGYOU%20pending/PLAN/Papers/Biten_2022_LaTr.pdf) | LaTr: Layout-Aware Transformer for Scene-Text VQA | Ali Furkan Biten et al. (2022) | CVPR 2022 | [`2112.14886`](https://arxiv.org/abs/2112.14886) | **Spatial Bridge**: Kết nối biểu diễn bố cục không gian (layout) giữa tài liệu và cảnh tự nhiên |
| **5** | [`Kim_2022_Donut.pdf`](file:///c:/Users/tungb/OneDrive%20-%20ptit.edu.vn/PTIT_BuuChinhVienThong/SOTSUGYOU%20pending/PLAN/Papers/Kim_2022_Donut.pdf) | OCR-free Document Understanding Transformer (Donut) | Geewook Kim et al. (2022) | ECCV 2022 | [`2111.15664`](https://arxiv.org/abs/2111.15664) | **OCR-free Baseline**: Kiến trúc End-to-End VLM trực tiếp từ pixel không qua OCR ngoài |
| **6** | [`Beyer_2024_PaliGemma.pdf`](file:///c:/Users/tungb/OneDrive%20-%20ptit.edu.vn/PTIT_BuuChinhVienThong/SOTSUGYOU%20pending/PLAN/Papers/Beyer_2024_PaliGemma.pdf) | PaliGemma: A versatile 3B VLM for transfer | Lucas Beyer et al. (2024) | Google Research | [`2407.07726`](https://arxiv.org/abs/2407.07726) | **Modern VLM Backbone**: Mô hình nền tảng 3B đa năng cho Fine-tuning & PEFT (LoRA) |

---

## 2. Chi Tiết Từng Bài Báo & Ý Nghĩa Trích Dẫn

### 1. Towards VQA Models That Can Read (TextVQA)
* **Tệp:** [`Singh_2019_TextVQA.pdf`](file:///c:/Users/tungb/OneDrive%20-%20ptit.edu.vn/PTIT_BuuChinhVienThong/SOTSUGYOU%20pending/PLAN/Papers/Singh_2019_TextVQA.pdf)
* **Tác giả:** Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, Marcus Rohrbach (Facebook AI Research)
* **arXiv / DOI:** [arXiv:1904.08920](https://arxiv.org/abs/1904.08920) | CVPR 2019
* **Nội dung chính:**
  - Giới thiệu bài toán hỏi đáp trên ảnh đòi hỏi đọc chữ trong không gian 3D tự nhiên (biển hiệu, số áo, bao bì sản phẩm, đồng hồ,...).
  - Cung cấp tập dữ liệu 45,336 câu hỏi trên 28,408 ảnh từ OpenImages kèm 10 câu trả lời từ người gán nhãn cho mỗi câu hỏi.
  - Chuẩn hóa độ đo **VQA Accuracy**:
    $$\text{Acc}(\text{pred}) = \min\left(\frac{\text{số annotators đồng thuận}}{3}, 1\right)$$
* **Vị trí trích dẫn:** Chương 1 (Đặt vấn đề), Chương 2 (Tổng quan bài toán TextVQA), Chương 4 (Thực nghiệm & Độ đo).

---

### 2. DocVQA: A Dataset for VQA on Document Images
* **Tệp:** [`Mathew_2021_DocVQA.pdf`](file:///c:/Users/tungb/OneDrive%20-%20ptit.edu.vn/PTIT_BuuChinhVienThong/SOTSUGYOU%20pending/PLAN/Papers/Mathew_2021_DocVQA.pdf)
* **Tác giả:** Minesh Mathew, Dimosthenis Karatzas, C. V. Jawahar (CVC Barcelona & IIIT Hyderabad)
* **arXiv / DOI:** [arXiv:2007.07394](https://arxiv.org/abs/2007.07394) | WACV 2021
* **Nội dung chính:**
  - Định hình bài toán VQA trên tài liệu quét phẳng (hóa đơn, thư từ, ghi chú, biểu mẫu UCSF Industry Documents Library).
  - Quy mô 50,000 câu hỏi trên 12,767 ảnh tài liệu có mật độ chữ cao, cấu trúc bảng biểu, key-value.
  - Đề xuất thước đo **ANLS (Average Normalized Levenshtein Similarity)** phạt mềm theo khoảng cách chỉnh sửa chuỗi (Edit Distance).
* **Vị trí trích dẫn:** Chương 1, Chương 2 (Tổng quan bài toán Document VQA), Chương 3 (Source Domain & Đánh giá).

---

### 3. Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA (M4C)
* **Tệp:** [`Hu_2020_M4C.pdf`](file:///c:/Users/tungb/OneDrive%20-%20ptit.edu.vn/PTIT_BuuChinhVienThong/SOTSUGYOU%20pending/PLAN/Papers/Hu_2020_M4C.pdf)
* **Tác giả:** Ronghang Hu, Amanpreet Singh, Trevor Darrell, Marcus Rohrbach (UC Berkeley & FAIR)
* **arXiv / DOI:** [arXiv:1911.06258](https://arxiv.org/abs/1911.06258) | CVPR 2020
* **Nội dung chính:**
  - Kiến trúc kinh điển kết hợp giữa Vision tokens, Question tokens và OCR tokens qua một Transformer chung (Multimodal Transformer).
  - Sử dụng cơ chế **Dynamic Pointer Network (Copy Mechanism)**: Mô hình tự quyết định sinh từ mới từ từ điển chung hoặc chọn trích xuất một từ OCR có sẵn trong ảnh.
  - Minh chứng rõ nét cho góc nhìn **Candidate Re-ranking (IR/RecSys)** khi xem tập token OCR là candidate pool.
* **Vị trí trích dẫn:** Chương 2 (Các kiến trúc VLM truyền thống), Chương 3 (Mô hình hóa bài toán Candidate Selection), Chương 4 (Baseline so sánh).

---

### 4. LaTr: Layout-Aware Transformer for Scene-Text VQA
* **Tệp:** [`Biten_2022_LaTr.pdf`](file:///c:/Users/tungb/OneDrive%20-%20ptit.edu.vn/PTIT_BuuChinhVienThong/SOTSUGYOU%20pending/PLAN/Papers/Biten_2022_LaTr.pdf)
* **Tác giả:** Ali Furkan Biten, Ron Litman, Yusheng Xie, Srikar Appalaraju, R. Manmatha (Amazon AWS AI)
* **arXiv / DOI:** [arXiv:2112.14886](https://arxiv.org/abs/2112.14886) | CVPR 2022
* **Nội dung chính:**
  - Tận dụng thông tin hình học bố cục (Spatial Layout / Bounding Boxes) để cải thiện khả năng đọc hiểu chữ trong ảnh cảnh tự nhiên.
  - Cầu nối học thuật trực tiếp giữa Document Layout (2D grid/planar) và Scene Text (3D perspective layout).
* **Vị trí trích dẫn:** Chương 2 & Chương 3 (Cơ chế căn chỉnh hình học và biểu diễn không gian Cross-Domain).

---

### 5. OCR-free Document Understanding Transformer (Donut)
* **Tệp:** [`Kim_2022_Donut.pdf`](file:///c:/Users/tungb/OneDrive%20-%20ptit.edu.vn/PTIT_BuuChinhVienThong/SOTSUGYOU%20pending/PLAN/Papers/Kim_2022_Donut.pdf)
* **Tác giả:** Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park et al. (NAVER Clova)
* **arXiv / DOI:** [arXiv:2111.15664](https://arxiv.org/abs/2111.15664) | ECCV 2022
* **Nội dung chính:**
  - Tiên phong trào lưu **OCR-free**: Nhận ảnh trực tiếp bằng Swin Transformer Encoder và sinh văn bản qua mBART Decoder.
  - Loại bỏ hoàn toàn sự phụ thuộc và chi phí tính toán của OCR Engine bên ngoài, hạn chế lỗi tích tụ (cascading errors) từ OCR pipeline.
* **Vị trí trích dẫn:** Chương 2 (Trường phái OCR-free VLM), Chương 4 (Thực nghiệm so sánh giữa OCR-based vs OCR-free).

---

### 6. PaliGemma: A Versatile 3B VLM for Transfer
* **Tệp:** [`Beyer_2024_PaliGemma.pdf`](file:///c:/Users/tungb/OneDrive%20-%20ptit.edu.vn/PTIT_BuuChinhVienThong/SOTSUGYOU%20pending/PLAN/Papers/Beyer_2024_PaliGemma.pdf)
* **Tác giả:** Lucas Beyer, Andreas Steiner, André Susano Pinto et al. (Google Research)
* **arXiv / DOI:** [arXiv:2407.07726](https://arxiv.org/abs/2407.07726)
* **Nội dung chính:**
  - Mô hình nền tảng Vision-Language hiện đại gồm SigLIP Vision Encoder + Gemma Language Model (~3 tỷ tham số).
  - Được huấn luyện chuyên biệt cho bài toán **Downstream Transfer Learning** trên đa dạng nhiệm vụ: DocVQA, TextVQA, ChartQA, Captioning, Detection.
  - Rất phù hợp với điều kiện GPU học thuật vừa và nhỏ khi áp dụng các kỹ thuật PEFT (LoRA/QLoRA).
* **Vị trí trích dẫn:** Chương 2, Chương 3 (Lựa chọn Backbone VLM hiện đại cho CrossVQA), Chương 4 (Kịch bản thực nghiệm).

---

## 3. Danh Mục BibTeX Trích Dẫn Chuẩn (Bibliography)

```bibtex
@inproceedings{singh2019towards,
  title={Towards VQA Models That Can Read},
  author={Singh, Amanpreet and Natarajan, Vivek and Shah, Meet and Jiang, Yu and Chen, Xinlei and Batra, Dhruv and Parikh, Devi and Rohrbach, Marcus},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={8317--8326},
  year={2019}
}

@inproceedings{mathew2021docvqa,
  title={DocVQA: A Dataset for VQA on Document Images},
  author={Mathew, Minesh and Karatzas, Dimosthenis and Jawahar, CV},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={2200--2209},
  year={2021}
}

@inproceedings{hu2020iterative,
  title={Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA},
  author={Hu, Ronghang and Singh, Amanpreet and Darrell, Trevor and Rohrbach, Marcus},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={9992--10002},
  year={2020}
}

@inproceedings{biten2022latr,
  title={LaTr: Layout-Aware Transformer for Scene-Text VQA},
  author={Biten, Ali Furkan and Litman, Ron and Xie, Yusheng and Appalaraju, Srikar and Manmatha, R},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={16514--16524},
  year={2022}
}

@inproceedings{kim2022ocr,
  title={Ocr-free document understanding transformer},
  author={Kim, Geewook and Hong, Teakgyu and Yim, Moonbin and Nam, JeongYeon and Park, Jinyoung and Yim, Jinyeong and Hwang, Wonseok and Yun, Sangdoo and Han, Dongyoon and Park, Seunghyun},
  booktitle={European Conference on Computer Vision (ECCV)},
  pages={498--517},
  year={2022}
}

@article{beyer2024paligemma,
  title={PaliGemma: A versatile 3B VLM for transfer},
  author={Beyer, Lucas and Steiner, Andreas and Pinto, Andr{\'e} Susano and Kolesnikov, Alexander and Wang, Xiao and Salz, Daniel and Neumann, Maxim and Alabdulmohsin, Ibrahim and others},
  journal={arXiv preprint arXiv:2407.07726},
  year={2024}
}
```
