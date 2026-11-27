# Danh Mục Bài Báo Nghiên Cứu (Downloaded Papers for CrossVQA)

> **Thư mục lưu trữ:** [`PLAN/Papers`](Papers)  
> **Đề tài:** CrossVQA: Nghiên cứu chuyển giao tri thức liên miền từ ảnh tài liệu sang ảnh cảnh tự nhiên cho mô hình Thị giác–Ngôn ngữ  
> **Bộ dữ liệu chính (Canonical):** [`gamusa/textvqa-canonical`](https://huggingface.co/datasets/gamusa/textvqa-canonical) & [`gamusa/docvqa-canonical`](https://huggingface.co/datasets/gamusa/docvqa-canonical) *(Kế thừa từ nguyên bản TextVQA v0.5.1 & SP-DocVQA Task 1)*  
> **Thời điểm cập nhật:** 26/11/2026  

---

## 1. Bảng Tổng Hợp Các Bài Báo Đã Tải

| STT | Tên Tệp PDF | Tiêu Đề Bài Báo | Tác Giả & Năm | Nguồn / Hội Nghị | arXiv ID | Vai Trò trong Luận Văn |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | [`Singh_2019_TextVQA.pdf`](Papers/Singh_2019_TextVQA.pdf) | Towards VQA Models That Can Read | Amanpreet Singh et al. (2019) | CVPR 2019 | [`1904.08920`](https://arxiv.org/abs/1904.08920) | **Target Domain**: Bộ dữ liệu & chuẩn đánh giá TextVQA, công thức VQA Accuracy |
| **2** | [`Mathew_2021_DocVQA.pdf`](Papers/Mathew_2021_DocVQA.pdf) | DocVQA: A Dataset for VQA on Document Images | Minesh Mathew et al. (2021) | WACV 2021 | [`2007.00398`](https://arxiv.org/abs/2007.00398) | **Source Domain**: Bộ dữ liệu DocVQA văn bản phẳng 2D, công thức đo ANLS |
| **3** | [`Hu_2020_M4C.pdf`](Papers/Hu_2020_M4C.pdf) | Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA (M4C) | Ronghang Hu et al. (2020) | CVPR 2020 | [`1911.06258`](https://arxiv.org/abs/1911.06258) | **OCR-based Baseline**: Kiến trúc Pointer Network xếp hạng / sao chép từ OCR |
| **4** | [`Biten_2022_LaTr.pdf`](Papers/Biten_2022_LaTr.pdf) | LaTr: Layout-Aware Transformer for Scene-Text VQA | Ali Furkan Biten et al. (2022) | CVPR 2022 | [`2112.14886`](https://arxiv.org/abs/2112.14886) | **Spatial Bridge**: Kết nối biểu diễn bố cục không gian (layout) giữa tài liệu và cảnh tự nhiên |
| **5** | [`Kim_2022_Donut.pdf`](Papers/Kim_2022_Donut.pdf) | OCR-free Document Understanding Transformer (Donut) | Geewook Kim et al. (2022) | ECCV 2022 | [`2111.15664`](https://arxiv.org/abs/2111.15664) | **OCR-free Baseline**: Kiến trúc End-to-End VLM trực tiếp từ pixel không qua OCR ngoài |
| **6** | [`Beyer_2024_PaliGemma.pdf`](Papers/Beyer_2024_PaliGemma.pdf) | PaliGemma: A versatile 3B VLM for transfer | Lucas Beyer et al. (2024) | Google Research | [`2407.07726`](https://arxiv.org/abs/2407.07726) | **Modern VLM Backbone**: Mô hình nền tảng 3B đa năng cho Fine-tuning & PEFT (LoRA) |

---

## 2. Chi Tiết Từng Bài Báo & Ý Nghĩa Trích Dẫn

### 1. Towards VQA Models That Can Read (TextVQA)
* **Tệp:** [`Singh_2019_TextVQA.pdf`](Papers/Singh_2019_TextVQA.pdf)
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
* **Tệp:** [`Mathew_2021_DocVQA.pdf`](Papers/Mathew_2021_DocVQA.pdf)
* **Tác giả:** Minesh Mathew, Dimosthenis Karatzas, C. V. Jawahar (CVC Barcelona & IIIT Hyderabad)
* **arXiv / DOI:** [arXiv:2007.00398](https://arxiv.org/abs/2007.00398) | WACV 2021
* **Nội dung chính:**
  - Định hình bài toán VQA trên tài liệu quét phẳng (hóa đơn, thư từ, ghi chú, biểu mẫu UCSF Industry Documents Library).
  - Quy mô 50,000 câu hỏi trên 12,767 ảnh tài liệu có mật độ chữ cao, cấu trúc bảng biểu, key-value.
  - Đề xuất thước đo **ANLS (Average Normalized Levenshtein Similarity)** phạt mềm theo khoảng cách chỉnh sửa chuỗi (Edit Distance).
* **Vị trí trích dẫn:** Chương 1, Chương 2 (Tổng quan bài toán Document VQA), Chương 3 (Source Domain & Đánh giá).

---

### 3. Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA (M4C)
* **Tệp:** [`Hu_2020_M4C.pdf`](Papers/Hu_2020_M4C.pdf)
* **Tác giả:** Ronghang Hu, Amanpreet Singh, Trevor Darrell, Marcus Rohrbach (UC Berkeley & FAIR)
* **arXiv / DOI:** [arXiv:1911.06258](https://arxiv.org/abs/1911.06258) | CVPR 2020
* **Nội dung chính:**
  - Kiến trúc kinh điển kết hợp giữa Vision tokens, Question tokens và OCR tokens qua một Transformer chung (Multimodal Transformer).
  - Sử dụng cơ chế **Dynamic Pointer Network (Copy Mechanism)**: Mô hình tự quyết định sinh từ mới từ từ điển chung hoặc chọn trích xuất một từ OCR có sẵn trong ảnh.
  - Minh chứng rõ nét cho góc nhìn **Candidate Re-ranking (IR/RecSys)** khi xem tập token OCR là candidate pool.
* **Vị trí trích dẫn:** Chương 2 (Các kiến trúc VLM truyền thống), Chương 3 (Mô hình hóa bài toán Candidate Selection), Chương 4 (Baseline so sánh).

---

### 4. LaTr: Layout-Aware Transformer for Scene-Text VQA
* **Tệp:** [`Biten_2022_LaTr.pdf`](Papers/Biten_2022_LaTr.pdf)
* **Tác giả:** Ali Furkan Biten, Ron Litman, Yusheng Xie, Srikar Appalaraju, R. Manmatha (Amazon AWS AI)
* **arXiv / DOI:** [arXiv:2112.14886](https://arxiv.org/abs/2112.14886) | CVPR 2022
* **Nội dung chính:**
  - Tận dụng thông tin hình học bố cục (Spatial Layout / Bounding Boxes) để cải thiện khả năng đọc hiểu chữ trong ảnh cảnh tự nhiên.
  - Cầu nối học thuật trực tiếp giữa Document Layout (2D grid/planar) và Scene Text (3D perspective layout).
* **Vị trí trích dẫn:** Chương 2 & Chương 3 (Cơ chế căn chỉnh hình học và biểu diễn không gian Cross-Domain).

---

### 5. OCR-free Document Understanding Transformer (Donut)
* **Tệp:** [`Kim_2022_Donut.pdf`](Papers/Kim_2022_Donut.pdf)
* **Tác giả:** Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park et al. (NAVER Clova)
* **arXiv / DOI:** [arXiv:2111.15664](https://arxiv.org/abs/2111.15664) | ECCV 2022
* **Nội dung chính:**
  - Tiên phong trào lưu **OCR-free**: Nhận ảnh trực tiếp bằng Swin Transformer Encoder và sinh văn bản qua mBART Decoder.
  - Loại bỏ hoàn toàn sự phụ thuộc và chi phí tính toán của OCR Engine bên ngoài, hạn chế lỗi tích tụ (cascading errors) từ OCR pipeline.
* **Vị trí trích dẫn:** Chương 2 (Trường phái OCR-free VLM), Chương 4 (Thực nghiệm so sánh giữa OCR-based vs OCR-free).

---

### 6. PaliGemma: A Versatile 3B VLM for Transfer
* **Tệp:** [`Beyer_2024_PaliGemma.pdf`](Papers/Beyer_2024_PaliGemma.pdf)
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

---

## 4. Bổ Sung: Bài Báo Về Chuyển Giao Liên Miền DocVQA ↔ TextVQA (Cross-Domain Transfer)

> **Thời điểm cập nhật:** 04/10/2026

| STT | Tên Tệp PDF | Tiêu Đề Bài Báo | Tác Giả & Năm | Nguồn / Hội Nghị | arXiv ID | Vai Trò trong Luận Văn |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **7** | [`Tang_2023_UDOP.pdf`](Papers/Tang_2023_UDOP.pdf) | Unifying Vision, Text, and Layout for Universal Document Processing (UDOP) | Zineng Tang et al. (2023) | CVPR 2023 | [`2212.02623`](https://arxiv.org/abs/2212.02623) | Mô hình thống nhất Vision–Text–Layout, pretrain đa tác vụ trên tài liệu; chuyển giao tốt sang nhiều benchmark (gồm DocVQA) |
| **8** | [`Lee_2023_Pix2Struct.pdf`](Papers/Lee_2023_Pix2Struct.pdf) | Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding | Kenton Lee et al. (2023) | ICML 2023 | [`2210.03347`](https://arxiv.org/abs/2210.03347) | Pretrain trên screenshot web, một mô hình dùng chung cho tài liệu, UI, ảnh tự nhiên (DocVQA, TextVQA-like); minh chứng chuyển giao đa miền |
| **9** | [`Huang_2022_LayoutLMv3.pdf`](Papers/Huang_2022_LayoutLMv3.pdf) | LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking | Yupan Huang et al. (2022) | ACM MM 2022 | [`2204.08387`](https://arxiv.org/abs/2204.08387) | Baseline Document AI (OCR + layout) cho Source Domain DocVQA |
| **10** | [`Appalaraju_2021_DocFormer.pdf`](Papers/Appalaraju_2021_DocFormer.pdf) | DocFormer: End-to-End Transformer for Document Understanding | Srikar Appalaraju et al. (2021) | ICCV 2021 | [`2106.11539`](https://arxiv.org/abs/2106.11539) | Transformer đa phương thức (text, vision, spatial) cho tài liệu; nền tảng của LaTr |
| **11** | [`Yang_2021_TAP.pdf`](Papers/Yang_2021_TAP.pdf) | TAP: Text-Aware Pre-training for Text-VQA and Text-Caption | Zhengyuan Yang et al. (2021) | CVPR 2021 | [`2012.04638`](https://arxiv.org/abs/2012.04638) | Pretrain nhận thức văn bản cho scene-text; baseline TextVQA, gợi ý pretraining miền trung gian |
| **12** | [`Kil_2023_PreSTU.pdf`](Papers/Kil_2023_PreSTU.pdf) | PreSTU: Pre-Training for Scene-Text Understanding | Jihyung Kil et al. (2023) | ICCV 2023 | [`2209.05534`](https://arxiv.org/abs/2209.05534) | Pretrain đọc chữ (OCR-like) cho scene-text VQA, chuyển giao sang TextVQA/ST-VQA |
| **13** | [`Ganz_2023_SeeAndRead.pdf`](Papers/Ganz_2023_SeeAndRead.pdf) | Towards Models that Can See and Read | Roy Ganz et al. (2023) | ICCV 2023 | [`2301.07389`](https://arxiv.org/abs/2301.07389) | Hợp nhất VQA và Image Captioning có đọc chữ (Scene-Text); cho thấy lợi ích chuyển giao đa tác vụ |
| **14** | [`Ye_2023_UReader.pdf`](Papers/Ye_2023_UReader.pdf) | UReader: Universal OCR-free Visually-situated Language Understanding with Multimodal LLM | Jiabo Ye et al. (2023) | EMNLP 2023 Findings | [`2310.05126`](https://arxiv.org/abs/2310.05126) | MLLM OCR-free tinh chỉnh chung trên tài liệu, biểu đồ, ảnh tự nhiên; đánh giá DocVQA & TextVQA |

---

## 5. Bổ Sung: Bài Báo Về Quy Ước Tọa Độ Không Gian & Hộp Bao OCR (Spatial Coordinate & Grounding Conventions)

> **Thời điểm cập nhật:** 06/10/2026

| STT | Tên Tệp PDF | Tiêu Đề Bài Báo | Tác Giả & Năm | Nguồn / Hội Nghị | arXiv ID | Vai Trò trong Luận Văn |
| :---: | :--- | :--- | :--- | :--- | :---: | :--- |
| **15** | [`Xu_2020_LayoutLM.pdf`](Papers/Xu_2020_LayoutLM.pdf) | LayoutLM: Pre-training of Text and Layout for Document Image Understanding | Yang Xu et al. (2020) | KDD 2020 | [`1912.13318`](https://arxiv.org/abs/1912.13318) | **Spatial Embedding Baseline**: Tiên phong chuẩn hóa tọa độ hộp bao OCR về thang rời rạc $[0, 1000]$ cho 2D spatial embeddings trong Transformer |
| **16** | [`Chen_2022_Pix2seq.pdf`](Papers/Chen_2022_Pix2seq.pdf) | Pix2seq: A Language Modeling Framework for Object Detection | Ting Chen et al. (2022) | ICLR 2022 | [`2109.10852`](https://arxiv.org/abs/2109.10852) | **Spatial Sequence Modeling**: Chuẩn hóa biểu diễn hộp bao dạng chuỗi $[ymin, xmin, ymax, xmax]$ lượng tử hóa 1000 bins cho ngôn ngữ sinh tự hồi quy |
| **17** | [`Chen_2022_PaLI.pdf`](Papers/Chen_2022_PaLI.pdf) | PaLI: A Jointly-Scaled Multilingual Language-Image Model | Xi Chen et al. (2023) | ICLR 2023 | [`2209.06794`](https://arxiv.org/abs/2209.06794) | **Spatial Location Tokens**: Chuẩn hóa cơ chế gán token vị trí chuyên biệt $\langle\text{loc}Y_{min}\rangle\langle\text{loc}X_{min}\rangle\langle\text{loc}Y_{max}\rangle\langle\text{loc}X_{max}\rangle$ làm nền tảng cho PaLI-X và PaliGemma |

### Chi Tiết Từng Bài Báo & Ý Nghĩa Trích Dẫn

#### 15. LayoutLM: Pre-training of Text and Layout for Document Image Understanding
* **Tệp:** [`Xu_2020_LayoutLM.pdf`](Papers/Xu_2020_LayoutLM.pdf)
* **Tác giả:** Yang Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou (Microsoft Research Asia)
* **arXiv / DOI:** [arXiv:1912.13318](https://arxiv.org/abs/1912.13318) | KDD 2020
* **Nội dung chính:**
  - Mô hình nền tảng đầu tiên kết hợp thông tin vị trí không gian 2D (2D layout spatial embeddings) cùng ngữ nghĩa văn bản trong kiến trúc Transformer cho Document AI.
  - Chuẩn hóa toàn bộ tọa độ bounding box OCR từ pixel thực về không gian số nguyên rời rạc $[0, 1000]$:
    $$x_0 = \left\lfloor \frac{x_{\text{min}}}{W} \times 1000 \right\rfloor, \quad y_0 = \left\lfloor \frac{y_{\text{min}}}{H} \times 1000 \right\rfloor, \quad x_1 = \left\lfloor \frac{x_{\text{max}}}{W} \times 1000 \right\rfloor, \quad y_1 = \left\lfloor \frac{y_{\text{max}}}{H} \times 1000 \right\rfloor$$
* **Vị trí trích dẫn:** Chương 2 (Biểu diễn không gian & Layout-aware VLM), Chương 3 (Quy chuẩn thiết kế Canonical Schema $[0, 1000]$ cho hộp bao OCR).

---

#### 16. Pix2seq: A Language Modeling Framework for Object Detection
* **Tệp:** [`Chen_2022_Pix2seq.pdf`](Papers/Chen_2022_Pix2seq.pdf)
* **Tác giả:** Ting Chen, Saurabh Saxena, Lala Li, David J. Fleet, Geoffrey Hinton (Google Research, Brain Team)
* **arXiv / DOI:** [arXiv:2109.10852](https://arxiv.org/abs/2109.10852) | ICLR 2022
* **Nội dung chính:**
  - Định hình bài toán phát hiện vật thể và định vị hình học dưới lăng kính bài toán mô hình hóa ngôn ngữ (Language Modeling).
  - Chuẩn hóa thứ tự biểu diễn tọa độ 4 chiều: $[ymin, xmin, ymax, xmax]$ được rời rạc hóa thành 1,000 bins số nguyên.
  - Cung cấp cơ sở lý thuyết cho việc xử lý hộp bao như các chuỗi token ngôn ngữ tự nhiên.
* **Vị trí trích dẫn:** Chương 2 (Language Modeling for Vision Tasks), Chương 3 (Quy ước thứ tự tọa độ $[ymin, xmin, ymax, xmax]$ trong `canonical_schema.py`).

---

#### 17. PaLI: A Jointly-Scaled Multilingual Language-Image Model
* **Tệp:** [`Chen_2022_PaLI.pdf`](Papers/Chen_2022_PaLI.pdf)
* **Tác giả:** Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski et al. (Google Research)
* **arXiv / DOI:** [arXiv:2209.06794](https://arxiv.org/abs/2209.06794) | ICLR 2023
* **Nội dung chính:**
  - Mở rộng khả năng của mô hình đa phương thức ngôn ngữ–thị giác quy mô lớn (lên tới 17B tham số) hỗ trợ hơn 100 ngôn ngữ.
  - Thiết lập chuẩn tokenization vị trí: biểu diễn hộp bao qua 1000 spatial location tokens đặc biệt `[<loc0000>, ..., <loc1000>]` xếp theo thứ tự $[ymin, xmin, ymax, xmax]$, kế thừa trực tiếp trong kiến trúc PaliGemma.
* **Vị trí trích dẫn:** Chương 2 (Kiến trúc VLM thế hệ mới), Chương 3 (Thiết kế token vị trí cho Vision-Language Grounding).

---

### Danh Mục BibTeX Trích Dẫn Bổ Sung

```bibtex
@inproceedings{xu2020layoutlm,
  title     = {LayoutLM: Pre-training of Text and Layout for Document Image Understanding},
  author    = {Xu, Yang and Li, Minghao and Cui, Lei and Huang, Shaohan and Wei, Furu and Zhou, Ming},
  booktitle = {Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining (KDD)},
  pages     = {1192--1200},
  year      = {2020},
  doi       = {10.1145/3394486.3403172}
}

@inproceedings{chen2022pix2seq,
  title     = {Pix2seq: A Language Modeling Framework for Object Detection},
  author    = {Chen, Ting and Saxena, Saurabh and Li, Lala and Fleet, David J and Hinton, Geoffrey},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  url       = {https://openreview.net/forum?id=nNO6SNBs14Z}
}

@inproceedings{chen2023pali,
  title     = {PaLI: A Jointly-Scaled Multilingual Language-Image Model},
  author    = {Chen, Xi and Wang, Xiao and Changpinyo, Soravit and Piergiovanni, AJ and Padlewski, Piotr and Salz, Daniel and Goodman, Sebastian and Grycner, Adam and Mustafa, Basil and Beyer, Lucas and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023},
  url       = {https://openreview.net/forum?id=adSSmdd3tl}
}
```

---

## 6. Bổ Sung: Bài Báo Về Vấn Đề Phân Tách Dữ Liệu & Rò Rỉ / Ô Nhiễm Benchmark (Dataset Split & Benchmark Leakage)

> **Thư mục lưu trữ con:** [`PLAN/Papers/Dataset_Split_Issue`](Papers/Dataset_Split_Issue)  
> **Thời điểm cập nhật:** 10/10/2026

| STT | Tên Tệp PDF | Tiêu Đề Bài Báo | Tác Giả & Năm | Nguồn / Hội Nghị | arXiv ID | Vai Trò trong Luận Văn |
| :---: | :--- | :--- | :--- | :--- | :---: | :--- |
| **18** | [`Kapoor_2023_Data_Leakage.pdf`](Papers/Dataset_Split_Issue/Kapoor_2023_Data_Leakage.pdf) | Leakage and the Reproducibility Crisis in Machine Learning-Based Science | Sayash Kapoor & Arvind Narayanan (2023) | Patterns (Cell Press) 2023 | [`2207.07048`](https://arxiv.org/abs/2207.07048) | **Lý thuyết Rò rỉ Dữ liệu & Phân tách Nhóm**: Khung phân loại 8 loại data leakage; cơ sở lý luận cho phân chia nhóm (Group-based split theo `image_id` và `ucsf_document_id`) để chống rò rỉ |
| **19** | [`Wang_2024_Both_Text_and_Images_Leaked.pdf`](Papers/Dataset_Split_Issue/Wang_2024_Both_Text_and_Images_Leaked.pdf) | Both Text and Images Leaked! A Systematic Analysis of Data Contamination in Multimodal LLM | MM-Detect Team (2024) | EMNLP 2025 Findings | [`2411.03823`](https://arxiv.org/abs/2411.03823) | **Ô nhiễm Dữ liệu Đa phương thức**: Phân tích rò rỉ unimodal và cross-modal trên benchmark VQA (trong đó có TextVQA); cơ sở kiểm toán nhiễm dữ liệu tiền huấn luyện ở Chương 4 |
| **20** | [`Li_2024_LMMs_Eval.pdf`](Papers/Dataset_Split_Issue/Li_2024_LMMs_Eval.pdf) | LMMs-Eval: Accelerating the Development of Large Multimodal Models | Bo Li et al. (2024) | arXiv 2024 | [`2407.12772`](https://arxiv.org/abs/2407.12772) | **Chuẩn hóa Đánh giá Benchmark**: Framework đánh giá chuẩn cho VLM mã nguồn mở; quy định TextVQA-val (5.000) và DocVQA-val (5.349) là tập benchmark chuẩn, minh chứng thực tiễn đánh giá |
| **21** | [`Agrawal_2018_VQA_CP.pdf`](Papers/Dataset_Split_Issue/Agrawal_2018_VQA_CP.pdf) | Don't Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering (VQA-CP) | Aishwarya Agrawal et al. (2018) | CVPR 2018 | [`1712.00377`](https://arxiv.org/abs/1712.00377) | **Thiên lệch Phân tách Dữ liệu VQA**: Bài báo kinh điển chứng minh phân tách ngẫu nhiên gây rò rỉ tiên nghiệm câu hỏi–đáp án; luận giải sự cần thiết của thiết kế split có kiểm soát |

### Chi Tiết Từng Bài Báo & Ý Nghĩa Trích Dẫn

#### 18. Leakage and the Reproducibility Crisis in Machine Learning-Based Science
* **Tệp:** [`Kapoor_2023_Data_Leakage.pdf`](Papers/Dataset_Split_Issue/Kapoor_2023_Data_Leakage.pdf)
* **Tác giả:** Sayash Kapoor, Arvind Narayanan (Center for Information Technology Policy, Princeton University)
* **arXiv / DOI:** [arXiv:2207.07048](https://arxiv.org/abs/2207.07048) | *Patterns*, Cell Press (2023)
* **Nội dung chính:**
  - Khảo sát hệ thống về khủng hoảng tái lập do rò rỉ dữ liệu (data leakage) trên 17 lĩnh vực khoa học sử dụng machine learning.
  - Phân loại chi tiết 8 dạng rò rỉ dữ liệu, trong đó nhấn mạnh rò rỉ do phụ thuộc giữa các mẫu cùng nhóm (lack of group independence across splits). Khi một thực thể (ví dụ: một bức ảnh hoặc một tài liệu đa trang) có nhiều câu hỏi liên kết mà bị chia ngẫu nhiên vào cả train và test/dev, mô hình sẽ học thuộc đặc trưng ảnh thay vì học năng lực suy luận.
  - Đề xuất mô hình thông tin chuẩn (model info sheets) để kiểm soát rò rỉ trước khi công bố.
* **Vị trí trích dẫn trong Luận văn:**
  - **Chương 3 (Pha B - Phương pháp phân tách dữ liệu chống rò rỉ):** Cung cấp cơ sở học thuật vững chắc cho việc bắt buộc sử dụng `GroupShuffleSplit` (nhóm theo `image_id` cho TextVQA và `ucsf_document_id` cho DocVQA) thay vì chia ngẫu nhiên ngây thơ.
  - **Chương 5 (Mối đe dọa tính hợp lệ - Internal Validity):** Luận giải cách đề tài triệt tiêu rò rỉ giữa train và validation.

---

#### 19. Both Text and Images Leaked! A Systematic Analysis of Data Contamination in Multimodal LLM
* **Tệp:** [`Wang_2024_Both_Text_and_Images_Leaked.pdf`](Papers/Dataset_Split_Issue/Wang_2024_Both_Text_and_Images_Leaked.pdf)
* **Tác giả:** MM-Detect Team
* **arXiv / DOI:** [arXiv:2411.03823](https://arxiv.org/abs/2411.03823) | Findings of EMNLP 2025
* **Nội dung chính:**
  - Nghiên cứu có tính hệ thống đầu tiên về hiện tượng ô nhiễm dữ liệu (data contamination) và rò rỉ benchmark trên các mô hình Đa phương thức Lớn (MLLM).
  - Chỉ ra rằng các tập benchmark VQA phổ biến (như TextVQA) thường xuyên bị thu thập vào kho dữ liệu tiền huấn luyện web-scale.
  - Đưa ra khung phân tích **MM-Detect** phân biệt rõ rò rỉ đơn phương thức (unimodal leakage) và rò rỉ chéo phương thức (cross-modal leakage).
* **Vị trí trích dẫn trong Luận văn:**
  - **Chương 4 (Pha C - Kiểm toán rò rỉ tiền huấn luyện / Contamination Audit):** Dẫn chứng khoa học lý giải vì sao đề tài không sử dụng các mô hình đã qua tinh chỉnh VQA thương mại (như PaliGemma-FT, LLaVA-1.5 fine-tuned) mà chỉ dùng checkpoint nền tảng (Base) để đảm bảo tính trong sạch của chuyển giao tri thức liên miền.

---

#### 20. LMMs-Eval: Accelerating the Development of Large Multimodal Models
* **Tệp:** [`Li_2024_LMMs_Eval.pdf`](Papers/Dataset_Split_Issue/Li_2024_LMMs_Eval.pdf)
* **Tác giả:** Bo Li, Peiyuan Zhang, Kaichen Zhang et al.
* **arXiv / DOI:** [arXiv:2407.12772](https://arxiv.org/abs/2407.12772)
* **Nội dung chính:**
  - Bộ công cụ đánh giá chuẩn mực (standard evaluation harness) được cộng đồng VLM quốc tế (Hugging Face, LLaVA, Mistral, InternVL) áp dụng rộng rãi.
  - Chuẩn hóa quy trình đánh giá TextVQA trên tập `textvqa_val` (5.000 mẫu) và DocVQA trên tập `docvqa_val` (5.349 mẫu), minh chứng rằng trong bối cảnh các server nộp bài kiểm thử đóng hoặc bảo mật nhãn, tập validation chính thức là thước đo học thuật được công nhận rộng rãi nhất.
* **Vị trí trích dẫn trong Luận văn:**
  - **Chương 2 (Các chuẩn đánh giá VLM) & Chương 4 (Thiết lập thực nghiệm):** Trích dẫn chuẩn mực đánh giá mở, giải thích và bảo vệ việc sử dụng tập `validation` chính thức làm tập kiểm thử độc lập cho kết quả công bố trong luận văn.

---

#### 21. Don't Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering (VQA-CP)
* **Tệp:** [`Agrawal_2018_VQA_CP.pdf`](Papers/Dataset_Split_Issue/Agrawal_2018_VQA_CP.pdf)
* **Tác giả:** Aishwarya Agrawal, Dhruv Batra, Devi Parikh, Aniruddha Kembhavi (Georgia Tech, FAIR, Allen Institute for AI)
* **arXiv / DOI:** [arXiv:1712.00377](https://arxiv.org/abs/1712.00377) | CVPR 2018
* **Nội dung chính:**
  - Bài báo kinh điển phát hiện lỗ hổng nghiêm trọng của cách phân tách tập dữ liệu ngẫu nhiên (random i.i.d splits) trong VQA: mô hình chỉ học thuộc phân phối tiên nghiệm của câu hỏi (question priors) thay vì thực sự nhìn vào ảnh.
  - Đề xuất tái cấu trúc phân tách dữ liệu dưới dạng **Changing Priors (VQA-CP)** để phân phối câu trả lời ở tập train và test khác nhau, buộc mô hình phải suy luận thị giác thực chất.
* **Vị trí trích dẫn trong Luận văn:**
  - **Chương 2 (Tổng quan bài toán VQA và các thiên lệch dữ liệu):** Phân tích rủi ro của việc mô hình học vẹt câu hỏi nếu không có chiến lược phân tách dữ liệu và kiểm tra out-of-distribution (OOD) nghiêm ngặt.
  - **Chương 3 (Thiết kế thực nghiệm chuyển giao miền Doc $\rightarrow$ Text):** Chứng minh năng lực chuyển giao thực chất từ tài liệu sang cảnh tự nhiên thay vì khai thác tiên nghiệm ngôn ngữ.

---

### Danh Mục BibTeX Trích Dẫn Bổ Sung (Dataset Split & Contamination)

```bibtex
@article{kapoor2023leakage,
  title     = {Leakage and the Reproducibility Crisis in Machine Learning-Based Science},
  author    = {Kapoor, Sayash and Narayanan, Arvind},
  journal   = {Patterns},
  volume    = {4},
  number    = {9},
  pages     = {100804},
  year      = {2023},
  publisher = {Elsevier},
  doi       = {10.1016/j.patter.2023.100804}
}

@article{wang2024both,
  title     = {Both Text and Images Leaked! A Systematic Analysis of Data Contamination in Multimodal LLM},
  author    = {Wang, et al.},
  journal   = {arXiv preprint arXiv:2411.03823},
  year      = {2024}
}

@article{li2024lmms,
  title     = {LMMs-Eval: Accelerating the Development of Large Multimodal Models},
  author    = {Li, Bo and Zhang, Peiyuan and Zhang, Kaichen and others},
  journal   = {arXiv preprint arXiv:2407.12772},
  year      = {2024}
}

@inproceedings{agrawal2018dont,
  title     = {Don't Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering},
  author    = {Agrawal, Aishwarya and Batra, Dhruv and Parikh, Devi and Kembhavi, Aniruddha},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {4971--4980},
  year      = {2018}
}
```

---

## 7. Bổ Sung: Bài Báo Về Kiến Trúc Mô Hình Nền Tảng VLM & Tinh Chỉnh Lượng Tử Hóa (VLM Backbones & PEFT/Quantization)

> **Thư mục lưu trữ con:** [`PLAN/Papers/Model Info`](Papers/Model%20Info)  
> **Thời điểm cập nhật:** 26/11/2026  
> **Tham chiếu chéo các mô hình nền tảng đã có sẵn trong [`PLAN/Papers`](Papers):** STT **5** ([`Kim_2022_Donut.pdf`](Papers/Kim_2022_Donut.pdf)), STT **6** ([`Beyer_2024_PaliGemma.pdf`](Papers/Beyer_2024_PaliGemma.pdf)), STT **8** ([`Lee_2023_Pix2Struct.pdf`](Papers/Lee_2023_Pix2Struct.pdf)).

| STT | Tên Tệp PDF | Tiêu Đề Bài Báo | Tác Giả & Năm | Nguồn / Hội Nghị | arXiv ID | Vai Trò trong Luận Văn |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **22** | [`Wang_2024_Qwen2_VL.pdf`](Papers/Model%20Info/Wang_2024_Qwen2_VL.pdf) | Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution | Peng Wang et al. (2024) | arXiv 2024 | [`2409.12191`](https://arxiv.org/abs/2409.12191) | **Primary Backbone**: Kiến trúc Naive Dynamic Resolution (`Qwen2-VL-2B`) & M-RoPE xử lý chênh lệch độ phân giải $4.8\times$ giữa DocVQA và TextVQA |
| **23** | [`Dehghani_2023_NaViT.pdf`](Papers/Model%20Info/Dehghani_2023_NaViT.pdf) | Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution | Mostafa Dehghani et al. (2023) | NeurIPS 2023 | [`2307.06304`](https://arxiv.org/abs/2307.06304) | **Dynamic Vision Foundation**: Cơ chế bảo toàn tỷ lệ khung hình & nén sequence patch nền tảng cho Vision Encoder của Qwen2-VL |
| **24** | [`Marafioti_2025_SmolVLM.pdf`](Papers/Model%20Info/Marafioti_2025_SmolVLM.pdf) | SmolVLM: Redefining small and efficient multimodal models | Andrés Marafioti et al. (2025) | arXiv 2025 | [`2504.05299`](https://arxiv.org/abs/2504.05299) | **Fast Pilot Backbone**: Mô hình VLM nhỏ gọn (`500M` / `2.2B`) tối ưu Pixel Shuffle & bộ nhớ GPU cho thử nghiệm đa seed trên Colab T4 |
| **25** | [`Laurencon_2024_Idefics3.pdf`](Papers/Model%20Info/Laurencon_2024_Idefics3.pdf) | Building and better understanding vision-language models: insights and future directions | Hugo Laurençon et al. (2024) | arXiv 2024 | [`2408.12637`](https://arxiv.org/abs/2408.12637) | **Idefics3 / Image-Splitting Design**: Kiến trúc chia ô ảnh và kết nối đa phương thức nền tảng cho dòng SmolVLM |
| **26** | [`Steiner_2024_PaliGemma2.pdf`](Papers/Model%20Info/Steiner_2024_PaliGemma2.pdf) | PaliGemma 2: A Family of Versatile VLMs for Transfer | Andreas Steiner et al. (2024) | arXiv 2024 | [`2412.03555`](https://arxiv.org/abs/2412.03555) | **Transfer Scaling Reference**: Phân tích tác động của độ phân giải (224/448/896px), kích thước mô hình & tốc độ học khi transfer trên DocVQA/OCR |
| **27** | [`Zhai_2023_SigLIP.pdf`](Papers/Model%20Info/Zhai_2023_SigLIP.pdf) | Sigmoid Loss for Language Image Pre-Training | Xiaohua Zhai et al. (2023) | ICCV 2023 | [`2303.15343`](https://arxiv.org/abs/2303.15343) | **Vision Encoder Foundation**: Bộ mã hóa ảnh SigLIP dùng chung trong PaliGemma / PaliGemma 2 và SmolVLM |
| **28** | [`Hu_2022_LoRA.pdf`](Papers/Model%20Info/Hu_2022_LoRA.pdf) | LoRA: Low-Rank Adaptation of Large Language Models | Edward J. Hu et al. (2022) | ICLR 2022 | [`2106.09685`](https://arxiv.org/abs/2106.09685) | **PEFT & Dual-Adapter Basis**: Nền tảng lý thuyết thích ứng hạng thấp cho cấu trúc Adapter tách miền ($\mathcal{A}_{\text{doc}}, \mathcal{A}_{\text{scene}}$) |
| **29** | [`Dettmers_2023_QLoRA.pdf`](Papers/Model%20Info/Dettmers_2023_QLoRA.pdf) | QLoRA: Efficient Finetuning of Quantized LLMs | Tim Dettmers et al. (2023) | NeurIPS 2023 | [`2305.14314`](https://arxiv.org/abs/2305.14314) | **Memory-Saving Optimization**: Lượng tử hóa 4-bit NF4, Double Quantization & Paged Optimizers giúp huấn luyện `Qwen2-VL-2B` trên GPU 16GB |

### Chi Tiết Từng Bài Báo & Ý Nghĩa Trích Dẫn

#### 22. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
* **Tệp:** [`Wang_2024_Qwen2_VL.pdf`](Papers/Model%20Info/Wang_2024_Qwen2_VL.pdf)
* **Tác giả:** Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang et al. (Alibaba Qwen Team)
* **arXiv / DOI:** [arXiv:2409.12191](https://arxiv.org/abs/2409.12191)
* **Nội dung chính:**
  - Đề xuất kiến trúc **Naive Dynamic Resolution** tự động ánh xạ ảnh có độ phân giải và tỷ lệ khung hình bất kỳ thành số lượng token thị giác tương ứng ($2\times 2$ spatial merge với stride hiệu dụng $28\text{ px}$).
  - Đề xuất **Multimodal Rotary Position Embedding (M-RoPE)** phân rã mã hóa vị trí thành 3 thành phần (thời gian, chiều cao, chiều rộng), bảo toàn cấu trúc không gian 2D của tài liệu.
* **Vị trí trích dẫn:** Chương 2 (Kiến trúc VLM phân giải động), Chương 3 (Lựa chọn backbone `Qwen/Qwen2-VL-2B` cho CrossVQA), Chương 4 (Cấu hình `min_pixels`/`max_pixels` và thực nghiệm chính).

---

#### 23. Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution
* **Tệp:** [`Dehghani_2023_NaViT.pdf`](Papers/Model%20Info/Dehghani_2023_NaViT.pdf)
* **Tác giả:** Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek et al. (Google DeepMind)
* **arXiv / DOI:** [arXiv:2307.06304](https://arxiv.org/abs/2307.06304) | NeurIPS 2023
* **Nội dung chính:**
  - Chứng minh việc đóng gói chuỗi patch từ ảnh có độ phân giải và tỷ lệ khung hình tùy ý (sequence packing) giúp tránh biến dạng hình học so với phép resize/crop vuông truyền thống.
  - Là nền tảng lý thuyết trực tiếp cho bộ mã hóa thị giác trong Qwen2-VL và Pix2Struct.
* **Vị trí trích dẫn:** Chương 2 (Biểu diễn thị giác không biến dạng tỷ lệ), Chương 3 (Xử lý chênh lệch hình học giữa ảnh tài liệu dọc và ảnh cảnh tự nhiên).

---

#### 24. SmolVLM: Redefining Small and Efficient Multimodal Models
* **Tệp:** [`Marafioti_2025_SmolVLM.pdf`](Papers/Model%20Info/Marafioti_2025_SmolVLM.pdf)
* **Tác giả:** Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan et al. (Hugging Face)
* **arXiv / DOI:** [arXiv:2504.05299](https://arxiv.org/abs/2504.05299)
* **Nội dung chính:**
  - Thiết kế họ VLM nhỏ gọn (`256M`, `500M`, `2.2B`) tối ưu hóa tỷ lệ nén token thị giác (Pixel Shuffle) và chi phí bộ nhớ GPU.
  - Phù hợp làm mô hình thử nghiệm nhanh (fast pilot) và kiểm chứng đa seed với chi phí VRAM thấp.
* **Vị trí trích dẫn:** Chương 3 (Phân tích đánh đổi chi phí–hiệu năng của các backbone nhỏ gọn), Chương 4 (Thử nghiệm so sánh & kiểm tra độ ổn định đa seed).

---

#### 25. Building and Better Understanding Vision-Language Models: Insights and Future Directions
* **Tệp:** [`Laurencon_2024_Idefics3.pdf`](Papers/Model%20Info/Laurencon_2024_Idefics3.pdf)
* **Tác giả:** Hugo Laurençon, Andrés Marafioti, Victor Sanh, Léo Tronchon (Hugging Face)
* **arXiv / DOI:** [arXiv:2408.12637](https://arxiv.org/abs/2408.12637)
* **Nội dung chính:**
  - Giới thiệu kiến trúc **Idefics3** với chiến lược chia ô ảnh độ phân giải cao (image splitting) kết hợp Pixel Shuffle để cải thiện khả năng đọc OCR và tài liệu mà không làm bùng nổ số token.
* **Vị trí trích dẫn:** Chương 2 (Cơ chế image tiling & connector nén token thị giác), Chương 3 (Đặc tả `Idefics3ForConditionalGeneration` cho SmolVLM).

---

#### 26. PaliGemma 2: A Family of Versatile VLMs for Transfer
* **Tệp:** [`Steiner_2024_PaliGemma2.pdf`](Papers/Model%20Info/Steiner_2024_PaliGemma2.pdf)
* **Tác giả:** Andreas Steiner, André Susano Pinto, Michael Tschannen, Daniel Keysers et al. (Google DeepMind)
* **arXiv / DOI:** [arXiv:2412.03555](https://arxiv.org/abs/2412.03555)
* **Nội dung chính:**
  - Khảo sát có hệ thống sự tương tác giữa kích thước mô hình (2B–27B), độ phân giải đầu vào (`224px`, `448px`, `896px`) và tốc độ học khi chuyển giao tri thức sang các tác vụ đọc chữ/tài liệu.
* **Vị trí trích dẫn:** Chương 2 (Tổng quan Transfer Learning cho VLM), Chương 3 & Chương 4 (Luận giải ảnh hưởng của độ phân giải đầu vào tới bài toán đọc văn bản nhỏ).

---

#### 27. Sigmoid Loss for Language Image Pre-Training (SigLIP)
* **Tệp:** [`Zhai_2023_SigLIP.pdf`](Papers/Model%20Info/Zhai_2023_SigLIP.pdf)
* **Tác giả:** Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer (Google Research)
* **arXiv / DOI:** [arXiv:2303.15343](https://arxiv.org/abs/2303.15343) | ICCV 2023
* **Nội dung chính:**
  - Thay thế hàm mất mát tương phản Softmax chuẩn bằng cặp Sigmoid loss độc lập theo batch, tạo ra bộ mã hóa ảnh SigLIP được dùng rộng rãi trong PaliGemma và SmolVLM.
* **Vị trí trích dẫn:** Chương 2 (Bộ mã hóa thị giác SigLIP và nền tảng căn chỉnh tương phản đa phương thức).

---

#### 28. LoRA: Low-Rank Adaptation of Large Language Models
* **Tệp:** [`Hu_2022_LoRA.pdf`](Papers/Model%20Info/Hu_2022_LoRA.pdf)
* **Tác giả:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu et al. (Microsoft)
* **arXiv / DOI:** [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | ICLR 2022
* **Nội dung chính:**
  - Đóng băng trọng số gốc $W_0$ và chèn các ma trận phân rã hạng thấp $BA$ song song vào các lớp chiếu tuyến tính của Transformer, giảm mạnh số tham số cần huấn luyện.
* **Vị trí trích dẫn:** Chương 2 (Cơ sở lý thuyết PEFT), Chương 3 (Thiết kế Dual Domain LoRA Adapters $\mathcal{A}_{\text{doc}}, \mathcal{A}_{\text{scene}}$ chống quên tri thức).

---

#### 29. QLoRA: Efficient Finetuning of Quantized LLMs
* **Tệp:** [`Dettmers_2023_QLoRA.pdf`](Papers/Model%20Info/Dettmers_2023_QLoRA.pdf)
* **Tác giả:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (University of Washington)
* **arXiv / DOI:** [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) | NeurIPS 2023
* **Nội dung chính:**
  - Đề xuất kiểu dữ liệu **4-bit NormalFloat (NF4)**, **Double Quantization** và **Paged Optimizers**, cho phép truyền ngược gradient qua trọng số lượng tử hóa 4-bit vào adapter 16-bit mà không suy giảm độ chính xác.
* **Vị trí trích dẫn:** Chương 3 & Chương 4 (Chiến lược tối ưu bộ nhớ VRAM: NF4 chọn lọc `llm_int8_skip_modules=["visual", "merger"]` và `paged_adamw_8bit`).

---

### Danh Mục BibTeX Trích Dẫn Bổ Sung (VLM Backbones & PEFT/Quantization)

```bibtex
@article{wang2024qwen2vl,
  title   = {Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author  = {Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and others},
  journal = {arXiv preprint arXiv:2409.12191},
  year    = {2024}
}

@inproceedings{dehghani2023patch,
  title     = {Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution},
  author    = {Dehghani, Mostafa and Mustafa, Basil and Djolonga, Josip and Heek, Jonathan and Minderer, Matthias and Caron, Mathilde and Steiner, Andreas and Puigcerver, Joan and Geirhos, Robert and Alabdulmohsin, Ibrahim and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}

@article{marafioti2025smolvlm,
  title   = {SmolVLM: Redefining small and efficient multimodal models},
  author  = {Marafioti, Andr{\'e}s and Zohar, Orr and Farr{\'e}, Miquel and Noyan, Merve and Bakouch, Elie and Cuenca, Pedro and Zakka, Cyril and Ben Allal, Loubna and Lozhkov, Anton and Tazi, Nouamane and others},
  journal = {arXiv preprint arXiv:2504.05299},
  year    = {2025}
}

@article{laurencon2024building,
  title   = {Building and better understanding vision-language models: insights and future directions},
  author  = {Lauren{\c{c}}on, Hugo and Marafioti, Andr{\'e}s and Sanh, Victor and Tronchon, L{\'e}o},
  journal = {arXiv preprint arXiv:2408.12637},
  year    = {2024}
}

@article{steiner2024paligemma2,
  title   = {PaliGemma 2: A Family of Versatile VLMs for Transfer},
  author  = {Steiner, Andreas and Pinto, Andr{\'e} Susano and Tschannen, Michael and Keysers, Daniel and Wang, Xiao and Bitton, Yonatan and Gritsenko, Alexey and Minderer, Matthias and Sherbondy, Anthony and Shang, Shangbang and others},
  journal = {arXiv preprint arXiv:2412.03555},
  year    = {2024}
}

@inproceedings{zhai2023sigmoid,
  title     = {Sigmoid Loss for Language Image Pre-Training},
  author    = {Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {11975--11986},
  year      = {2023}
}

@inproceedings{hu2022lora,
  title     = {LoRA: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022}
}

@inproceedings{dettmers2023qlora,
  title     = {QLoRA: Efficient Finetuning of Quantized LLMs},
  author    = {Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}
```

---

## 7. Bổ Sung: Công Bằng Ngân Sách Tính Toán & Đối Chiếu Thực Nghiệm (Compute-Budget Fairness)

> **Thời điểm cập nhật:** 26/11/2026

| STT | Tên Tệp PDF | Tiêu Đề Bài Báo | Tác Giả & Năm | Nguồn / Hội Nghị | arXiv ID | Vai Trò trong Luận Văn |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **30** | [`Dodge_2019_ShowYourWork.pdf`](Papers/Dodge_2019_ShowYourWork.pdf) | Show Your Work: Improved Reporting of Experimental Results | Jesse Dodge et al. (2019) | EMNLP 2019 | [`1909.03004`](https://arxiv.org/abs/1909.03004) | **Equal-Compute Methodology**: Chứng minh so sánh chỉ dựa trên điểm validation cao nhất mà bỏ qua ngân sách tính toán có thể đảo ngược kết luận; chuẩn hóa đối chiếu dưới ngân sách tính toán tương đương |
| **31** | [`Bartoldson_2023_ComputeEfficientDL.pdf`](Papers/Bartoldson_2023_ComputeEfficientDL.pdf) | Compute-Efficient Deep Learning: Algorithmic Trends and Opportunities | Brian R. Bartoldson et al. (2023) | JMLR 2023 | [`2210.06640`](https://arxiv.org/abs/2210.06640) | **Compute-Matched Evaluation**: Thiết lập chuẩn đánh giá công bằng theo FLOPs / số bước cập nhật tương đương (compute-matched), cô lập đóng góp thuật toán khỏi thời gian/khối lượng huấn luyện |
| **32** | [`Hoffmann_2022_Chinchilla.pdf`](Papers/Hoffmann_2022_Chinchilla.pdf) | Training Compute-Optimal Large Language Models (Chinchilla) | Jordan Hoffmann et al. (2022) | NeurIPS 2022 | [`2203.15556`](https://arxiv.org/abs/2203.15556) | **Fixed-Compute & Cosine Schedule**: Chuẩn hóa thực nghiệm theo ngân sách tính toán cố định và điều chỉnh chu kỳ cosine LR theo đúng tổng số bước mục tiêu (`MAX_TRAIN_STEPS`) |

### Chi Tiết Từng Bài Báo & Ý Nghĩa Trích Dẫn

#### 30. Show Your Work: Improved Reporting of Experimental Results
* **Tệp:** [`Dodge_2019_ShowYourWork.pdf`](Papers/Dodge_2019_ShowYourWork.pdf)
* **Tác giả:** Jesse Dodge, Suchin Gururangan, Dallas Card, Roy Schwartz, Noah A. Smith (Allen Institute for AI & University of Washington / CMU)
* **arXiv / DOI:** [arXiv:1909.03004](https://arxiv.org/abs/1909.03004) | EMNLP 2019
* **Nội dung chính:**
  - Chỉ ra rằng việc so sánh các mô hình chỉ dựa trên điểm validation tốt nhất (`best` checkpoint) hoặc điểm test cuối cùng mà không kiểm soát ngân sách tính toán có thể dẫn đến kết luận sai lệch hoặc đảo ngược hoàn toàn thứ hạng mô hình.
  - Đề xuất đánh giá theo đường cong hiệu năng–ngân sách tính toán (Expected Validation Performance) để đảm bảo công bằng giữa các phương pháp.
* **Vị trí trích dẫn:** Chương 3 & Chương 4 (Lập luận cân bằng số bước cập nhật optimizer giữa `B1`/`B2` và `B3`/`M1–M3`).

---

#### 31. Compute-Efficient Deep Learning: Algorithmic Trends and Opportunities
* **Tệp:** [`Bartoldson_2023_ComputeEfficientDL.pdf`](Papers/Bartoldson_2023_ComputeEfficientDL.pdf)
* **Tác giả:** Brian R. Bartoldson, Bhavya Kailkhura, Davis Blalock (Lawrence Livermore National Laboratory & MosaicML)
* **arXiv / DOI:** [arXiv:2210.06640](https://arxiv.org/abs/2210.06640) | JMLR 2023
* **Nội dung chính:**
  - Hệ thống hóa khái niệm tăng tốc thuật toán và đưa ra quy chuẩn đánh giá **compute-matched** (đối chiếu ở cùng mức FLOPs hoặc số bước tối ưu tương đương thay vì để một phương pháp chạy nhiều vòng lặp hơn).
* **Vị trí trích dẫn:** Chương 3 & Chương 4 (Thiết kế thực nghiệm đối chiếu công bằng giữa mô hình đơn miền và đa miền).

---

#### 32. Training Compute-Optimal Large Language Models (Chinchilla)
* **Tệp:** [`Hoffmann_2022_Chinchilla.pdf`](Papers/Hoffmann_2022_Chinchilla.pdf)
* **Tác giả:** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch et al. (DeepMind)
* **arXiv / DOI:** [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) | NeurIPS 2022
* **Nội dung chính:**
  - Chuẩn hóa phương pháp so sánh các mô hình dưới ngân sách tính toán cố định (fixed compute budget) và nhấn mạnh tầm quan trọng của việc co giãn lịch suy giảm **cosine learning rate** khớp đúng với tổng số bước huấn luyện mục tiêu.
* **Vị trí trích dẫn:** Chương 3 & Chương 4 (Cấu hình `MAX_TRAIN_STEPS` và tự động co giãn `get_cosine_schedule_with_warmup` trong `CrossVQATrainer`).

### Danh Mục BibTeX Trích Dẫn Bổ Sung (Compute-Budget Fairness)

```bibtex
@inproceedings{dodge2019show,
  title     = {Show Your Work: Improved Reporting of Experimental Results},
  author    = {Dodge, Jesse and Gururangan, Suchin and Card, Dallas and Schwartz, Roy and Smith, Noah A.},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP)},
  pages     = {2185--2194},
  year      = {2019}
}

@article{bartoldson2023compute,
  title   = {Compute-Efficient Deep Learning: Algorithmic Trends and Opportunities},
  author  = {Bartoldson, Brian R. and Kailkhura, Bhavya and Blalock, Davis},
  journal = {Journal of Machine Learning Research (JMLR)},
  volume  = {24},
  number  = {122},
  pages   = {1--77},
  year    = {2023}
}

@inproceedings{hoffmann2022training,
  title     = {Training Compute-Optimal Large Language Models},
  author    = {Hoffmann, Jordan and Borgeaud, Sebastian and Mensch, Arthur and Buchatskaya, Elena and Cai, Trevor and Rutherford, Eliza and de Las Casas, Diego and Hendricks, Lisa Anne and Welbl, Johannes and Clark, Aidan and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}
```




