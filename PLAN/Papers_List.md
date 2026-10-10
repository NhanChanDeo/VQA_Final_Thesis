# Danh Mục Bài Báo Nghiên Cứu (Downloaded Papers for CrossVQA)

> **Thư mục lưu trữ:** [`PLAN/Papers`](Papers)  
> **Đề tài:** CrossVQA: Nghiên cứu chuyển giao tri thức liên miền từ ảnh tài liệu sang ảnh cảnh tự nhiên cho mô hình Thị giác–Ngôn ngữ  
> **Bộ dữ liệu chính:** TextVQA (`facebook/textvqa`) & DocVQA (`lmms-lab/DocVQA`)  
> **Thời điểm cập nhật:** 10/10/2026  

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


