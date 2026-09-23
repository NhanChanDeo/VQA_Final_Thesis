Tên đề tài: CrossVQA: Nghiên cứu chuyển giao tri thức liên miền từ ảnh tài liệu sang ảnh cảnh tự nhiên cho mô hình Thị giác–Ngôn ngữ
Nội dung chính:
Bộ dữ liệu (dataset) dự kiến sử dụng: DocVQA và TextVQA
Phân tích khoảng cách miền dữ liệu (domain gap) giữa văn bản tài liệu 2D phẳng, có cấu trúc và văn bản không có bố cục cấu trúc xác định xuất hiện trong không gian 3D với góc nhìn, ánh sáng, ngữ cảnh, phong cách khác nhau. Đồng thời, nghiên cứu về thách thức liên quan đến sự khác biệt đến cấu trúc dữ liệu giữa dataset DocVQA và TextVQA, từ đó đề ra phương pháp để tiền xử lý (preprocess) dữ liệu trước khi huấn luyện mô hình.

Đề xuất: quy trình thích nghi miền Domain Adaptation/Transfer Learning, sử dụng các kiến trúc Vision-Language hoặc bộ trích xuất đặc trưng đa phương thức để chuyển giao biểu diễn không gian và ngữ nghĩa từ DocVQA sang TextVQA.

Metric đánh giá performance của mô hình (dự kiến): ANLS (Average Normalized Levenshtein Similarity) và VQA Accuracy lần lượt với DocVQA và TextVQA.

Nhận xét, kết luận và đánh giá triển vọng ứng dụng và phát triển mô hình.


Cơ sở dữ liệu ban đầu:
TextVQA (huggingface: facebook/textvqa)
Nguồn ảnh: lấy từ tập OpenImages

Quy mô: 45,336 câu hỏi trên 28,408 ảnh.

Chia tách dữ liệu: train 34,602 / validation 5,000 / test 5,734 câu hỏi

Bài báo gốc: "Towards VQA Models That Can Read" (Singh et al., CVPR 2019)

Cấu trúc dataset:

image_id : Value('string')
question_id : Value('int32')
question : Value('string')
question_tokens : List(Value('string'))
image : Image(mode=None, decode=True)
image_width : Value('int32')
image_height : Value('int32')
flickr_original_url : Value('string')
flickr_300k_url : Value('string')
answers : List(Value('string'))
image_classes : List(Value('string'))
set_name : Value('string')

Giấy phép: CC BY 4.0

DocVQA (huggingface: lmms-lab-encoder/DocVQA)
Nguồn ảnh gốc: các ảnh tài liệu lấy từ UCSF Industry Documents Library, gồm nội dung in, đánh máy và viết tay; đa dạng loại tài liệu như thư từ, biên bản ghi nhớ, ghi chú, báo cáo

Quy mô gốc: 50.000 câu hỏi trên 12.767 ảnh, chia ngẫu nhiên theo tỷ lệ 80-10-10 (train 39.463 câu/10.194 ảnh, val 5.349 câu/1.286 ảnh, test 5.188 câu/1.287 ảnh)

Bài báo gốc: "DocVQA: A Dataset for VQA on Document Images" (Mathew, Karatzas, Jawahar — WACV 2021)

Repo lmms-lab/DocVQA trên HF: đây là bản đóng gói lại (12.1 GB, định dạng parquet) dùng cho benchmark trong hệ sinh thái lmms-eval, thường gồm hai subset con: DocVQA và InfographicVQA (một biến thể mở rộng với ảnh infographic).

Giấy phép: Apache-2.0

Cấu trúc dataset:

questionId : Value('string')
question : Value('string')
question_types : List(Value('string'))
image : Image(mode=None, decode=True)
docId : Value('int64')
ucsf_document_id : Value('string')
ucsf_document_page_no : Value('string')
answers : List(Value('string'))
data_split : Value('string')