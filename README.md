# URA OCR - Solution Submit Tốt Nhất Hiện Tại

Thư mục này chứa solution V004, hiện được chọn làm baseline tốt nhất để submit
cho bài toán OCR của cuộc thi URA.

## Cấu trúc

```text
ura-ocr-repo/
├── README.md
├── ura_ocr_submission.txt
├── submission.csv
└── input/
    ├── train_labels.csv
    └── label_manifest.csv
```

| File | Mục đích |
|---|---|
| `ura_ocr_submission.txt` | Notebook dạng Python percent-format để đưa lên Kaggle |
| `submission.csv` | Kết quả V004 gồm 2.006 ảnh test |
| `input/train_labels.csv` | Nhãn dùng để huấn luyện product head |
| `input/label_manifest.csv` | Chia train/dev theo nhóm, tránh rò rỉ dữ liệu |

Ảnh test, `test.csv` và `sample_submission.csv` thuộc competition data trên
Kaggle nên không được sao chép vào repository này.

## Bài toán

Mỗi ảnh cần dự đoán hai trường:

- `ocr_text`: toàn bộ nội dung chữ nhận diện được;
- `product_name`: tên sản phẩm xuất hiện hoặc được đề cập trong ảnh.

Điểm tổng hợp:

```text
Score = 0.6 * Product F1 + 0.4 * (1 - OCR CER)
```

Do Product F1 chiếm 60%, solution không chỉ cố lấy nhiều chữ nhất. OCR cần giữ
được các token thương hiệu và tên sản phẩm, đồng thời hạn chế text nhiễu làm
sai bước suy luận sản phẩm.

## Cách tiếp cận

Pipeline gồm bốn bước.

### 1. Chuẩn hóa ảnh

Ảnh được đọc bằng Pillow, chuyển sang RGB và resize giữ nguyên tỷ lệ:

- cạnh dài tối thiểu: `720`;
- cạnh dài tối đa: `1280`;
- nội suy Lanczos;
- không crop cố định.

Giữ toàn bộ ảnh giúp tránh mất headline, watermark hoặc tên sản phẩm nằm sát
mép ảnh.

### 2. OCR bằng EasyOCR

EasyOCR chạy trên CPU với hai ngôn ngữ:

```python
easyocr.Reader(["vi", "en"], gpu=False)
```

Pass chính:

- đọc toàn bộ ảnh;
- giữ detection có confidence từ `0.25`;
- sắp xếp bounding box theo dòng và từ trái sang phải;
- ghép các detection thành `ocr_text`.

### 3. Retry có điều kiện

Nếu pass chính rỗng, quá ngắn hoặc có confidence thấp, ảnh được tăng contrast
và sharpen rồi OCR lại với threshold `0.20`.

Hai pass được so sánh bằng quality score:

```text
quality =
    0.55 * mean_confidence
  + 0.30 * length_score
  + 0.15 * box_score
```

Chỉ pass có quality cao hơn được sử dụng. Retry có điều kiện giúp tăng khả năng
đọc ảnh khó mà không áp dụng preprocessing mạnh cho toàn bộ dữ liệu.

### 4. Suy luận tên sản phẩm

`product_name` được dự đoán bằng hai lớp:

1. Luật regex và fuzzy matching cho các thương hiệu/họ sản phẩm phổ biến.
2. Mô hình nhẹ TF-IDF ký tự kết hợp Logistic Regression.

Các luật cụ thể được ưu tiên trước luật thương hiệu chung, ví dụ:

- Nestlé NAN INFINIPRO A2;
- Nestlé NAN OPTIPRO PLUS;
- Pate Cột Đèn Hải Phòng;
- Đồ Hộp Hạ Long;
- Ha Long Canfoco;
- Highlands Coffee;
- Aptamil.

Product head chỉ xuất dự đoán khi xác suất đạt threshold đã đóng băng:

```python
FROZEN_PRODUCT_THRESHOLD = 0.70
```

## Validation

Tập dev được chia theo `group_id` để các ảnh gần giống nhau không xuất hiện ở
cả tune và confirmation:

| Tập | Số ảnh | Product F1 | OCR CER | Composite |
|---|---:|---:|---:|---:|
| Tune | 100 | 0.6995 | 0.4507 | 0.6394 |
| Confirmation | 865 | 0.5904 | 0.4606 | 0.5700 |
| Toàn bộ dev | 965 | 0.6017 | 0.4595 | 0.5772 |

V004 được giữ làm baseline vì V009 dù cải thiện CER offline nhưng cho kết quả
Kaggle thấp hơn. Bài học chính là phải đánh giá toàn bộ composite, không được
promote chỉ dựa trên OCR CER.

## Cách submit

1. Tạo notebook Kaggle mới.
2. Thêm Competition Data của cuộc thi URA.
3. Thêm `train_labels.csv` và `label_manifest.csv` từ thư mục `input/` như một
   Kaggle Dataset nếu competition data chưa chứa hai file này.
4. Dùng nội dung `ura_ocr_submission.txt` làm notebook.
5. Giữ:

```python
RUN_MODE = "test"
```

6. Chạy toàn bộ notebook.
7. Submit file được tạo tại:

```text
/kaggle/working/submission.csv
```

Notebook kiểm tra số dòng, ID, null, newline và thứ tự cột trước khi ghi kết
quả. File `submission.csv` trong repository là output tham chiếu của V004.

