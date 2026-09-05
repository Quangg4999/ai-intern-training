# Bài 2.2 — Classification và threshold

## Mục tiêu

Dùng Logistic Regression phân loại Breast Cancer, báo cáo đủ metric và phân tích năm threshold theo [đề bài](https://docs.google.com/document/d/1bmBCggHgg5WIE0FH0iNsDSfu60QtCeFE/edit).

## Cài đặt từ môi trường sạch

Đã kiểm tra với Python 3.13 trên Windows. Mở PowerShell ở thư mục gốc repository:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt notebook==7.6.2 nbconvert==7.17.1
```

Các thư viện tính toán được cố định phiên bản trong `requirements.txt` ở gốc repository. Dataset có sẵn trong scikit-learn. Không commit môi trường `.venv`.

## Chạy bài

Từ thư mục gốc repository:

```powershell
.\.venv\Scripts\python.exe -m notebook week02/bai2.2/assignment_classification.ipynb
```

Mở notebook, chọn kernel Python của môi trường `.venv`, rồi **Restart Kernel and Run All Cells**. Đọc các cell từ trên xuống và lưu notebook sau khi chạy.

Cũng có thể chạy toàn bộ bằng terminal (vẫn ở thư mục gốc repository):

```powershell
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=120 week02/bai2.2/assignment_classification.ipynb
```

Notebook lưu biểu đồ vào `outputs/precision_recall_threshold.png` trong thư mục bài khi chạy theo các lệnh trên. Nếu thay dữ liệu hoặc cấu hình, cần cập nhật nhận xét và số liệu trong report theo output mới.

## Cấu hình và quy ước

- Train/test = 80/20; `random_state=42`, `stratify=y`; không dùng validation trong bài này.
- `StandardScaler` chỉ fit trên train, sau đó transform train và test.
- `LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)`.
- Giữ nhãn gốc: **0 = malignant (positive), 1 = benign (negative)**.
- Lấy xác suất nhãn 0 bằng vị trí trong `model.classes_`; áp quy tắc `P(positive) >= threshold`.
- Threshold: **0.2, 0.3, 0.5, 0.7, 0.8**. Metric đánh giá trên test.

## Bài nộp và kết quả chính

Trên test, dự đoán ban đầu đạt Accuracy **0.982456**, Precision/Recall/F1 đều **0.976190**, với **FP = 1, FN = 1**. Khi nâng threshold từ **0.2 lên 0.8**, FP giảm từ **6 xuống 0**, còn FN tăng từ **1 lên 2**.

- [Notebook đã chạy](assignment_classification.ipynb): Accuracy, Precision, Recall, F1, Confusion Matrix, bảng năm threshold và biểu đồ.
- [Báo cáo](report.md): số liệu cụ thể và phân tích tác động threshold trong tình huống AV endpoint.
- [Biểu đồ Precision/Recall](outputs/precision_recall_threshold.png).
- [Lý thuyết tuần 2](../theory.md).

## Điều học được

Phân biệt xác suất và nhãn dự đoán; tránh rò rỉ dữ liệu bằng cách chỉ fit scaler trên train; xác định positive nhất quán; đọc TP/FP/TN/FN và sự đánh đổi giữa phát hiện malware với báo động nhầm khi thay threshold.

Trước buổi review, tự chạy lại, tự viết/điều chỉnh nhận xét theo cách hiểu của mình và giải thích được từng cell. Theo đề, mentor có thể đổi input hoặc yêu cầu sửa code trực tiếp.
