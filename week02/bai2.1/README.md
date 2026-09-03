# Bài 2.1 — Linear Regression từ đầu

## 1. Mục tiêu

Hoàn thành hai cách fit cùng một dữ liệu tuyến tính có nhiễu:

1. Dùng `sklearn.linear_model.LinearRegression` làm mốc đối chiếu.
2. Tự cài đặt Linear Regression bằng batch Gradient Descent, không dùng sklearn để train phần này.

Chương trình sinh dữ liệu có thể lặp lại bằng seed 42, chia 80% train và 20% test, báo cáo MSE, thử bốn learning rate và tạo hai biểu đồ bắt buộc.

## 2. Cấu trúc thư mục

```text
bai2.1/
├── assignment_linear_regression.ipynb
├── linear_regression_experiment.py
├── report.md
└── outputs/
    ├── regression_line.png
    └── loss_by_learning_rate.png
```

Notebook là bài nộp chính để đọc tuần tự. File Python chứa cùng logic, dùng để chạy lại toàn bộ experiment từ terminal và tạo hai hình trong `outputs/`.

## 3. Setup từ môi trường sạch

Thực hiện từ thư mục gốc repository:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu dùng VS Code, chọn interpreter `.venv\\Scripts\\python.exe`, mở notebook rồi chọn **Run All**. Nếu cần chạy notebook bằng trình duyệt, cài thêm Jupyter vào chính môi trường đó:

```powershell
.\.venv\Scripts\python.exe -m pip install notebook
.\.venv\Scripts\python.exe -m notebook
```

## 4. Chạy bài, theo thứ tự

### Bước 1 — Chạy code tái lập experiment

```powershell
.\.venv\Scripts\python.exe week02\bai2.1\linear_regression_experiment.py
```

Bạn sẽ thấy bốn phần: công thức dữ liệu, kết quả sklearn, bảng learning rate và lần chạy Gradient Descent cuối cùng.

### Bước 2 — Kiểm tra dữ liệu đầu vào

Trong notebook, xem cell tạo dữ liệu:

```python
np.random.seed(42)
x = np.linspace(0, 100, 200)
y = 3.5 * x + 20 + np.random.normal(0, 20, 200)
```

- `x` gồm 200 điểm từ 0 đến 100.
- `3.5*x + 20` là đường thẳng thật.
- `np.random.normal(0, 20, 200)` thêm nhiễu Gaussian có mean 0, standard deviation 20.
- Seed 42 bảo đảm mỗi lần chạy tạo đúng cùng một dataset.

### Bước 3 — Chạy baseline sklearn

`LinearRegression` học hệ số `w` và intercept `b`. So sánh Train MSE với Test MSE: nếu hai số gần nhau, chưa có bằng chứng rõ ràng model overfit trên dữ liệu này.

### Bước 4 — Đọc Gradient Descent

Các hàm cần hiểu theo thứ tự:

1. `predict(x, w, b)`: tính `w*x + b`.
2. `mse(y_true, y_pred)`: tính lỗi bình phương trung bình.
3. `compute_gradients(...)`: tính hướng và độ lớn cần thay đổi của `w`, `b`.
4. `train(...)`: lặp qua các epoch, cập nhật `w`, `b`, rồi lưu loss.

### Bước 5 — Đọc learning rate

Tất cả bốn learning rate chạy 1.000 epoch để đáp ứng mức tối thiểu của đề. Lần chạy cuối dùng `learning_rate = 0.0001` trong 100.000 epoch để thấy nó thực sự tiến gần nghiệm tối ưu. Chi tiết và kết luận nằm trong [report.md](report.md).

### Bước 6 — Xem hai biểu đồ

- `outputs/regression_line.png`: dữ liệu train/test và đường fit của sklearn, Gradient Descent.
- `outputs/loss_by_learning_rate.png`: Train MSE theo epoch, trục y log để vẫn quan sát được learning rate diverge.

## 5. Kết quả chính

Với seed 42 và split `random_state=42`:

| Phương pháp | w | b | Train MSE | Test MSE |
|---|---:|---:|---:|---:|
| sklearn | 3,533703 | 17,060662 | 352,227434 | 308,766067 |
| Gradient Descent, lr=0,0001, 100.000 epoch | 3,535011 | 16,972281 | 352,229488 | 308,737406 |

## 6. Điều đã học

- MSE là loss để đo sai lệch giữa dự đoán và giá trị thật.
- Gradient cho biết hướng giảm MSE; learning rate quyết định độ dài mỗi bước.
- Dữ liệu có noise nên hệ số học được không bắt buộc đúng tuyệt đối 3,5 và 20.
- Cần đánh giá trên test, không chỉ nhìn train loss.
