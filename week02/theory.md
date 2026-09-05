# Lý thuyết nền — Tuần 2

## Regression và classification

Regression dự đoán một giá trị số liên tục, ví dụ giá nhà hoặc doanh thu. Classification dự đoán một nhãn rời rạc, ví dụ benign/malignant. Bài 2.1 là regression vì `y` là số thực.

## Linear Regression

Mô hình đường thẳng là:

\[
\hat{y} = wx + b
\]

- `w` là hệ số góc: khi `x` tăng 1, dự đoán thay đổi khoảng `w`.
- `b` là intercept: dự đoán khi `x = 0`.
- `\hat{y}` là giá trị model dự đoán.

Trong bài, dữ liệu được tạo theo `y = 3.5x + 20 + noise`, nên mô hình cần học các giá trị gần `w = 3.5`, `b = 20`.

## MSE

MSE đo sai số trung bình bình phương:

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
\]

Bình phương khiến sai số lớn bị phạt mạnh hơn và luôn cho giá trị không âm. MSE càng thấp càng tốt, nhưng luôn phải đọc trên train và test để kiểm tra khả năng tổng quát hóa.

## Gradient Descent

Gradient Descent bắt đầu từ `w = 0`, `b = 0`, tính độ dốc của MSE rồi cập nhật:

\[
w \leftarrow w - learning\_rate \times \frac{\partial MSE}{\partial w}
\]

\[
b \leftarrow b - learning\_rate \times \frac{\partial MSE}{\partial b}
\]

Với Linear Regression một biến:

\[
\frac{\partial MSE}{\partial w} = \frac{2}{n}\sum x_i(\hat{y}_i-y_i)
\]

\[
\frac{\partial MSE}{\partial b} = \frac{2}{n}\sum (\hat{y}_i-y_i)
\]

`learning_rate` quá nhỏ làm loss giảm rất chậm. Nó quá lớn làm mỗi bước vượt quá điểm thấp nhất của loss, có thể dao động hoặc diverge.

## Epoch và train/test split

Một epoch là một lần Gradient Descent nhìn toàn bộ tập train. Bài này chạy tối thiểu 1.000 epoch cho từng learning rate và chạy 100.000 epoch cho cấu hình ổn định được chọn để kiểm tra hội tụ đầy đủ.

Tập dữ liệu được chia 80% train và 20% test. Train được dùng để học `w`, `b`; test chỉ được dùng sau đó để đánh giá mô hình trên dữ liệu chưa thấy.

## Logistic Regression và sigmoid — Bài 2.2

Logistic Regression dùng tổng có trọng số `z = wᵀx + b`, rồi sigmoid `p = 1 / (1 + exp(-z))` để đưa giá trị về khoảng 0–1. Trong sklearn với bài toán nhị phân, sigmoid của `decision_function` ứng với `classes_[1]`. Bài 2.2 chọn positive là nhãn 0, nên lấy đúng cột của nhãn 0 trong `predict_proba` theo `model.classes_`.

Probability là xác suất do model ước lượng. Threshold là ngưỡng để đổi xác suất positive thành quyết định: `p_positive >= threshold` thì dự đoán positive. Thay ngưỡng không huấn luyện lại model.

## Chuẩn hóa và tránh rò rỉ dữ liệu

StandardScaler biến đổi từng feature theo `z = (x - mean_train) / std_train`. Phải chia train/test trước và chỉ fit scaler trên train. Test dùng lại scaler đó. Nếu fit trên toàn bộ dữ liệu, bước chuẩn bị đã dùng thông tin của tập đánh giá.

## Metric classification

- TP: positive được đoán positive; FP: negative bị đoán positive.
- TN: negative được đoán negative; FN: positive bị đoán negative.
- Accuracy = `(TP + TN) / (TP + FP + TN + FN)`.
- Precision = `TP / (TP + FP)`: trong các cảnh báo, tỷ lệ đúng là bao nhiêu?
- Recall = `TP / (TP + FN)`: phát hiện được bao nhiêu positive thật?
- F1 = `2 * Precision * Recall / (Precision + Recall)`.

Trong bài 2.2, positive = malignant (nhãn 0), các metric dùng `pos_label=0`. Confusion Matrix đặt thứ tự `[negative, positive] = [1, 0]`, hàng là thật, cột là dự đoán, nên đọc thành `[[TN, FP], [FN, TP]]`.

Hạ threshold khiến nhiều mẫu được đoán positive hơn: Recall không giảm nhưng FP có thể tăng. Nâng threshold giảm hoặc giữ nguyên FP nhưng có thể tăng FN. Precision không được bảo đảm đơn điệu. Với giả định positive là malware, đây là sự đánh đổi giữa báo động/chặn nhầm và bỏ lọt mã độc.

## Overfitting và underfitting cơ bản

Overfitting là học quá sát dữ liệu train, kể cả nhiễu, nên tổng quát hóa kém trên dữ liệu mới. Underfitting là mô hình chưa nắm đủ quy luật, thường cho kết quả kém cả train và dữ liệu mới. Tập test dùng đánh giá sau training; nếu liên tục chọn cấu hình theo test thì kết quả không còn là đánh giá độc lập. Bài 2.2 chỉ so sánh năm threshold được giao và không tuyên bố tìm ngưỡng tối ưu triển khai.
