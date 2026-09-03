# Lý thuyết nền — Bài 2.1

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
