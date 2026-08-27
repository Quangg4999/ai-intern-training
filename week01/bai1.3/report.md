# Báo cáo Bài 1.3 – Phân tích dữ liệu (EDA)

**Dataset:** `sklearn.datasets.load_breast_cancer()` (Breast Cancer Wisconsin – Diagnostic)
**Notebook đính kèm:** `assignment_eda.ipynb`
**Phạm vi:** Chỉ phân tích dữ liệu, không train model.

---

## 1. Tổng quan dataset

| Chỉ tiêu | Giá trị |
|---|---|
| Số sample | 569 |
| Số feature | 30 |
| Số class | 2 |
| Tên label | `malignant` (ác tính), `benign` (lành tính) |
| Quy ước mã hoá | 0 = malignant, 1 = benign |
| Kiểu dữ liệu | Toàn bộ `float64` |

Mỗi dòng là kết quả đo từ ảnh chọc hút kim nhỏ (FNA) một khối u vú. Các feature mô tả đặc điểm hình học của nhân tế bào.

### Cấu trúc 30 feature

30 cột thực chất là **10 phép đo gốc** × **3 cách thống kê**:

| Nhóm | Ý nghĩa | Số cột |
|---|---|---|
| `mean ...` | Trung bình trên các nhân tế bào trong mẫu | 10 |
| `... error` | Sai số chuẩn của phép đo | 10 |
| `worst ...` | Trung bình 3 giá trị lớn nhất | 10 |

10 phép đo gốc: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension.

Đây là chi tiết cấu trúc quan trọng, vì nó giải thích trước hiện tượng đa cộng tuyến ở mục 5.

---

## 2. Kiểm tra missing values

| Kiểm tra | Kết quả |
|---|---|
| Tổng số ô `NaN` trong 569 × 30 = 17.070 ô | **0** |
| Số ô `NaN` ở cột nhãn | 0 |
| Số ô có giá trị vô cực (`inf`) | 0 |
| Số dòng trùng lặp hoàn toàn | 0 |

**Kết luận:** Dataset sạch hoàn toàn về mặt kỹ thuật, không cần bước xử lý missing value.

**Lưu ý nghiệp vụ:** Đây là dataset mẫu đã được làm sạch sẵn. Trong dự án thực tế, `isnull().sum()` trả về 0 chưa đủ để kết luận dữ liệu sạch — giá trị thiếu thường được mã hoá thành `-1`, `0` hoặc `999`, là các giá trị hợp lệ về kiểu dữ liệu nên `isnull()` không phát hiện được.

---

## 3. Thống kê mô tả (mean / std / min / max)

Bảng đầy đủ 30 feature nằm trong notebook. Dưới đây là các mốc đáng chú ý:

| Feature | mean | std | min | max |
|---|---|---|---|---|
| `worst area` | 880,5831 | 569,3570 | 185,20 | 4254,00 |
| `mean area` | 654,8891 | 351,9141 | 143,50 | 2501,00 |
| `worst perimeter` | 107,2612 | 33,6025 | 50,41 | 251,20 |
| `mean radius` | 14,1273 | 3,5240 | 6,981 | 28,11 |
| `area error` | 40,3371 | 45,4910 | 6,802 | 542,20 |
| `mean smoothness` | 0,0964 | 0,0141 | 0,0526 | 0,1634 |
| `fractal dimension error` | 0,0038 | 0,0026 | 0,0009 | 0,0298 |

**Các điểm rút ra:**

- Feature có mean lớn nhất: `worst area` = 880,58
- Feature có mean nhỏ nhất: `fractal dimension error` = 0,0038
- **Tỉ lệ chênh lệch thang đo: khoảng 232.000 lần**
- `area error` có std (45,49) **lớn hơn cả mean** (40,34), tức CV > 1 — dấu hiệu của phân phối lệch mạnh với đuôi dài

---

## 4. Phân bố class

| Mã | Tên class | Số lượng | Tỉ lệ |
|---|---|---|---|
| 0 | malignant | 212 | 37,26% |
| 1 | benign | 357 | 62,74% |

Tỉ lệ benign : malignant ≈ **1,68 : 1** — mất cân bằng nhẹ, chưa nghiêm trọng nhưng đủ để ảnh hưởng tới việc chọn chỉ số đánh giá.

**Baseline cần ghi nhớ:** Một model luôn dự đoán "benign" cho mọi mẫu vẫn đạt accuracy 62,74%. Mọi model xây dựng sau này phải vượt rõ rệt con số này mới có giá trị.

---

## 5. 10 cặp feature có |correlation| cao nhất

### Phương pháp

Ma trận correlation 30 × 30 có 900 ô, nhưng chỉ chứa **435 cặp thực sự khác nhau** (= 30 × 29 / 2), vì 30 ô đường chéo luôn bằng 1 và ma trận đối xứng. Đã dùng `np.triu(..., k=1)` để lấy tam giác trên không tính đường chéo, tránh lỗi top-10 bị lấp đầy bởi đường chéo và các cặp trùng lặp.

### Kết quả

| # | Feature 1 | Feature 2 | Correlation | \|Correlation\| |
|---|---|---|---|---|
| 1 | mean radius | mean perimeter | 0,9979 | 0,9979 |
| 2 | worst radius | worst perimeter | 0,9937 | 0,9937 |
| 3 | mean radius | mean area | 0,9874 | 0,9874 |
| 4 | mean perimeter | mean area | 0,9865 | 0,9865 |
| 5 | worst radius | worst area | 0,9840 | 0,9840 |
| 6 | worst perimeter | worst area | 0,9776 | 0,9776 |
| 7 | radius error | perimeter error | 0,9728 | 0,9728 |
| 8 | mean perimeter | worst perimeter | 0,9704 | 0,9704 |
| 9 | mean radius | worst radius | 0,9695 | 0,9695 |
| 10 | mean perimeter | worst radius | 0,9695 | 0,9695 |

### Phân bố mức tương quan trên toàn bộ 435 cặp

| Ngưỡng | Số cặp |
|---|---|
| \|r\| > 0,9 | 21 |
| \|r\| > 0,8 | 44 |

Cặp có tương quan **âm** mạnh nhất chỉ đạt r = -0,3116 (`mean radius` – `mean fractal dimension`), yếu hơn nhiều so với các tương quan dương.

### Liên hệ với nhãn (phân tích bổ sung)

Nhắc lại: 0 = malignant, nên tương quan **âm** với `target` nghĩa là feature càng lớn thì khả năng ác tính càng cao.

| Feature | Correlation với target |
|---|---|
| `worst concave points` | -0,7936 |
| `worst perimeter` | -0,7829 |
| `mean concave points` | -0,7766 |
| `worst radius` | -0,7765 |
| `mean perimeter` | -0,7426 |

Ngược lại, các feature gần như không liên quan tới nhãn: `symmetry error` (0,0065), `texture error` (0,0083), `mean fractal dimension` (0,0128).

---

## 6. Biểu đồ

### Histogram (5 feature)
Đã vẽ cho: `mean radius`, `mean texture`, `mean area`, `mean smoothness`, `mean concavity`. Bổ sung thêm bản histogram tách theo class để đánh giá khả năng phân biệt.

Độ lệch (skewness) của 5 feature này:

| Feature | Skewness |
|---|---|
| `mean area` | 1,646 |
| `mean concavity` | 1,401 |
| `mean radius` | 0,942 |
| `mean texture` | 0,650 |
| `mean smoothness` | 0,456 |

Các feature lệch nhất toàn dataset đều thuộc nhóm `error`: `area error` (5,447), `concavity error` (5,110), `fractal dimension error` (3,924).

### Correlation matrix (10 feature đầu)
Heatmap `imshow` với thang màu `coolwarm`, có ghi giá trị số trong từng ô. Quan sát rõ nhất: cụm radius – perimeter – area sáng đỏ đậm (r ≈ 0,99), trong khi `mean texture` gần như trung tính với mọi feature khác (r ≈ 0,02 – 0,33).

---

## 7. Năm nhận xét từ dữ liệu

### Nhận xét 1 — Dữ liệu sạch về kỹ thuật, nhưng "sạch" không đồng nghĩa với "không cần kiểm tra"

Không có `NaN`, `inf`, hay dòng trùng lặp; toàn bộ 30 cột là `float64`. Không cần xử lý missing value. Tuy nhiên đây là dataset mẫu đã qua làm sạch, và trong thực tế cần kiểm tra thêm các giá trị thay thế được mã hoá bằng số hợp lệ.

### Nhận xét 2 — Class mất cân bằng khiến accuracy trở thành chỉ số dễ gây hiểu lầm

Với 62,74% mẫu thuộc lớp benign, một model không học gì vẫn đạt accuracy 62,74%. Cần dùng precision, recall, F1-score và confusion matrix thay thế.

Trong ngữ cảnh y tế, **recall của lớp malignant là chỉ số quan trọng nhất**: bỏ sót một khối u ác tính (false negative) nguy hiểm hơn nhiều so với báo động nhầm một khối u lành tính (false positive).

### Nhận xét 3 — Chênh lệch thang đo hơn 200.000 lần, bắt buộc chuẩn hoá trước khi train

`worst area` có mean 880,58 trong khi `fractal dimension error` có mean 0,0038.

Nối trực tiếp với hàm `euclidean_distance` đã tự cài đặt ở Bài 1.1: khoảng cách Euclid cộng bình phương chênh lệch của từng feature, nên `worst area` sẽ chi phối gần như toàn bộ giá trị khoảng cách, còn các feature thang đo nhỏ bị triệt tiêu hoàn toàn — bất kể ý nghĩa y học của chúng.

Kết luận: KNN, SVM, K-Means, hồi quy logistic và mạng nơ-ron **bắt buộc** chuẩn hoá (`StandardScaler` hoặc `MinMaxScaler`). Cây quyết định và Random Forest không cần, vì chỉ so sánh ngưỡng trên từng feature riêng lẻ.

### Nhận xét 4 — Đa cộng tuyến nghiêm trọng do quan hệ hình học tất định

21 trong 435 cặp có |r| > 0,9; cả top 10 đều ≥ 0,969.

Nguyên nhân không ngẫu nhiên: với hình gần tròn, chu vi = 2πr và diện tích = πr². Ba feature radius, perimeter, area đo cùng một đại lượng là kích thước khối u, chỉ khác đơn vị. Tương tự, ba nhóm `mean` / `error` / `worst` xuất phát từ cùng tập phép đo gốc.

Hệ quả: lượng thông tin thực tế thấp hơn nhiều so với con số 30 feature. Điều này gây bất ổn hệ số trong hồi quy tuyến tính và tăng nguy cơ overfitting. Hướng xử lý: loại bỏ feature dư thừa hoặc giảm chiều bằng PCA.

### Nhận xét 5 — Phân phối lệch phải mạnh, đuôi dài mang tín hiệu bệnh lý chứ không phải nhiễu

Không feature nào phân phối chuẩn. `area error` có skewness 5,447, mean 40,34 nhưng max 542,20 — gấp hơn 13 lần mean.

Hai hệ quả:
1. Với phân phối lệch, mean bị giá trị cực đoan kéo lệch; **median mô tả giá trị điển hình tốt hơn**.
2. Histogram tách theo class cho thấy đuôi phải chủ yếu là mẫu malignant. Do đó **không được loại bỏ chúng như outlier thông thường** — đó chính là tín hiệu cần giữ lại. Nếu cần đưa về gần chuẩn, dùng biến đổi log hoặc Box-Cox thay vì cắt bỏ.

---

## 8. Trả lời câu hỏi nghiệm thu

### 8.1 Ý nghĩa của mean, std, correlation

**Mean** mô tả vị trí trung tâm của dữ liệu. Điểm yếu: rất nhạy với giá trị cực đoan — `area error` là ví dụ trong chính dataset này.

**Std** là căn bậc hai của phương sai, đo mức phân tán quanh giá trị trung bình, và có cùng đơn vị với dữ liệu gốc nên đọc trực tiếp được.

Điểm mấu chốt: **mean và std phải đọc cùng nhau**; riêng mean không nói lên độ tin cậy của con số. Muốn so sánh mức phân tán giữa các feature khác đơn vị, phải dùng hệ số biến thiên CV = std / mean, vì std của `mean area` (351,91) và của `mean smoothness` (0,0141) không so sánh trực tiếp được.

**Correlation (Pearson)** đo mức chặt chẽ của quan hệ **tuyến tính** giữa hai biến, nằm trong [-1, 1]. Về hình học, đây chính là cosine similarity giữa hai vector đã trừ giá trị trung bình — cùng cấu trúc với hàm `cosine_similarity` đã cài đặt ở Bài 1.1.

Hai giới hạn quan trọng:
- **r = 0 không có nghĩa là độc lập.** Với y = x² và x đối xứng quanh 0, r ≈ 0 dù y hoàn toàn xác định bởi x. Pearson mù với quan hệ phi tuyến.
- **r rất nhạy với outlier.** Vài điểm cực đoan có thể tạo ra hoặc phá huỷ một hệ số tương quan.

### 8.2 Vì sao correlation không chứng minh quan hệ nhân quả

Correlation chỉ khẳng định hai biến **biến thiên cùng nhau**; nhân quả khẳng định biến này **gây ra** biến kia. Khi quan sát A tương quan với B, có ít nhất bốn khả năng:

**1. Chiều nhân quả ngược lại.** Correlation đối xứng: r(A,B) = r(B,A). Bản thân con số không chứa thông tin về chiều.

**2. Biến gây nhiễu (confounding).** Một nguyên nhân thứ ba tác động lên cả hai. Ví dụ kinh điển: doanh số kem và số vụ đuối nước tương quan mạnh, nhưng nhiệt độ mùa hè mới là nguyên nhân chung.

Ví dụ ngay trong dataset: `mean radius` và `mean perimeter` có r = 0,9979, nhưng bán kính không "gây ra" chu vi — cả hai là hệ quả toán học của cùng một đại lượng, theo công thức chu vi = 2πr.

**3. Trùng hợp ngẫu nhiên.** Với 435 cặp được kiểm tra, một số cặp có tương quan cao thuần tuý do may rủi là điều thống kê dự đoán được (vấn đề *multiple comparisons*).

**4. Thiên lệch chọn mẫu.** Dataset chỉ gồm bệnh nhân **đã đi khám và đã được chỉ định sinh thiết**, không phải mẫu ngẫu nhiên từ dân số chung. Quan hệ quan sát được có thể không đúng ngoài phạm vi nhóm này.

**Áp dụng vào bài toán hiện tại:** `worst concave points` có r = -0,7936 với nhãn, tức khối u có nhiều điểm lõm sâu thì khả năng ác tính cao. Nhưng dữ liệu này **không** cho phép kết luận điểm lõm *gây ra* ung thư. Nhiều khả năng cả hai đều là biểu hiện của cùng một quá trình sinh học: tế bào ác tính phát triển mất trật tự, làm biến dạng đường viền nhân tế bào.

Phân biệt cuối cùng:
- Với mục tiêu **dự đoán**, tương quan là đủ — model vẫn hoạt động tốt mà không cần biết nhân quả.
- Với mục tiêu **can thiệp** (thay đổi X để đạt Y), bắt buộc phải có bằng chứng nhân quả, đến từ thí nghiệm có đối chứng ngẫu nhiên (RCT) hoặc các phương pháp suy luận nhân quả từ dữ liệu quan sát (biến công cụ, hồi quy gián đoạn, difference-in-differences).

*Correlation là điều kiện cần nhưng không đủ để kết luận nhân quả.*

---

## 9. Kết luận và hướng tiếp theo

Dataset sạch, không cần xử lý missing value, nhưng có ba đặc điểm phải xử lý trước khi train model:

1. **Chuẩn hoá thang đo** — bắt buộc với mọi thuật toán dựa trên khoảng cách hoặc gradient.
2. **Xử lý đa cộng tuyến** — loại feature dư thừa hoặc dùng PCA.
3. **Chọn chỉ số đánh giá phù hợp** — ưu tiên recall của lớp malignant thay vì accuracy.

Ngoài ra cần chia train/test có **phân tầng theo class** (`stratify=y`) để giữ nguyên tỉ lệ 62,74% / 37,26% ở cả hai tập.
