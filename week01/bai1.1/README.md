# Bài 1.1 – Tự cài đặt phép toán vector

## 1. Mô tả

File `vector_math.py` tự cài đặt bốn phép toán vector cơ bản **không dùng sklearn**, và không dùng `np.dot` / `np.linalg.norm` trong phần cài đặt chính. NumPy chỉ xuất hiện ở phần kiểm chứng cuối file.

Bốn hàm đã cài đặt:

| Hàm | Công thức | Ý nghĩa |
|---|---|---|
| `dot_product(a, b)` | $\sum_{i} a_i b_i$ | Tích vô hướng |
| `vector_norm(a)` | $\sqrt{\sum_i a_i^2}$ | Độ dài vector (chuẩn L2) |
| `euclidean_distance(a, b)` | $\sqrt{\sum_i (a_i - b_i)^2}$ | Khoảng cách giữa hai điểm |
| `cosine_similarity(a, b)` | $\dfrac{a \cdot b}{\|a\| \, \|b\|}$ | Độ tương đồng về hướng |

Ba hàm sau đều được xây trên `dot_product`: `vector_norm(a)` gọi `dot_product(a, a)`, `euclidean_distance` tính vector hiệu rồi gọi `vector_norm`, `cosine_similarity` gọi cả hai. Chỉ cần một hàm gốc đúng thì cả ba hàm còn lại đều đúng theo.

## 2. Cách chạy

```bash
# Từ thư mục gốc của repo
cd week01/bai1.1
python vector_math.py
```

Môi trường: Python 3.14 + NumPy (xem `requirements.txt` ở thư mục gốc repo).

## 3. Kết quả

Dữ liệu đầu vào:

```
A = [1, 2, 3, 4]
B = [2, 4, 6, 8]
C = [8, 1, 0, 2]
```

| Đại lượng | Tự cài đặt | NumPy | Sai số tuyệt đối |
|---|---|---|---|
| `cos(A, B)` | 1.0 | 1.0 | **0.0** |
| `cos(A, C)` | 0.39562828403747224 | 0.39562828403747224 | **0.0** |
| `distance(A, B)` | 5.477225575051661 | 5.477225575051661 | **0.0** |
| `distance(A, C)` | 7.937253933193772 | 7.937253933193772 | **0.0** |

Sai số bằng đúng 0 ở cả bốn phép, không phải xấp xỉ 0.

Kết quả này không hiển nhiên. Số thực dấu phẩy động không có tính kết hợp: `(x + y) + z` và `x + (y + z)` có thể cho kết quả khác nhau ở chữ số cuối. Sai số bằng 0 tuyệt đối chứng tỏ vòng lặp cộng dồn trong `dot_product` thực hiện các phép cộng theo **đúng thứ tự** mà NumPy dùng nội bộ, nên hai bên tạo ra chuỗi bit giống hệt nhau.

Với vector dài hàng nghìn phần tử, điều này sẽ không còn đúng: NumPy dùng cộng theo khối (pairwise summation) để giảm tích luỹ sai số, còn vòng lặp tuần tự thì không. Khi đó sai số sẽ khác 0, cỡ $10^{-15}$ — vẫn chấp nhận được, nhưng không còn bằng 0 tuyệt đối.

---

## 4. Trả lời câu hỏi

### Câu 1: Vì sao A và B không bằng nhau nhưng cosine similarity bằng 1?

Vì **B là bội số vô hướng của A**: `B = 2 × A`. Hai vector này khác nhau về **độ lớn** nhưng trùng nhau hoàn toàn về **hướng**.

Cosine similarity chỉ đo hướng, không đo độ lớn. Có thể thấy điều này ngay từ công thức:

$$\cos(A, 2A) = \frac{A \cdot 2A}{\|A\| \cdot \|2A\|} = \frac{2(A \cdot A)}{\|A\| \cdot 2\|A\|} = \frac{2\|A\|^2}{2\|A\|^2} = 1$$

Hệ số 2 xuất hiện ở cả tử số và mẫu số nên triệt tiêu. Kết luận tổng quát: với mọi $k > 0$, $\cos(A, kA) = 1$.

Nguyên nhân sâu hơn nằm ở phép chia cho $\|A\| \cdot \|B\|$ ở mẫu số. Phép chia này **chuẩn hoá** hai vector về độ dài 1 trước khi so sánh, tức là xoá bỏ thông tin về độ lớn. Những gì còn lại chỉ là góc giữa hai vector, mà góc giữa A và 2A bằng 0, nên cosine bằng 1.

Đối chiếu với Euclidean distance trên cùng dữ liệu: `distance(A, B) = 5.477`, hoàn toàn khác 0. Hai thước đo cho hai kết luận trái ngược trên cùng một cặp vector:

- **Cosine nói:** giống hệt nhau (cùng hướng)
- **Euclidean nói:** cách nhau khá xa (khác độ lớn)

Cả hai đều đúng. Chúng chỉ đang trả lời hai câu hỏi khác nhau. Điều này dẫn thẳng tới câu 2.

### Câu 2: Khi nào cosine similarity hữu ích hơn Euclidean distance?

Câu trả lời ngắn: **khi độ lớn của vector là nhiễu, còn tỉ lệ giữa các thành phần mới mang thông tin.**

Bốn tình huống cụ thể:

**a) Văn bản có độ dài khác nhau.** Đây là trường hợp gặp ngay ở Bài 1.2. Một bài báo 2000 từ về malware và một câu 10 từ về malware sẽ có vector đếm từ chênh lệch rất lớn về độ lớn, nên Euclidean distance giữa chúng rất xa. Nhưng cosine similarity vẫn cao, vì **tỉ lệ** các từ khoá tương tự nhau. Người dùng tìm kiếm quan tâm chủ đề, không quan tâm độ dài văn bản.

Đây chính là lý do `TfidfVectorizer` mặc định chuẩn hoá L2 mọi vector về độ dài 1, và cũng là lý do mọi hệ thống tìm kiếm dựa trên vector đều dùng cosine chứ không dùng Euclidean.

**b) Dữ liệu nhiều chiều và thưa.** Trong không gian hàng nghìn chiều với phần lớn giá trị bằng 0, mọi cặp điểm đều có xu hướng cách xa nhau gần như bằng nhau. Hiện tượng này gọi là *curse of dimensionality*, khiến Euclidean distance mất khả năng phân biệt. Cosine vẫn hoạt động vì nó chỉ quan tâm góc.

**c) Các feature chưa được chuẩn hoá thang đo.** Đây là vấn đề tôi gặp trực tiếp ở Bài 1.3: dataset breast cancer có `worst area` với mean 880,58 trong khi `fractal dimension error` chỉ 0,0038, chênh nhau khoảng 232.000 lần. Với Euclidean distance, feature thang đo lớn sẽ chiếm gần như toàn bộ giá trị khoảng cách và nuốt chửng mọi feature còn lại.

**d) So sánh "khẩu vị" thay vì "mức độ".** Trong hệ thống gợi ý: người dùng X chấm điểm ba phim là [5, 4, 1], người dùng Y chấm [3, 2, 0]. Euclidean distance coi hai người này khác nhau, vì X chấm điểm rộng tay hơn. Cosine nhận ra họ có cùng sở thích, chỉ khác thói quen chấm điểm.

**Khi nào ngược lại, Euclidean tốt hơn?** Khi độ lớn chính là thông tin cần đo:

- Khoảng cách địa lý thật (tìm cửa hàng gần nhất)
- Bài toán vật lý có đơn vị cụ thể
- Dữ liệu đã chuẩn hoá cùng thang đo và giá trị tuyệt đối có ý nghĩa
- K-Means clustering (thuật toán này định nghĩa dựa trên Euclidean)

Nguyên tắc chọn: **hỏi xem "gấp đôi mọi giá trị" có làm thay đổi ý nghĩa của dữ liệu không.** Nếu không đổi ý nghĩa, dùng cosine. Nếu đổi, dùng Euclidean.

### Câu 3: Nếu một vector bằng vector 0 thì cosine similarity có vấn đề gì?

**Vấn đề toán học:** mẫu số bằng 0.

Với $\vec{0} = [0, 0, 0, 0]$ ta có $\|\vec{0}\| = \sqrt{0+0+0+0} = 0$. Khi đó:

$$\cos(A, \vec{0}) = \frac{A \cdot \vec{0}}{\|A\| \cdot \|\vec{0}\|} = \frac{0}{\|A\| \cdot 0} = \frac{0}{0}$$

Đây là dạng vô định. Trong Python, phép chia này ném ra `ZeroDivisionError` và làm dừng chương trình.

**Vấn đề sâu hơn — không chỉ là lỗi kỹ thuật:** cosine similarity đo **góc** giữa hai vector, nhưng vector 0 là một điểm, không có hướng. Góc giữa một hướng và một điểm không tồn tại. Vậy nên `cos(A, 0)` không phải là "khó tính" mà là **không được định nghĩa** về mặt toán học. Không có giá trị nào đúng để trả về.

**Cách xử lý trong code:**

```python
def cosine_similarity(a, b):
    tu_so = dot_product(a, b)
    mau_so = vector_norm(a) * vector_norm(b)
    if mau_so == 0:
        return 0.0
    return tu_so / mau_so
```

Trả về `0.0` là quy ước thực dụng được scikit-learn dùng, mang nghĩa "không có thông tin về sự tương đồng". Nó tránh làm sập chương trình và không tạo ra kết quả sai lệch, vì 0 là điểm số thấp nhất trên thang [0, 1] của dữ liệu không âm — vector rỗng sẽ tự động rơi xuống cuối bảng xếp hạng.

Cần hiểu rõ đây là **quy ước**, không phải đáp án toán học. Giá trị 0 ở đây mang nghĩa "không xác định được", khác hẳn với 0 nghĩa là "hai vector vuông góc" trong trường hợp bình thường.

**Khi nào gặp vector 0 trong thực tế:** trường hợp phổ biến nhất là ở Bài 1.2. Khi người dùng nhập một query mà **không từ nào có trong từ điển** của kho document, `TfidfVectorizer.transform()` trả về vector toàn 0. Nếu không có dòng bảo vệ trên, chương trình sẽ sập ngay khi người dùng gõ sai chính tả hoặc tìm một từ lạ.

Các trường hợp khác: document rỗng, dòng chỉ chứa khoảng trắng, hoặc câu chỉ gồm stopwords đã bị loại bỏ hết.

**Lưu ý về so sánh số thực:** dòng `if mau_so == 0` hoạt động đúng ở đây vì `vector_norm` trả về đúng 0.0 khi mọi phần tử bằng 0. Nhưng trong tình huống mẫu số là kết quả của một chuỗi phép tính phức tạp, giá trị có thể là $10^{-300}$ thay vì đúng 0 — khi đó phép chia không lỗi nhưng cho kết quả tràn số vô nghĩa. Cách viết an toàn hơn là `if mau_so < 1e-10`.
