# Báo cáo Bài 1.2 — Mini Search Engine bằng TF-IDF

## 1. Mục tiêu

Xây một máy tìm kiếm nhỏ trên kho 30 câu tự tạo. Người dùng nhập một truy vấn (query), hệ thống trả về 5 document giống nhất kèm điểm số, xếp giảm dần. Toàn bộ dựa trên hai bước: biểu diễn văn bản bằng vector TF-IDF, rồi đo độ tương đồng bằng cosine similarity. Không dùng bất kỳ API LLM hay mô hình embedding nào.

## 2. Phương pháp

**Vector hóa — TF-IDF.** Mỗi câu được biến thành một vector, mỗi chiều ứng với một từ trong từ điển của kho. Giá trị mỗi chiều là `tf × idf`:

- `tf(t, d)` = số lần từ `t` xuất hiện trong document `d`.
- `idf(t) = ln((1 + N) / (1 + df(t))) + 1`, với `N` là tổng số document và `df(t)` là số document chứa từ `t` (công thức mặc định của scikit-learn khi `smooth_idf=True`).

Ý nghĩa của IDF: từ càng hiếm trong kho thì trọng số càng cao; từ phổ biến (như `the`, `an`, `from`) bị dìm xuống gần 1. Nhờ đó những từ đặc trưng (`malware`, `powershell`, `registry`) mới có tiếng nói trong phép so sánh.

**Chuẩn hóa L2.** Mỗi vector được chia cho độ dài của chính nó, đưa mọi vector về độ dài 1. Hệ quả quan trọng: cosine similarity rút gọn thành tích vô hướng, và câu dài không còn lợi thế điểm số chỉ vì có nhiều từ.

**Đo tương đồng — cosine similarity.** Query cũng được vector hóa bằng đúng bộ IDF đã học từ kho (gọi `transform`, không phải `fit_transform`), rồi tính cosine với cả 30 document trong một phép nhân ma trận.

**Công cụ:** Python 3.14, scikit-learn (`TfidfVectorizer`, `cosine_similarity`), numpy.

## 3. Dữ liệu

Kho gồm 30 câu tiếng Anh, chia ba nhóm để có thể đánh giá định lượng:

- **doc 0–9:** Windows / security / malware (nhóm mục tiêu).
- **doc 10–19:** công nghệ nhưng không phải security (nhiễu gần — dùng chung một số từ với nhóm security như `download`, `file`, `network`).
- **doc 20–29:** chủ đề bất kỳ (nhiễu xa — cà phê, thời tiết, thể thao...).

Thứ tự này không ảnh hưởng đến kết quả thuật toán (TF-IDF coi mỗi câu là một túi từ độc lập), nhưng giúp việc chấm điểm trực quan: nếu một document nhóm 20–29 lọt vào Top của một query security, đó là dấu hiệu hệ thống trả sai.

Kho được thiết kế có sẵn ba **cặp bẫy** để phục vụ phần phân tích:

- doc 1 (`malware ... persistence`) và doc 2 (`trojan ... startup folder`) — cùng nghĩa, không chung từ.
- doc 12 (`machine learning model`) và doc 13 (`neural networks ... training`) — cùng chủ đề AI, không chung từ.
- doc 10 (`web page`) và doc 11 (`browser`) — hai mặt của khái niệm trình duyệt web.

Ma trận TF-IDF thu được có kích thước **(30, 190)**: 30 document, 190 từ khác nhau. Mỗi câu chỉ chứa khoảng 8–10 từ, nên hơn 95% ô của ma trận bằng 0 — đây là vector **thưa (sparse)**, một đặc điểm sẽ được bàn ở mục 5.

## 4. Kết quả — 5 query × Top 5

Nhóm document: **S** = security (0–9), **T** = tech (10–19), **M** = misc (20–29).

### Query 1: `malware persistence`

| Hạng | Score | Doc | Câu | Nhóm |
|---|---|---|---|---|
| 1 | 0.5509 | 1 | Malware created a Run registry key for persistence | S |
| 2 | 0.0000 | 29 | The old library closes on Sunday afternoons | M |
| 3 | 0.0000 | 27 | Fresh vegetables are cheaper at the local market | M |
| 4 | 0.0000 | 26 | The train to the mountains leaves at six in the morning | M |
| 5 | 0.0000 | 25 | He practices guitar for an hour before bed | M |

### Query 2: `powershell suspicious command`

| Hạng | Score | Doc | Câu | Nhóm |
|---|---|---|---|---|
| 1 | 0.4255 | 0 | PowerShell executed an encoded command from a remote server | S |
| 2 | 0.2180 | 6 | A suspicious process connected to an unknown IP address | S |
| 3 | 0.0000 | 27 | Fresh vegetables are cheaper at the local market | M |
| 4 | 0.0000 | 26 | The train to the mountains leaves at six in the morning | M |
| 5 | 0.0000 | 29 | The old library closes on Sunday afternoons | M |

### Query 3: `web browser`

| Hạng | Score | Doc | Câu | Nhóm |
|---|---|---|---|---|
| 1 | 0.2656 | 10 | Chrome opened a normal web page in a new tab | T |
| 2 | 0.2570 | 11 | The browser cached images to make pages load faster | T |
| 3 | 0.0000 | 27 | Fresh vegetables are cheaper at the local market | M |
| 4 | 0.0000 | 26 | The train to the mountains leaves at six in the morning | M |
| 5 | 0.0000 | 29 | The old library closes on Sunday afternoons | M |

### Query 4: `machine learning`

| Hạng | Score | Doc | Câu | Nhóm |
|---|---|---|---|---|
| 1 | 0.5321 | 12 | The machine learning model was trained on a large dataset | T |
| 2 | 0.0000 | 29 | The old library closes on Sunday afternoons | M |
| 3 | 0.0000 | 27 | Fresh vegetables are cheaper at the local market | M |
| 4 | 0.0000 | 26 | The train to the mountains leaves at six in the morning | M |
| 5 | 0.0000 | 25 | He practices guitar for an hour before bed | M |

### Query 5: `registry windows`

| Hạng | Score | Doc | Câu | Nhóm |
|---|---|---|---|---|
| 1 | 0.2771 | 3 | Windows Defender scanned a downloaded file and found a threat | S |
| 2 | 0.2311 | 1 | Malware created a Run registry key for persistence | S |
| 3 | 0.2186 | 7 | The attacker modified registry values to disable security tools | S |
| 4 | 0.0000 | 27 | Fresh vegetables are cheaper at the local market | M |
| 5 | 0.0000 | 29 | The old library closes on Sunday afternoons | M |

## 5. Nhận xét: trường hợp tốt và chưa tốt

### 5.1. Trường hợp search TỐT

**`registry windows` là ví dụ đẹp nhất.** Cả ba document có điểm khác 0 đều thuộc nhóm security (doc 3, 1, 7), không có nhiễu lọt vào. Lý do: từ khóa `registry` và `windows` là những từ hiếm nhưng xuất hiện ở **nhiều** document trong kho, nên hệ thống có đủ dữ liệu để gom đúng một cụm câu liên quan. Đây là điều kiện lý tưởng của TF-IDF: query trùng từ vựng với nhiều document cùng chủ đề.

**`web browser` cũng thành công.** Hai kết quả đầu (doc 10 về web page, doc 11 về browser) đều đúng chủ đề trình duyệt web, điểm sát nhau (0.2656 so với 0.2570). Người dùng hài lòng với cả hai. Chênh lệch nhỏ đến từ độ dài câu: doc 10 ngắn hơn một từ nên từ trùng `web` bị pha loãng ít hơn `browser` trong doc 11.

*(Chỗ này bạn có thể thêm nhận xét riêng: bạn thấy kết quả tốt vì lý do gì khác không?)*

### 5.2. Trường hợp search CHƯA TỐT

**`malware persistence` bỏ sót document cùng nghĩa.** Chỉ doc 1 có điểm; bốn dòng còn lại đều bằng 0 và thực chất là "Top 5 giả" — rác được sắp xếp ngẫu nhiên. Điều đáng nói nhất: **doc 2 ("The trojan added itself to the startup folder") cũng nhận 0 điểm.** Câu này mô tả chính xác một hành vi malware persistence, nhưng vì không chung một từ nào với query (`trojan` ≠ `malware`, `startup folder` ≠ `persistence`), hệ thống coi nó không liên quan — ngang với câu về thư viện đóng cửa Chủ nhật.

**`machine learning` lặp lại đúng vấn đề đó.** Chỉ doc 12 có điểm. Doc 13 ("Neural networks require millions of labeled training examples") — cùng chủ đề AI hoàn toàn — nhận 0 điểm vì không trùng chữ `machine` hay `learning`.

**Mẫu hình chung:** query nào có từ khóa chỉ nằm ở đúng một document thì hệ thống gãy (Top 5 giả); query nào có từ khóa trải trên nhiều document thì làm tốt. TF-IDF sống nhờ sự trùng lặp từ vựng trong kho.

**Một quan sát về điểm số:** không query nào đạt tới 0.6, kể cả khi tìm đúng. Nguyên nhân là query chỉ 2–3 từ trong khi mỗi document 8–10 từ, nên phần chồng lấn từ vựng luôn nhỏ. Đây là đặc tính cố hữu của phương pháp match-từ trên câu ngắn.

**Vài hạn chế kỹ thuật quan sát được:**

- Hệ thống không cắt gốc từ (stemming): `opens` (doc 21) và `opened` (doc 11) bị coi là hai từ khác nhau. `TfidfVectorizer` mặc định không xử lý biến thể hình thái.
- Các từ một ký tự bị loại: chữ `a` trong "a Run registry key" biến mất khỏi từ điển, nên không tham gia tính điểm.
- IDF chưa dìm được từ phổ biến xuống đủ thấp vì kho quá nhỏ: trong ma trận, từ nối `an` vẫn có trọng số 0.2782 so với `powershell` 0.3685 — chỉ kém 25%. Với kho hàng triệu document, `an` sẽ có IDF sát 1 và gần như vô nghĩa.

## 6. Hệ thống này khác semantic embedding search hiện đại ở đâu?

Đây là khác biệt cốt lõi và cũng là bài học lớn nhất của bài tập.

**TF-IDF khớp CHỮ. Embedding hiện đại khớp NGHĨA.**

TF-IDF chỉ biết hai từ giống hệt nhau về mặt ký tự hay không. Nó không có khái niệm "gần nghĩa": với nó, `trojan` và `malware` xa lạ như `trojan` và `guitar`. Đó là lý do doc 2 và doc 13 trong kho nhận 0 điểm dù đúng chủ đề — chúng bị bỏ sót không phải vì thuật toán lỗi, mà vì bản chất của phương pháp là so khớp từ vựng bề mặt.

Semantic embedding search khắc phục đúng điểm này. Mô hình embedding được huấn luyện trên hàng tỷ câu, học được rằng `trojan`, `malware`, `ransomware` thường xuất hiện trong cùng ngữ cảnh, nên đặt chúng gần nhau trong không gian vector. Khi đó query `malware persistence` sẽ tìm ra doc 2 dễ dàng, vì nó so **ý nghĩa** chứ không so **mặt chữ**.

Ba khác biệt kỹ thuật cụ thể:

| Khía cạnh | TF-IDF (bài này) | Semantic embedding |
|---|---|---|
| Cơ sở so khớp | Trùng từ chính xác | Gần nghĩa trong không gian ngữ nghĩa |
| Kiểu vector | Thưa (sparse), 190 chiều, >95% bằng 0 | Dày (dense), 384–768 chiều, không chiều nào bằng 0 |
| Nguồn gốc vector | Đếm từ + IDF, không cần huấn luyện | Học từ hàng tỷ câu qua mạng nơ-ron |
| Xử lý đồng nghĩa | Không (trojan ≠ malware) | Có (trojan ≈ malware) |
| Xử lý đa nghĩa/ngữ cảnh | Không | Có (một từ đổi nghĩa theo ngữ cảnh) |

**Điểm mạnh còn lại của TF-IDF:** nhanh, nhẹ, không cần GPU, không cần huấn luyện, và **giải thích được** — ta biết chính xác vì sao một document được điểm cao (do trùng từ nào). Embedding mạnh hơn về ngữ nghĩa nhưng là hộp đen, tốn tài nguyên hơn, và khó truy vết lý do xếp hạng. Trong thực tế, nhiều hệ thống tìm kiếm hiện đại kết hợp cả hai (hybrid search) để tận dụng ưu điểm của mỗi bên.

## 7. Kết luận

Hệ thống hoàn thành đủ yêu cầu: vector hóa 30 câu bằng TF-IDF, nhận query, tính cosine similarity, trả Top 5 xếp giảm dần, test trên 5 query. Qua thực nghiệm, hệ thống làm tốt khi query trùng từ vựng với nhiều document (`registry windows`), và gãy khi document liên quan dùng từ đồng nghĩa thay vì từ trùng khớp (`malware persistence`, `machine learning`). Chính điểm gãy này minh họa rõ ràng ranh giới giữa tìm kiếm theo từ khóa và tìm kiếm ngữ nghĩa hiện đại.
