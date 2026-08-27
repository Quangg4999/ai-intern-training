document = []

with open("documents.txt", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            document.append(line)

print("So luong document:", len(document))
print("Document dau tien:", document[0])
print("Document cuoi cung:", document[-1])

#CountVectorizer: Chuyen van ban thanh vector so dem , dem so lan xuat hien cua cac tu trong van ban
#TfidfVectorizer: Chuyen van ban thanh vector tf-idf , tinh toan trong so cua cac tu trong van ban
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(document)

print("Kich thuoc cua ma tran vector:", doc_vectors.shape)
print("So luong tu trong tu dien:", len(vectorizer.get_feature_names_out()))
print("20 tu dau tien trong tu dien:", vectorizer.get_feature_names_out()[:20])

import numpy as np

vector_0 = doc_vectors[0].toarray()[0]
tu_dien = vectorizer.get_feature_names_out()

print("\n--- Vector cua document 0 ---")
print("Cau:", document[0])
print("Do dai vector (L2 norm):", np.linalg.norm(vector_0))

for i in range(len(vector_0)):
    if vector_0[i] > 0:
        print(f"  {tu_dien[i]:15s}: {vector_0[i]: .4f}" )

from sklearn.metrics.pairwise import cosine_similarity

def search(query, top_k=5):
    # Vector hóa query
    query_vector = vectorizer.transform([query])
    # Tinh consine
    scores = cosine_similarity(query_vector, doc_vectors)[0]
    # Sap xep va lay top_k ket qua
    top_indices = np.argsort(scores)[::-1][:top_k]

    for rank, idx in enumerate(top_indices, start=1):
        if scores[idx] == 0:
            nhan = "  <-- khong lien quan (diem = 0)"
        else:
            nhan = ""
        print(f"{rank}. [{scores[idx]:.4f}] (doc {idx}) {document[idx]}{nhan}")

search("malware persistence")
search("powershell suspicious command")
search("web browser")
search("machine learning")
search("registry windows")