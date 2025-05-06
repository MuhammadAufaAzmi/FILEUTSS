# FILEUTSS
Proyek ini merupakan implementasi pemrosesan data hasil serangan DDoS dari tiga sumber berbeda, yang kemudian digabungkan menjadi satu DataFrame. Dataset gabungan digunakan untuk pelatihan model klasifikasi menggunakan algoritma Decision Tree. Proyek ini mencakup proses preprocessing data, pemisahan fitur dan label, pelatihan model, evaluasi akurasi, serta visualisasi model pohon keputusan dan confusion matrix.

Import Library
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as lol
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
Library yang digunakan:
-pandas dan numpy: manipulasi data
-matplotlib dan seaborn: visualisasi data
-scikit-learn: pembelajaran mesin dan evaluasi model

Pembacaan Data CSV
```python
dataset1 = pd.read_csv("DDoS ICMP Flood.csv")
dataset2 = pd.read_csv("DDoS UDP Flood.csv")
dataset3 = pd.read_csv("DoS ICMP Flood.csv")
```
Tiga file CSV yang berisi data serangan:
-DDoS ICMP Flood
-DDoS UDP Flood
-DoS ICMP Flood

Penggabungan DataFrame
```python
hasilgabung = pd.concat([dataset1, dataset2, dataset3], ignore_index=True)
```
Data dari ketiga sumber digabung menjadi satu DataFrame untuk pemrosesan selanjutnya.

Pemisahan Fitur dan Label
```python
x = hasilgabung.iloc[:, 7:76]
y = hasilgabung['Label']
```
-Fitur (X): Kolom ke-7 hingga ke-75
-Label (Y): Kolom 'Label', yang berisi target klasifikasi (jenis serangan)

Pembagian Data Latih dan Uji
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
Data dibagi menjadi:
-80% data latih
-20% data uji

Pelatihan Model Decision Tree
```python
alya = DecisionTreeClassifier(criterion='entropy', splitter='random')
alya.fit(x_train, y_train)
y_pred = alya.predict(x_test)
```
Model Decision Tree dibuat dan dilatih dengan parameter:
-criterion='entropy': menggunakan entropi untuk mengukur ketidakmurnian
-splitter='random': memilih pemisahan secara acak

Evaluasi Akurasi
```python
accuracy = accuracy_score(y_test, y_pred)
```
Menghitung akurasi model dengan membandingkan hasil prediksi dan data aktual.

Visualisasi Pohon Keputusan
```python
fig = plt.figure(figsize=(10, 7))
tree.plot_tree(alya, feature_names=x.columns.values, class_names=np.array(['Benign Traffic','DDos ICMP Flood','DDoS UDP Flood']), filled=True)
plt.show()
```
Struktur pohon keputusan divisualisasikan, menunjukkan bagaimana fitur digunakan untuk memisahkan kelas serangan.


Confusion Matrix
```python
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
label = np.array(['MQTT Malformed','Recon OS Scan','Recon Ping Sweep'])
lol.heatmap(conf_matrix, annot=True, xticklabels=label, yticklabels=label)
plt.xlabel('Prediksi')
plt.ylabel('Fakta')
plt.show()
```
Menampilkan matriks kebingungan dalam bentuk heatmap, yang menggambarkan kinerja klasifikasi model terhadap masing-masing kelas. Namun, label yang digunakan (MQTT Malformed, Recon OS Scan, dll.) tidak sesuai dengan isi dataset sebenarnya dan perlu diperbaiki.

KESIMPULAN
Script ini membentuk sebuah pipeline analitik lengkap untuk klasifikasi serangan jaringan berdasarkan data DDoS dan DoS. Dimulai dari pembacaan dan penggabungan data, pemrosesan fitur, pelatihan model Decision Tree, hingga visualisasi hasil evaluasi. Model mampu mengklasifikasi jenis serangan jaringan secara otomatis berdasarkan fitur statistik yang tersedia. Visualisasi pohon keputusan dan confusion matrix memberikan gambaran yang berguna dalam mengevaluasi performa dan pemahaman terhadap pengambilan keputusan oleh model.
