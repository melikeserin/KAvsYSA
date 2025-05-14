import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv("adult.data.txt", header=None, names=column_names, na_values=' ?')

df.dropna(inplace=True)

label_encoders = {}
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df.drop("income", axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Karar Ağacı Eğitimi ===
start_dt = time.time()
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
end_dt = time.time()
dt_duration = end_dt - start_dt
y_pred_dt = dt.predict(X_test)

# === Yapay Sinir Ağı Eğitimi ===
start_mlp = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train_scaled, y_train)
end_mlp = time.time()
mlp_duration = end_mlp - start_mlp
y_pred_mlp = mlp.predict(X_test_scaled)

# Çıktı
print("=== Karar Ağacı (Decision Tree) ===")
print(classification_report(y_test, y_pred_dt))
print("=== Yapay Sinir Ağı (MLPClassifier) ===")
print(classification_report(y_test, y_pred_mlp))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# === Eğitim Sürelerini Yazdır ===
print(f"Karar Ağacı eğitim süresi: {dt_duration:.4f} saniye")
print(f"Yapay Sinir Ağı eğitim süresi: {mlp_duration:.4f} saniye")

# Karar Ağacı karışıklık matrisi
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Karar Ağacı Karışıklık Matrisi")
axes[0].set_xlabel("Tahmin")
axes[0].set_ylabel("Gerçek")

# Yapay Sinir Ağı karışıklık matrisi
sns.heatmap(confusion_matrix(y_test, y_pred_mlp), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title("Yapay Sinir Ağı Karışıklık Matrisi")
axes[1].set_xlabel("Tahmin")
axes[1].set_ylabel("Gerçek")

plt.tight_layout()
plt.show()





