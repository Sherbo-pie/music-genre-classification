import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------
# LOAD CSV
# ------------------------------------
csv_path = r"C:\Users\Shirsha Pattanaik\gtzan_features.csv"
df = pd.read_csv(csv_path)

print("Dataset Loaded Successfully!")
print(df.head())

# ------------------------------------
# SEPARATE FEATURES & LABELS
# ------------------------------------
X = df.drop("genre", axis=1)
y = df["genre"]

# ------------------------------------
# TRAIN / TEST SPLIT (80/20)
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------
# STANDARDIZE FEATURES
# ------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------
# 1. KNN
# ------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

# ------------------------------------
# 2. SVM (RBF kernel)
# ------------------------------------
svm = SVC(kernel="rbf", gamma="scale", C=10)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

# ------------------------------------
# 3. ANN (MLPClassifier)
# ------------------------------------
ann = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
ann.fit(X_train, y_train)
ann_pred = ann.predict(X_test)
ann_acc = accuracy_score(y_test, ann_pred)

# ------------------------------------
# 4. Gradient Boosting
# ------------------------------------
gbt = GradientBoostingClassifier()
gbt.fit(X_train, y_train)
gbt_pred = gbt.predict(X_test)
gbt_acc = accuracy_score(y_test, gbt_pred)

# ------------------------------------
# PRINT ACCURACIES
# ------------------------------------
print("\n=== MODEL ACCURACIES ===")
print(f"KNN Accuracy: {knn_acc * 100:.2f}%")
print(f"SVM Accuracy: {svm_acc * 100:.2f}%")
print(f"ANN Accuracy: {ann_acc * 100:.2f}%")
print(f"Gradient Boosting Accuracy: {gbt_acc * 100:.2f}%")

# ------------------------------------
# PRINT CONFUSION MATRICES
# ------------------------------------
print("\n=== CONFUSION MATRIX: SVM ===")
print(confusion_matrix(y_test, svm_pred))

print("\n=== CONFUSION MATRIX: GBT ===")
print(confusion_matrix(y_test, gbt_pred))

print("CONFUSION MATRIX KNN")
print(confusion_matrix(y_test, knn_pred))

print("CONFUSION MATRIX ANN")
print(confusion_matrix(y_test, ann_pred))

# ------------------------------------
# PRINT CLASSIFICATION REPORT
# ------------------------------------
print("\n=== CLASSIFICATION REPORT: BEST MODEL ===")
if svm_acc >= max(knn_acc, ann_acc, gbt_acc):
    print("Best Model: SVM\n")
    print(classification_report(y_test, svm_pred))
elif gbt_acc >= max(knn_acc, svm_acc, ann_acc):
    print("Best Model: Gradient Boosting\n")
    print(classification_report(y_test, gbt_pred))
elif ann_acc >= max(knn_acc, svm_acc, gbt_acc):
    print("Best Model: ANN\n")
    print(classification_report(y_test, ann_pred))
else:
    print("Best Model: KNN\n")
    print(classification_report(y_test, knn_pred))
