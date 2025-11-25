import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load your dataset (same folder as this script)
df = pd.read_csv("/Users/gabindavesne/Desktop/cours/M1/S1/Introduction to machine learning/Malis_project/classification task/malware_classification.csv")

# 2. Separate features and target
#    Drop non-numeric / ID columns from X
X = df.drop(columns=["label"])  # remove 'file_id' if you added it
y = df["label"]

# 3. Encode the labels (benign, ransomware, trojan, worm, spyware, adware)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Train / test split (stratify to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)

# 5. Train a classifier (Random Forest is a good default)
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)

clf.fit(X_train, y_train)
# Train accuracy
y_train_pred = clf.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)

# Test accuracy (you already had this)
test_acc = accuracy_score(y_test, y_pred := clf.predict(X_test))

print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")

# Optional: cross-validation on the whole dataset
scores = cross_val_score(clf, X, y_encoded, cv=5)
print("CV accuracies:", scores)
print("Mean CV accuracy:", scores.mean())


if test_acc >= 0.80:
    print("✅ Constraint satisfied: accuracy is at least 80%.")
else:
    print("⚠️ Accuracy is below 80%. You may want to adjust the data generation.")

print("\nClassification report (per class):")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(y_test, y_pred))
