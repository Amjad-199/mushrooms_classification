"""
Mushroom Classification Machine Learning Pipeline
Author: Amjad Jouria
Description: End-to-end data pipeline, Exploratory Data Analysis (EDA), 
and Supervised Machine Learning model comparison (Logistic Regression, SVM, Random Forest)
on the Mushroom Dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================

# Load the dataset (Make sure 'mushrooms.csv' is in the same directory)
df = pd.read_csv("mushrooms.csv")

# Drop columns with missing/redundant values
df.drop(columns=["veil-type", "stalk-root"], inplace=True)

# Encode categorical variables into numerical values
label_encoders = {}
for col in df.columns:
  le = LabelEncoder()
  df[col] = le.fit_transform(df[col])
  label_encoders[col] = le

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS
# ==========================================

# 2.1 Edible vs Poisonous Distribution Chart
class_counts = df.iloc[:, 0].value_counts()
total = class_counts.sum()
percentages = (class_counts / total) * 100

plt.figure(figsize=(6, 4))
bars = plt.bar(class_counts.index, class_counts.values, color=["green", "red"])

for bar, count, percent in zip(bars, class_counts.values, percentages):
  plt.text(
      bar.get_x() + bar.get_width() / 2,
      bar.get_height() + 50,
      f"{count} ({percent:.1f}%)",
      ha="center",
      va="bottom",
      fontsize=12,
      fontweight="bold",
  )

plt.xlabel("Mushroom Type")
plt.ylabel("Count")
plt.title("Number of Edible vs. Poisonous Mushrooms")
plt.xticks(ticks=[0, 1], labels=["Edible", "Poisonous"], rotation=0)
plt.ylim(0, class_counts.max() * 1.1)
plt.show()

# 2.2 Feature Correlation Matrix Bar Chart
df_encoded = df.apply(lambda col: col.astype("category").cat.codes)
corr_matrix = df_encoded.corr()
class_corr = corr_matrix.iloc[0, 1:].sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=class_corr.values, y=class_corr.index, palette="coolwarm")
plt.xlabel("Correlation with Class (Edible/Poisonous)")
plt.ylabel("Features")
plt.title("Feature Correlation with Mushroom Class")
plt.axvline(0, color="black", linewidth=0.8)
plt.show()

# 2.3 Gill Color vs Class Countplot
plt.figure(figsize=(10, 6))
sns.countplot(x="gill-color", hue="class", data=df)
plt.title("Gill Color vs Class (Edible/Poisonous)", fontsize=16)
plt.xlabel("Gill Color", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()

# 2.4 Bruises vs Class Barplot
count_data = df.groupby(["bruises", "class"]).size().reset_index(name="count")
count_data = count_data[count_data["count"] > 0]
valid_bruises = count_data["bruises"].unique()

plt.figure(figsize=(10, 6))
ax = sns.barplot(
    x="bruises", hue="class", y="count", data=count_data, errorbar=None
)

if 0 not in valid_bruises:
  ax.set_xticks([1])
  ax.set_xticklabels(["Yes"])

plt.xticks(ticks=[0, 1], labels=["Yes", "No"])
plt.title("Bruises vs Class (Edible/Poisonous)", fontsize=16)
plt.xlabel("Bruises", fontsize=12)
plt.ylabel("Count", fontsize=12)

for p in ax.patches:
  height = p.get_height()
  if height > 0:
    x = p.get_x() + p.get_width() / 2
    y = height
    ax.text(x, y + 20, f"{int(height)}", ha="center", va="bottom", fontsize=10)

plt.show()

# 2.5 Ring-Type Distribution by Class
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="ring-type", hue="class", palette="Set2")
plt.title("Distribution of Ring-Type by Mushroom Class")
plt.xlabel("Ring-Type")
plt.ylabel("Count")
plt.show()

# ==========================================
# 3. MACHINE LEARNING MODELING
# ==========================================

# Define Features and Target
X = df.drop(columns=["class"])
y = df["class"]

# Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 3.1 Logistic Regression & SVM ---
lr_model = LogisticRegression()
svm_model = SVC(kernel="linear", random_state=42)

lr_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)

# Evaluate Logistic Regression
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr)*100:.2f}%")
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

# Evaluate SVM
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm)*100:.2f}%")
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Confusion Matrices for LR and SVM
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm_svm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("SVM Confusion Matrix")
plt.show()

# --- 3.2 Random Forest Classifier ---
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

print(f"\nRandom Forest Accuracy: {rf_accuracy * 100:.2f}%")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix for Random Forest
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Random Forest Confusion Matrix")
plt.show()