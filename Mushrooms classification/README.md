# 🍄 Identifying Poisonous Mushrooms Based on Physical Characteristics

A machine learning classification project analyzing physical and morphological features of mushrooms to predict whether they are edible or poisonous. Developed as part of a data science bootcamp portfolio.

---

## 📋 Project Overview
Accurately identifying edible versus poisonous mushrooms is a critical safety challenge. This project applies Exploratory Data Analysis (EDA) and Supervised Machine Learning models to the classic Mushroom Dataset containing **8,124 total samples**. 

By evaluating feature correlations and training three different classifiers, this pipeline identifies the most reliable predictors of toxicity and achieves **100% classification accuracy** using ensemble learning.

## ⚙️ Methodology & Technical Workflow
1. **Data Preprocessing:** Handled categorical encoding of all physical features to prepare strings for numerical machine learning models.
2. **Exploratory Data Analysis (EDA):** Used `Pandas`, `Matplotlib`, and `Seaborn` to generate correlation matrices, distribution plots, and cross-tabulations analyzing gill structures, cap shapes, and bruising indicators.
3. **Model Training & Evaluation:** Implemented an 80/20 train-test split using `Scikit-Learn` to train and benchmark three distinct supervised classifiers:
   * **Logistic Regression** (baseline linear boundary)
   * **Support Vector Machine (SVM)** (kernel-based separation)
   * **Random Forest Classifier** (ensemble decision trees)
4. **Performance Validation:** Evaluated models using confusion matrices and accuracy metrics to prevent overfitting and ensure reliable generalisation.

---

## 📊 Dataset Summary
* **Total Samples:** 8,124 instances
* **Class Distribution:** 
  * 🟢 **Edible:** 4,208 samples (51.8%)
  * 🔴 **Poisonous:** 3,916 samples (48.2%)
* **Data Characteristics:** A well-balanced dataset with entirely categorical features capturing gill size, gill color, cap properties, bruising, and odors.

---

## 🔍 Key Exploratory Data Analysis (EDA) Insights
* **Gill Color & Size:** Gills have a massive impact on toxicity. Notably, **100% of the 1,728 mushrooms with buff-colored gills** in the dataset turned out to be poisonous. Narrow gill sizes also lean heavily toward toxicity.
* **Bruising:** Most poisonous mushrooms exhibited bruising. If a mushroom was bruised, it carried a **69% probability** of being poisonous.

---

## 🤖 Machine Learning Modeling & Results
The models were evaluated using an **80/20 train-test split**:

| Model | Accuracy Score | Performance Summary |
| :--- | :--- | :--- |
| **Logistic Regression** | **94.40%** | Solid baseline performance with minor false positives/negatives. |
| **Support Vector Machine (SVM)** | **95.75%** | Improved boundary separation; reduced errors compared to baseline. |
| **Random Forest Classifier** | **100.0%** | Flawless classification utilizing ensemble decision trees to capture complex non-linear feature interactions. |

### Conclusion
Because all variables are categorical and represent a finite set of distinct mushroom profiles without significant outliers, the **Random Forest Classifier** successfully mapped all patterns to achieve **100% accuracy**.

*(And remember: next time you're in a Random Forest, you can be certain you won't be poisoned!)*

---

## 📂 Repository Contents
* `Presentation1.pptx`: The slide deck summarizing the high-level insights and visual findings.
* Jupyter Notebook / Python code files containing the complete data pipeline, EDA plots, and model training scripts.

## 👤 Author
**Amjad Jouria**  
*MSc Data Science Student | University of Sheffield*  
[GitHub Profile](https://github.com/your-username)
