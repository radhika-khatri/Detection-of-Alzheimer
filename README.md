# 🧠 Detection of Alzheimer's Disease

Alzheimer's is a progressive neurological disorder, and catching it early is one of the best ways to slow its progression. This project uses machine learning to classify patients based on clinical and cognitive data, comparing a wide range of algorithms to see which ones are most effective for this kind of problem.

Unlike the standard approach of picking one model and calling it a day, this repo implements ten different models so you can study how each one handles the same data and compare their strengths.

---

## 🤖 Models Implemented

Each model lives in its own Python file:

| File | Model |
|---|---|
| `DecisionTreeClassifier.py` | Decision Tree |
| `KNeighborsClassifier.py` | K-Nearest Neighbors |
| `LogisticRegression.py` | Logistic Regression |
| `NaiveBayes.py` | Naive Bayes |
| `RandomForest.py` | Random Forest |
| `GradientBoostingClassifier.py` | Gradient Boosting |
| `SVM.py` | Support Vector Machine |
| `GaussianMixture.py` | Gaussian Mixture Model |
| `Autoencoder.py` | Autoencoder (Neural Network) |
| `Hybrid.py` | Hybrid Model (combination approach) |

The `Hybrid.py` is particularly interesting since it combines multiple models to try to get better results than any single algorithm on its own.

---

## 🗃️ Dataset

The project uses the **OASIS (Open Access Series of Imaging Studies)** dataset or a similar Alzheimer's clinical dataset containing features like:

- Age, gender, education level
- Socioeconomic status
- MMSE score (Mini-Mental State Examination)
- CDR (Clinical Dementia Rating)
- Brain volume measurements (eTIV, nWBV, ASF)

The target variable is whether a patient is classified as **Demented**, **Non-Demented**, or **Converted**.

---

## 📁 Repo Structure

```
Detection-of-Alzheimer/
├── Autoencoder.py
├── DecisionTreeClassifier.py
├── GaussianMixture.py
├── GradientBoostingClassifier.py
├── Hybrid.py
├── KNeighborsClassifier.py
├── LogisticRegression.py
├── NaiveBayes.py
├── RandomForest.py
└── SVM.py
```

---

## ⚙️ Requirements

Install the dependencies with:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow
```

> TensorFlow is needed for `Autoencoder.py`. The rest only require scikit-learn.

---

## 🚀 How to Run

### Clone the repo

```bash
git clone https://github.com/radhika-khatri/Detection-of-Alzheimer.git
cd Detection-of-Alzheimer
```

### Run any model

```bash
python LogisticRegression.py
python RandomForest.py
python Autoencoder.py
```

Each script will load the dataset, train the model, and print evaluation metrics like accuracy, confusion matrix, and classification report.

### Run in Google Colab

If you prefer Colab, upload any `.py` file to a notebook cell and run it. Or paste the code into a Colab cell directly. Make sure to upload the dataset file to the Colab environment first.

---

## 📊 What Each Script Does

All scripts generally follow the same structure:

1. Load and explore the dataset
2. Handle missing values and preprocess features
3. Encode categorical variables
4. Split into train/test sets
5. Train the model
6. Evaluate with accuracy, confusion matrix, precision, recall, and F1-score
7. Visualize results where applicable

The `Autoencoder.py` goes further by using an unsupervised neural network to learn compressed feature representations, then using those for classification. The `Hybrid.py` combines predictions from multiple models to arrive at a final answer.

---

## ⚠️ Note

This project is for educational and research purposes only. It should not be used as a diagnostic tool in any medical setting.

---

## 👤 Author

**Radhika Khatri**  
[GitHub Profile](https://github.com/radhika-khatri)
