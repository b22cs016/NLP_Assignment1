# Sports vs. Politics Classifier 🏈 🏛️

This project implements a binary text classifier to distinguish between **Sports** and **Politics** articles using Machine Learning.

## 📂 Project Structure
- `RollNumber_prob4.py`: The Python script for training and evaluation.
- `Report.pdf`: Detailed 5-page analysis of the models and methodology.
- `model_comparison.png`: Visual performance comparison of the three algorithms.

## 📊 Dataset Description
We utilized the **20 Newsgroups Dataset**, a standard NLP benchmark.
- **Total Documents:** 4,618
- **Categories:**
    - **Sports (1,993 docs):** `rec.sport.hockey`, `rec.sport.baseball`
    - **Politics (2,625 docs):** `talk.politics.misc`, `talk.politics.guns`, `talk.politics.mideast`
- **Preprocessing:** Removed headers, footers, and quotes to ensure unbiased training.

## 🛠️ Methodology
1. **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency) with a vocabulary limit of 5,000 words.
2. **Models Evaluated:**
    - Multinomial Naive Bayes
    - Logistic Regression
    - Linear SVM

## 📈 Experimental Results
Contrary to initial expectations, **Naive Bayes** slightly outperformed the other models.

| Model | Accuracy | F1-Score (Weighted) |
|-------|----------|---------------------|
| **Naive Bayes** | **95.67%** | **0.96** |
| Logistic Regression | 95.35% | 0.95 |
| Linear SVM | 94.70% | 0.95 |

### Key Observations
- **Naive Bayes** achieved the highest accuracy, proving that for distinct vocabularies (like "touchdown" vs "legislation"), simple probabilistic models are highly effective.
- **Politics Class:** The models were extremely good at identifying Politics articles (Recall ~99%).
- **Sports Class:** Slightly lower recall (90-92%), indicating some sports articles were misclassified as politics.

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install scikit-learn pandas matplotlib seaborn
