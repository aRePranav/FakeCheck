# FakeCheck — Real-Time Fake News Detection System

> **Major Project | Intrainz AI Internship**
> **Author: R Pranav**

---

## What This Project Does

FakeCheck is an end-to-end **fake news detection system** that:
- Accepts any **news headline or article text** as input
- Returns: **FAKE / REAL** verdict + **confidence %** + **word-level explanation**
- Trained on **72,134 labeled articles** (WELFake dataset)
- Compares **4 ML models** and selects the best one automatically
- Supports **batch prediction** from a CSV file
- Fully **explainable** — shows exactly which words drove the decision

---

## Project Structure

```
fakecheck/
│
├── data/
│   └── (place WELFake_Dataset.csv here)
│
├── models/
│   └── (auto-saved after training)
│
├── outputs/
│   └── (auto-generated plots + predictions CSV)
│
├── src/
│   ├── preprocess.py        # Text cleaning + dataset preparation
│   ├── vectorize.py         # TF-IDF feature extraction
│   ├── train_evaluate.py    # Train + evaluate all 4 models
│   ├── fakecheck.py         # Core prediction + explainability class
│   └── visualize.py         # All plots (confusion matrix, model comparison)
│
├── batch_predict.py         # Classify articles from a CSV file
├── main.py                  # Run the full pipeline end-to-end
├── requirements.txt
└── README.md
```

---

## Dataset

**WELFake Dataset** — 72,134 labeled news articles (Real + Fake)

Download from Kaggle:
https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

Place the file as: `data/WELFake_Dataset.csv`

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/fakecheck.git
cd fakecheck

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data (one-time)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 4. Place dataset in data/ folder

# 5. Run full pipeline (training + evaluation + demo)
python main.py

# 6. Predict a single article
python main.py --predict "NASA confirms water ice on the Moon near polar craters"

# 7. Batch predict from CSV
python batch_predict.py --input data/articles.csv --text_col text --output outputs/predictions.csv
```

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **Passive Aggressive (BEST)** | **96.2%** | **96.5%** | **95.9%** | **96.2%** | **0.989** |
| Linear SVC | 95.8% | 96.0% | 95.5% | 95.8% | 0.985 |
| Logistic Regression | 94.7% | 94.9% | 94.4% | 94.6% | 0.979 |
| Naive Bayes | 92.1% | 91.8% | 92.5% | 92.1% | 0.969 |

---

## Sample Output

```
==============================
  FAKECHECK RESULT
==============================
  Input      : SHOCKING: Government secretly poisoning water supply!
  Verdict    : FAKE
  Confidence : 97.8%

  Top words pushing toward FAKE:
    + shocking          weight: 0.5312
    + secret            weight: 0.4891
    + poison            weight: 0.4203

  Top words pushing toward REAL:
    - water             weight: 0.0812
==============================
```

---

## Tech Stack

- Python 3.8+
- scikit-learn, pandas, numpy
- NLTK
- matplotlib, seaborn
- joblib

---

## Author

**R Pranav**
- Email: pranavchinnu100@gmail.com
- LinkedIn: https://www.linkedin.com/in/pranavr25
