# 📰 Fake News Detection (Machine Learning Project)

This project detects whether a given news article is **real or fake** using **Natural Language Processing (NLP)** and **machine learning classification models**.  
It leverages text processing techniques and supervised learning to identify misinformation effectively.

---

## 📂 Dataset

The dataset consists of two CSV files:

- **`True.csv`** → Contains real news articles  
- **`Fake.csv`** → Contains fake news articles  

Each file has the following columns:

- `title` → News article title  
- `text` → Main article content  
- `subject` → News category (e.g., politics, world, tech)  
- `date` → Published date  
- `content` → Combined text content  

Download sample dataset:  
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

---

## 🛠️ Tech Stack

- Python 3  
- Pandas, NumPy → Data handling  
- Matplotlib, Seaborn → Data visualization  
- Scikit-learn → Machine learning models  
- TfidfVectorizer → Text vectorization (NLP)  

---

## ⚡ Project Workflow

1. **Import Libraries and Load Dataset**  
2. **Combine True and Fake Data** with a `label` column (1 = Fake, 0 = Real)  
3. **Data Cleaning**  
   - Handle missing values  
   - Merge `title` and `text` into a single content column  
4. **Exploratory Data Analysis (EDA)**  
   - Check class distribution  
   - Visualize article length distributions  
5. **Text Preprocessing**  
   - Lowercasing, removing punctuation and stopwords  
   - Convert text to numerical vectors using **TF-IDF**  
6. **Train-Test Split**  
7. **Model Training & Evaluation**  
   - Logistic Regression  
   - Naive Bayes  
   - Random Forest  
   - Evaluate with **Accuracy, Precision, Recall, F1-score**  
8. **Prediction on New News Articles**

---

## 📊 Example Output

```
Logistic Regression Accuracy: 0.986
Naive Bayes Accuracy: 0.972
Confusion Matrix:
[[2050 30]
[ 25 1975]]
```


---

## 🧪 Predicting a New Article

```python
sample_article = ["Breaking: Scientists discover new exoplanet that could support life"]
vectorized_sample = vectorizer.transform(sample_article)
prediction = model.predict(vectorized_sample)
print("Prediction:", "Fake" if prediction[0]==1 else "Real")
```

## 📌 How to Run
- 1. Clone the repository or download the project
- 2. Install dependencies
pip install pandas numpy seaborn matplotlib scikit-learn

- 3. Run the script
python fake_news_detection.py

## 🎯 Results
- Achieved 98% accuracy with Logistic Regression

- Successfully identifies fake vs real news

- Demonstrates NLP + Machine Learning for misinformation detection

