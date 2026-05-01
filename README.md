# 🎬 Movie Reviews Sentiment Analysis

A Machine Learning project that analyzes movie reviews and classifies them as **Positive 😊** or **Negative 😡** using Natural Language Processing (NLP).

---

## 🚀 Project Overview

This project uses the **IMDB Movie Reviews dataset (50,000 reviews)** to build a sentiment analysis model.  
It applies **TF-IDF with n-grams** and trains multiple models to achieve high accuracy.

🏆 **Best Model:** Linear SVM  
📊 **Accuracy:** ~90–92%

---

## 📂 Project Structure
Movie-Reviews-Sentiment-Analysis/
│
├── Dataset/
│   └── IMDB.csv
│
├── Models/
│   ├── best_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── Images/
│
├── movie_reviews_sentiment_analysis.py
├── requirements.txt
└── README.md

---

## ⚙️ Technologies Used 🛠️

- Python 🐍  
- NumPy  
- Pandas  
- Scikit-learn  
- NLTK  
- Joblib  

---

## 🔍 Workflow

1️⃣ **Data Loading**  
- IMDB dataset with 50,000 reviews  

2️⃣ **Text Preprocessing**  
- Remove HTML tags  
- Convert to lowercase  

3️⃣ **Feature Extraction 🔥**  
- TF-IDF Vectorization  
- Unigrams + Bigrams  

4️⃣ **Model Training 🤖**  
- Multinomial Naive Bayes  
- Logistic Regression  
- Linear SVM (Best)  

5️⃣ **Evaluation 📊**  
- Accuracy Score  
- Classification Report  

6️⃣ **Model Saving 💾**  
- Saved using `joblib`  

---

## 📊 Model Performance

| Model                | Accuracy |
|---------------------|--------|
| Multinomial NB      | ~86–88% |
| Logistic Regression | ~89–91% |
| Linear SVM ✅       | **~90–92%** |
| Random Forest       | ~84–87% |

---

## 🧠 Why Linear SVM?

- Handles **high-dimensional text data** efficiently  
- Works great with **TF-IDF features**  
- Provides **better accuracy than Naive Bayes**  

---

## ▶️ How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/wanimuhammad08-lgtm/Movie-Reviews-Sentiment-Analysis.git
cd Movie-Reviews-Sentiment-Analysis

    2️⃣ Install dependencies
    pip install -r requirements.txt

    3️⃣ Run the project
    python movie_reviews_sentiment_analysis.py

💾 Saved Models

After training:

🧠 best_model.pkl → Trained SVM model
🔡 tfidf_vectorizer.pkl → Feature extractor
📌 Example Use Cases
🛒 Product review analysis
📱 Social media sentiment tracking
🎥 Movie recommendation systems
📈 Future Improvements
🌐 Build Streamlit UI
🤖 Use Deep Learning (LSTM / BERT → 95%+)
🚀 Deploy as a web application

👨‍💻 Author
Muhammad Ahmad
🔗 GitHub: https://github.com/wanimuhammad08-lgtm
