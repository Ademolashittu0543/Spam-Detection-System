# Spam Detection System
This project implements a spam detection system using various machine learning models to classify emails as spam or not spam.

## Purpose
The goal of this project is to demonstrate how different machine learning algorithms can be used to detect spam emails. It includes data preprocessing, feature extraction, model training, and evaluation.You can try the deployed version of this spam detection system here.>>[Spam Detection System](https://spam-detection-system-vlvakga8tabxfyc3s7kufo.streamlit.app)


## Technologies Used
- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- pickle

  ## How to Run
1. Ensure you have Jupyter Notebook installed. If not, you can install it using `pip install notebook`.
2. Clone this repository to your local machine.
3. Navigate to the repository directory and launch Jupyter Notebook using `jupyter notebook`.
4. Open the `Spam_Detection_System_ipnb.ipynb` file and run the cells sequentially.

   ## Dataset
The dataset used in this project is `spam.csv`, which contains labeled emails as spam or not spam. The dataset is loaded and preprocessed in the notebook.

## Project Steps
1. **Data Loading:** Load the dataset using pandas.
2. **Data Preprocessing:** Handle missing values, remove duplicates, and convert labels to binary.
3. **Feature Extraction:** Use TF-IDF vectorization to convert text data into numerical features.
4. **Model Training:** Train multiple models including Naive Bayes, SVM, Random Forest, and Logistic Regression.
5. **Model Evaluation:** Evaluate the models using accuracy, precision, recall, F1-score, and ROC-AUC.
6. **Prediction:** Use the trained models to predict whether new emails are spam or not.

## Results
The SVM model performed the best with an accuracy of 0.9835, precision of 0.9474, recall of 0.9265, F1-score of 0.9368, and ROC-AUC of 0.9891.

## Using the Trained Model
The trained SVM model and the TF-IDF vectorizer are saved as `model_svm.pkl` and `tfidf_vectorizer.pkl`, respectively. To use them for prediction:

1. Load the vectorizer and the model using pickle.
2. Transform the new email text using the vectorizer.
3. Use the model to predict whether the email is spam or not.

Example code:
```python
import pickle

# Load the vectorizer and model
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
with open('model_svm.pkl', 'rb') as file:
    model = pickle.load(file)

# New email text
new_email = ["Congratulations! You've won a free cruise. Call now to claim!"]

# Transform the text
new_email_tfidf = vectorizer.transform(new_email)

# Predict
prediction = model.predict(new_email_tfidf)
print("Spam\" if prediction[0] == 1 else \"Not Spam")

