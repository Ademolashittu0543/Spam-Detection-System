#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('spam.csv')
df.head()


# In[2]:


# Rename columns for clarity
df.columns = ['label', 'message']
# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
# Preview
df.head()


# In[3]:


from matplotlib import pyplot as plt
import seaborn as sns
df.groupby('label').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)


# In[4]:


# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Check unique labels
print("\nUnique labels:", df['label'].unique())

# Check for duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())


# In[5]:


# Remove duplicate rows
df = df.drop_duplicates()

# Confirm removal
print("New shape after removing duplicates:", df.shape)


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Separate features and labels
X = df['message']
y = df['label']

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization (text → numerical features)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# In[7]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

models = {
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear', probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)


# In[8]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

for name, model in models.items():
    y_pred = model.predict(X_test_tfidf)
    print(f"{name} Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test_tfidf)[:, 1]):.4f}\n")


# In[9]:


new_email = ["Congratulations! You've won a free cruise. Call now to claim!"]
new_email_2 = ["Hey, just checking in. Are we still on for lunch today?"]


# In[10]:


new_email_tfidf = vectorizer.transform(new_email)
nb_pred = models['Naive Bayes'].predict(new_email_tfidf)
print("Naive Bayes Prediction:", "Spam" if nb_pred[0] == 1 else "Not Spam")

new_email_2_tfidf = vectorizer.transform(new_email_2)
nb_pred = models['Naive Bayes'].predict(new_email_2_tfidf)
print("Naive Bayes Prediction:", "Spam" if nb_pred[0] == 1 else "Not Spam")


# In[11]:


svm_pred = models['SVM'].predict(new_email_tfidf)
print("SVM Prediction:", "Spam" if svm_pred[0] == 1 else "Not Spam")

svm_pred = models['SVM'].predict(new_email_2_tfidf)
print("SVM Prediction:", "Spam" if svm_pred[0] == 1 else "Not Spam")


# In[12]:


rf_pred = models['Random Forest'].predict(new_email_tfidf)
print("Random Forest Prediction:", "Spam" if rf_pred[0] == 1 else "Not Spam")

rf_pred = models['Random Forest'].predict(new_email_2_tfidf)
print("Random Forest Prediction:", "Spam" if rf_pred[0] == 1 else "Not Spam")


# In[13]:


lr_pred = models['Logistic Regression'].predict(new_email_tfidf)
print("Logistic Regression Prediction:", "Spam" if lr_pred[0] == 1 else "Not Spam")

lr_pred = models['Logistic Regression'].predict(new_email_2_tfidf)
print("Logistic Regression Prediction:", "Spam" if lr_pred[0] == 1 else "Not Spam")


# In[32]:


import pickle

# Example: saving SVM model
with open('model_svm.pkl', 'wb') as file:
    pickle.dump(models['SVM'], file)

