import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import re

from prefect import task, Flow

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stem = PorterStemmer()
lemma = WordNetLemmatizer()

@task
def load_data(file_path):
    """
    Load data from a CSV file.
    """
    df = pd.read_csv(file_path)
    df['label'] = df['Ratings'].apply(lambda x: 'negative' if x <= 2 else 'positive')
    return df

@task
def split_inputs_output(data, inputs, output):
    """
    Split features and target variables.
    """
    X = data[inputs]
    y = data[output]
    return X, y

@task
def split_train_test(X, y, test_size=0.25, random_state=0):
    """
    Split data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

@task
def preprocess_data(X_train, X_test):
    """
    Preprocess the text data and transform it into numerical using CountVectorizer.
    """
    def preprocess_text(text):
        if isinstance(text, str):
            # removes special characters
            text = re.sub("[^a-zA-Z]"," ", text)
            
            # converts words to lowercase
            text = text.lower()
            
            # tokenization
            text = text.split()
            
            # removes the stop words
            text = [word for word in text if word not in stopwords.words('english')]
            
            # applying lemmatization
            text = [lemma.lemmatize(word) for word in text]
            
            return " ".join(text)
        else:
            return ""  # Replace non-string values with an empty string
    
    # Preprocess training data
    X_train_preprocessed = [preprocess_text(text) for text in X_train]
    
    # Preprocess testing data
    X_test_preprocessed = [preprocess_text(text) for text in X_test]
    
    # Transform text data into numerical using CountVectorizer
    cv = CountVectorizer()
    X_train_vectorized = cv.fit_transform(X_train_preprocessed)
    X_test_vectorized = cv.transform(X_test_preprocessed)
    
    return X_train_vectorized, X_test_vectorized


@task
def train_model(X_train_vectorized, y_train):
    """
    Train the machine learning model.
    """
    nv_model =  MultinomialNB()
    nv_model.fit(X_train_vectorized, y_train)
    return nv_model

@task
def evaluate_model(model, X_train_vectorized, y_train,X_test_vectorized , y_test):
    """
    Evaluating the model.
    
    """
    from sklearn import metrics
    y_train_pred = model.predict(X_train_vectorized)
    y_test_pred = model.predict(X_test_vectorized)

    train_score = metrics.accuracy_score(y_train, y_train_pred)
    test_score = metrics.accuracy_score(y_test, y_test_pred)
    
    return train_score, test_score

@flow(name="Naive_bayes_model_development")
def workflow(data_path):
    DATA_PATH = "data.csv"

    # Load data
    df = load_data(DATA_PATH)
    
    # Identify Inputs and Output
    X, y = split_inputs_output(df, "Review text", "label")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Preprocess the data and transform it into numerical using CountVectorizer
    X_train_vectorized, X_test_vectorized= preprocess_data(X_train, X_test)
    
    # Build a model
    model = train_model(X_train_vectorized, y_train)
    
    # Evaluation
    train_score, test_score = evaluate_model(model, X_train_vectorized, y_train, X_test_vectorized, y_test)
    
    print("Train Score:", train_score)
    print("Test Score:", test_score)

    if __name__ == "__main__":
    workflow.serve(
        name="model_prefect_integration",
        cron="* * * * *"
    )