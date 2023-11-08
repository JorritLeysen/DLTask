import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import category_encoders as ce
import streamlit as st

# Fetch dataset
car_evaluation = fetch_ucirepo(id=19)

# Data (as pandas dataframes)
car_evaluation_df = pd.DataFrame(car_evaluation.data.original)
feature_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

X = car_evaluation_df[feature_cols]  # Features
y = car_evaluation_df['class']  # Labels

ce_ord = ce.OrdinalEncoder(cols=feature_cols)
X_cat = ce_ord.fit_transform(X)

ce_ordY = ce.OrdinalEncoder(cols='class')
Y_cat = ce_ordY.fit_transform(y)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_cat, Y_cat, test_size=0.3, random_state=42)  # 70% training and 30% test

# Display the accuracy for Decision Tree
clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X_train, y_train)
y_pred_dt = clf.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# Display the accuracy for K-Nearest Neighbors
clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# Display the accuracy for HistGradientBoostingClassifier
clf_hgbc = HistGradientBoostingClassifier(max_iter=100)
clf_hgbc.fit(X_train, y_train)
y_pred_hgbc = clf_hgbc.predict(X_test)
accuracy_hgbc = accuracy_score(y_test, y_pred_hgbc)

# Streamlit app
st.title('Car Evaluation Dataset')

st.subheader('Dataset Information')
st.write(f"Dataset Size: {car_evaluation_df.shape[0]} rows and {car_evaluation_df.shape[1]} columns")
st.write('Features:', feature_cols)
st.write('Target:', 'class')

# Display the dataset size
st.write(f"Dataset size: {X.shape[0]} rows and {X.shape[1]} columns")

# Show the target (class) distribution
st.write("Target Distribution:")
st.write(y.value_counts())

# Display the first few rows of the dataset
st.write("Sample Data:")
st.write(X.head())
st.write(y.head())

st.subheader('Decision Tree Classifier')
st.write(f"Accuracy: {accuracy_dt:.2f}")

st.subheader('K-Nearest Neighbors Classifier')
st.write(f"Accuracy: {accuracy_knn:.2f}")

st.subheader('HistGradientBoosting Classifier')
st.write(f"Accuracy: {accuracy_hgbc:.2f}")

# Add more Streamlit components for visualizing EDA and other comparisons

# Display confusion matrices or graphs here
