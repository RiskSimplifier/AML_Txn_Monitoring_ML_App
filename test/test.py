
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


st.title("Random Forest Variable Importance Calculator")

# Load your dataset into a Pandas DataFrame
df = pd.read_csv("test.csv")
df = df.drop(['Risk Rating'],axis=1)



# Split the dataset into features and target
X = df.drop(columns=["STR"])
y = df["STR"]

# Train a Random Forest model on the entire dataset
rf = RandomForestClassifier()
rf.fit(X, y)

# Get the feature importances from the trained model and store them in a dictionary
importance_dict = {}
for feature, importance in zip(X.columns, rf.feature_importances_):
    importance_dict[feature] = importance

# Sort the dictionary by values in descending order
sorted_importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

# Create a horizontal bar chart to show the sorted feature importance values for each variable
fig, ax = plt.subplots(figsize=(12, 10))
ax.barh(range(len(sorted_importance_dict)), list(sorted_importance_dict.values()), align='center')
ax.set_yticks(range(len(sorted_importance_dict)))
ax.set_yticklabels(list(sorted_importance_dict.keys()))
ax.set_xlabel("Feature Importance")
ax.set_ylabel("Variable")
st.pyplot(fig)

