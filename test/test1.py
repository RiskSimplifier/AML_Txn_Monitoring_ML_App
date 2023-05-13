import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Increase the width of the Streamlit app
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")
# Load data
df = pd.read_csv("test.csv")
df = df.drop(['Risk Rating'],axis=1)
# Compute the correlation matrix
corr = df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.pcolor(corr.values, cmap='coolwarm')
fig.colorbar(im)
ax.set_xticks(np.arange(len(corr.columns))+0.5, minor=False)
ax.set_yticks(np.arange(len(corr.index))+0.5, minor=False)
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.index)
ax.set_title("Correlation Matrix (Matplotlib)")
st.pyplot(fig)