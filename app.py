
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


import warnings
warnings.filterwarnings('ignore')

st.title('Employee Perfomance Prediction')

uploaded_file = st.file_uploader("Upload your input CSV file", 
                                 type=["csv"])

if uploaded_file is not None:
   df = pd.read_csv(uploaded_file)
   
    
    
num_rows = st.number_input('Enter the number of rows to display', 
                            min_value=0, max_value=30, value=5)


st.header("Data Sample")

st.dataframe(df.head(num_rows))

def plot_cat(df, cat_var):
    st.header("Plot of " + cat_var)
    fig, ax = plt.subplots()
    sns.set_style('darkgrid')
    sns.countplot(data=df, x=cat_var)
    plt.title(cat_var)
    plt.show()
    st.pyplot(fig)

columns = df.columns.tolist()

cat_var = st.selectbox('Select a column to plot', columns)

plot_cat(df, cat_var)

def encode_cat(df, cat_var):
    encoder = OrdinalEncoder()
    df[cat_var] = encoder.fit_transform(df[[cat_var]])
    return data

for i in df.columns:
    if df[i].dtypes == 'object':
        encode_cat(df, i)


st.header("Data Encoded Dataframe Sample")
st.dataframe(df.head(3))


X = df.drop(columns=['PerformanceRating'])


model = RandomForestClassifier(
             n_estimators= 1135,
            min_samples_split= 10,
             min_samples_leaf= 1,
             max_features= 'auto',
             max_depth= 10,
             criterion= 'gini',
             bootstrap= False
             )
model.fit(X, df['PerformanceRating'])


prediction = model.predict(X)


data['PerformanceRating_prediction'] = prediction

num_rows_pred = st.number_input('Enter the number of rows to display', 
                            min_value=0, max_value=50, value=5)

st.header("Predictions")
st.dataframe(df.head(num_rows_pred))


st.header("Classification Report")
st.text("1 = Low, 2 = Good , 3 = Better , 4 = Very High")

class_report = classification_report(data['PerformanceRating'],
                                    data['PerformanceRating_prediction'])
st.text(class_report)
