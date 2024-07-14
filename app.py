
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
if  uploaded_file is not None:
    
    data = pd.read_csv(uploaded_file)
    
else:
    st.warning("No data to display.") 
num_rows = st.number_input('Enter the number of rows to display', 
                            min_value=0, max_value=30, value=5)
st.header("Data Sample")
st.dataframe(data.head(num_rows))
def plot_cat(data, cat_var):
    st.header("Plot of " + cat_var)
    fig, ax = plt.subplots()
    sns.set_style('darkgrid')
    sns.countplot(data=data, x=cat_var)
    plt.title(cat_var)
    plt.show()
    st.pyplot(fig)
columns = data.columns.tolist()
cat_var = st.selectbox('Select a column to plot', columns)
plot_cat(data, cat_var)

def encode_cat(data, cat_var):
    encoder = OrdinalEncoder()
    data[cat_var] = encoder.fit_transform(data[[cat_var]])
    return data
for i in data.columns:
    if data[i].dtypes == 'object':
        encode_cat(data, i)
st.header("Data Encoded Dataframe Sample")
st.dataframe(data.head(5))

#performance rating by department
st.header("Performance Rating by Department")
fig, ax = plt.subplots()
sns.barplot( x="EmpDepartment", y ="PerformanceRating" ,data=data,palette="Set2" )
plt.title("Performance Rating by Department")
plt.show()
st.pyplot(fig)
st.text("Departments: 0 = Data Science, 1= Research & Development, 2 = Finance, 3 = Human Resources, 4 = Sales")




X = data.drop(columns=['PerformanceRating'])
model = RandomForestClassifier(
             n_estimators= 1135,
             min_samples_split= 10,
             min_samples_leaf= 1,
             max_features= 'sqrt',
             max_depth= 10,
             criterion= 'gini',
             bootstrap= False
             )
model.fit(X, data['PerformanceRating'])
# make predictions using the model
prediction = model.predict(X)
data['PerformanceRating_prediction'] = prediction

num_rows_pred = st.number_input('Enter the number of rows to display', 
                            min_value=0, max_value=50, value=5)

st.header("Predictions")
st.dataframe(data.head(num_rows_pred))

st.text("Performance Ratings: 1 = Low, 2 = Good , 3 = Better , 4 = Very High")

st.header("Classification Report")
class_report = classification_report(data['PerformanceRating'],
                                    data['PerformanceRating_prediction'])
st.text(class_report)





