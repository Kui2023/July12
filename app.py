
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
