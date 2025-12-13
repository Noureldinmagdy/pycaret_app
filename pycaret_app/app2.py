import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import time 
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from pycaret.classification import setup as clf_setup, compare_models as clf_compare_models, pull as clf_pull
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull

pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',100)
pd.set_option('display.width',160)

#st.set_page_config(page_title="PyCaret", layout="wide")

st.title("Welcome To PyCaret App")
time.sleep(1)
st.write("                      ")
st.sidebar.title("Configuration")
tabs = st.tabs(["Get started now 🚀"])

with tabs[0]:
    upload_f = st.file_uploader("upload dataset csv type", type=['csv'])
    if upload_f :
        df=pd.read_csv(upload_f)



        st.dataframe(df.head(10))
        st.write('rows >',df.shape,'< columns')
        st.subheader("Columns Name & type")
        st.dataframe(df.dtypes)
        st.subheader("missing valus")
        st.dataframe((df.isna().sum() / len(df)*100).sort_values(ascending=False))

        cat_cols = df.select_dtypes(include=['object','category']).columns
        for i in cat_cols:
            df[i]=df[i].fillna(df[i].mode())

        num_cols = df.select_dtypes(include=['int64','float64']).columns
        for i in num_cols:
            df[i]= df[i].fillna(df[i].median())
        st.subheader("after fill missing values")   
        st.dataframe((df.isna().sum() / len(df)*100).sort_values(ascending=False))
        
        ohe = OneHotEncoder(sparse_output=False , handle_unknown= 'ignore')
        cc = ohe.fit_transform(df[cat_cols])
        c_c =pd.DataFrame(cc)
        final_df = pd.concat([df[num_cols] , c_c] , axis=1)
        final_df.columns = final_df.columns.astype(str)

    
        target_col = st.selectbox("Select Target Column", df.columns)
        
        X = final_df.drop(columns=[target_col], errors='ignore')
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier( n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.subheader("Model Accuracy")
        st.write(f"Accuracy: {acc:.2%}")



with st.sidebar:
    st.tabs(["Get info ℹ️"])
    st.write("""This program was designed to facilitate the following tasks
        exploratory data analysis
        handeling missing values -
        encoding -
        machine learning and
        evaluate the model
        """)
    st.write("• upload data and click buttons to see details")
    st.warning("• This for csv dataset only")
    
