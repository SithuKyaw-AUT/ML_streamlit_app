import streamlit as st
import numpy as np

def eda(df, prj):
	st.title(prj)
	st.title('Exploratory Data Analysis')
	if df is None:
		st.warning('No file has been uploaded.')
	else:
		st.dataframe(df.head())
		st.write("Shape of the dataset: ", df.shape)
		#st.write("Data types: ", df.dtypes)
		st.write("Summary: ", df.describe().T)
		st.write("Null values: ", df.isnull().sum())
		
