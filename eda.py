import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

def eda(df_data, df_target, prj):
	st.title(prj)
	st.title('Exploratory Data Analysis')
	if df_data is None:
		st.warning('No file has been uploaded.')
	else:
		st.dataframe(df_data)
		st.write("Shape of the dataset: ", df_data.shape)
		st.write("Unique target variable: ", len(np.unique(df_target)))
		#st.write("Data types: ", df.dtypes)
		#st.write("Summary: ", df_data.describe().T)
		#st.write("Null values: ", df_data.isnull().sum())
		#st.write("Null values: ", df_target.isnull().sum())

		fig = plt.figure()
		sns.boxplot(data = df_data, orient = 'h')
		st.pyplot()

		plt.hist(df_data)
		st.pyplot()

