import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

def eda(df, df_data, df_target, prj):
	st.title(prj)
	st.header('Exploratory Data Analysis')
	if df_data is None:
		st.warning('No file has been uploaded.')
	else:
		if st.checkbox("Data table ?"):
			st.dataframe(df)
		if st.checkbox("Shape of the dataset ?"):
			st.write(df.shape)
		if st.checkbox("Category of target variable ?"):
			st.write("Unique target variable: ", len(np.unique(df_target)))
		#st.write("Data types: ", df.dtypes)
		if st.checkbox("Summary of the variables ?"):
			st.write("Summary: ", df.describe().T)
		if st.checkbox("Null values in the variables ?"):
			st.write("Null values: ", df.isnull().sum())
		#st.write("Null values: ", df.isnull().sum())
		if st.checkbox("Distribution of data for each variable by a boxplot ?"):
			fig = plt.figure()
			sns.boxplot(data = df, orient = 'h')
			st.pyplot()	
		if st.checkbox("Correlation matrix ?"):
			corr_table_map = st.radio("",("Table","Heatmap"))
			if corr_table_map == "Table":
				st.write(df.corr())
			else:
				st.write(sns.heatmap(df.corr(),vmax=1, square=True, annot=False, cmap='viridis'))
				st.pyplot()
		if st.checkbox("Pairplot ?"):
			st.write(sns.pairplot(df, diag_kind='kde'))
			st.pyplot()
		if st.checkbox("Data balance ?"):
			all_columns = df.columns.to_list()
			pie_column = st.selectbox("Select variable: ", all_columns)
			piechart = df[pie_column].value_counts().plot.pie(autopct="%1.1f%%")
			st.write(piechart)
			st.pyplot()

