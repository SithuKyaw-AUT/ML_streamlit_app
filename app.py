import streamlit as st

st.set_page_config(layout="wide", page_title='An Interactive ML app')

import pandas as pd
from load_css import local_css

from sklearn import datasets

# Models import
from models.DecisionTree import dt_param_selector
from models.KNearestNeighbors import knn_param_selector
from models.RandomForest import rf_param_selector
from models.NaiveBayes import nb_param_selector
from models.SVC import svc_param_selector
from models.NeuralNetwork import nn_param_selector
from models.LogisticsRegression import lr_param_selector
from models.GradientBoosting import gb_param_selector

# local imports
import home
import eda
import train_model
import result
import conclusion

local_css("style.css")

@st.cache
def load_data(uploaded_dataset):
	if uploaded_dataset is not None:
		data_file = pd.read_csv(uploaded_dataset)
	else:
		data_file = None
	return data_file

def get_data(name):
	data = None
	prj = None
	if name == 'Iris':
		data = datasets.load_iris()
		prj = "Classification with Iris Dataset"
	elif name == 'Wine':
		data = datasets.load_wine()
		prj = "Classification with Wine Dateset"
	else:
		data = datasets.load_breast_cancer()
		prj = "Classification with Breast Cancer Dataset"

	x = data.data
	y = data.target

	return x, y, prj


# Name the project
prj_name = None
data_file = None
data_file_data = None
data_file_target = None

st.sidebar.header("1. DATA")
dataset_container = st.sidebar.expander("Configure your dataset", True)

with dataset_container:
	if st.checkbox("Load sample data ?"):
		dataset_name = st.selectbox("Choose sample dataset:", ['Iris', 'Breast Cancer Classification', 'Wine'])
		data_file_data, data_file_target, prj_name = get_data(dataset_name)
	else:
		prj_name = st.text_input('Name your project: ')

		# Sidebar options and file upload
		st.write('Load your csv file to start the app')
		uploaded_dataset = st.file_uploader(' ', type=['.csv'])
		data_file = load_data(uploaded_dataset)

		if data_file is not None:
			st.success('successfully uploaded!') 
			st.write('Select featurs and target')
			new_data=st.multiselect("Select feature columns first. NB: Make Sure Target column is selected LAST",data_file.columns)
			df1=data_file[new_data]
			#st.sidebar.dataframe(df1)
			data_file_data=df1.iloc[:,0:-1]
			data_file_target=df1.iloc[:,-1]


# Choose type of analysis

st.sidebar.header("2. MODELLING")
with st.sidebar.expander("Split the dataset", False):
	test_size = st.number_input("Set the Test datase size", 0.20, 0.40, 0.2, 0.1)
	seed=st.slider('Seed',1,200)


model_train_container = st.sidebar.expander("Train a classification model", False)
with model_train_container:
	algorithm = st.selectbox(
			"Choose a model",
			(
				"Logistics Regression",
				"Random Forest",
				"Decision Tree",
				"Naive Bayes",
				"K-Nearest Neighbors",
				"Support Vector Machine",
				"Neural Networks",
				"Gradient Boosting",
			),	
		)

	if algorithm == "Logistics Regression":
		model = lr_param_selector()
	elif algorithm == "Random Forest":
		model = rf_param_selector()
	elif algorithm == "Decision Tree":
		model = dt_param_selector()
	elif algorithm == "Naive Bayes":
		model = nb_param_selector()
	elif algorithm == "K-Nearest Neighbors":
		model = knn_param_selector()
	elif algorithm == "Support Vector Machine":
		model = svc_param_selector()
	elif algorithm == "Neural Networks":
		model = nn_param_selector()
	elif algorithm == "Gradient Boosting":
		model = gb_param_selector()


st.sidebar.header("3. EVALUATION")
with st.sidebar.expander("Evaluation Metrics", False):
	st.multiselect("Select Metrics: ", ['MSE', 'RMSE', 'MAPE', 'SMAPE', 'MAE'])


# sidebar navigation
st.sidebar.title('Steps')
step = st.sidebar.radio('Select a page:', 
	['Home','EDA', 'Model','Result', 'Conclusion'])

if step == 'Home':
	home.home()
elif step == 'EDA':
	eda.eda(data_file_data, data_file_target, prj_name)
elif step == 'Model':
	train_model.build_model(data_file_data, data_file_target, test_size, seed, prj_name, model, algorithm)
elif step == 'Result':
	result.result(data_file, prj_name)
elif step == 'Conclusion':
	conclusion.conclusion(data_file, prj_name)

