import streamlit as st

st.set_page_config(layout="wide", page_title='An Interactive ML Classifier App')
# to show radio button horizontally
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
# to avoid showing error message on plots
st.set_option('deprecation.showPyplotGlobalUse', False)

import numpy as np
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
# Using sample data
def get_data(name):
	data = None
	prj = None
	if name == 'Iris':
		data = datasets.load_iris()
		df = pd.DataFrame(np.column_stack((data.data, data.target)), columns = data.feature_names+['target'])
		prj = "Classification with Iris Dataset"
	elif name == 'Wine':
		data = datasets.load_wine()
		df = pd.DataFrame(np.column_stack((data.data, data.target)), columns = data.feature_names+['target'])
		prj = "Classification with Wine Dateset"
	else:
		data = datasets.load_breast_cancer()
		df = pd.DataFrame(np.column_stack((data.data, data.target)), columns = np.append(data.feature_names,['target']))
		prj = "Classification with Breast Cancer Dataset"

	x = pd.DataFrame(data.data,columns=data['feature_names'])
	y = pd.DataFrame(data.target, columns= ['target'])

	return df, x, y, prj
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Uploading dataset
def load_data(uploaded_dataset):
	if uploaded_dataset is not None:
		data_file = pd.read_csv(uploaded_dataset)
	else:
		data_file = None
	return data_file
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Initializing variables
prj_name = None
data_file = None
data_file_data = None
data_file_target = None
uploaded = None

st.sidebar.header("1. DATA")
with st.sidebar.expander("Configure your dataset", True):
	if st.checkbox("Load sample data ?"):
		dataset_name = st.selectbox("Choose sample dataset:", ['Iris', 'Breast Cancer Classification', 'Wine'])
		data_file, data_file_data, data_file_target, prj_name = get_data(dataset_name)

	else:
		prj_name = st.text_input('Name your project: ')

		# Sidebar options and file upload
		st.write('Load your csv file to start the app')
		uploaded_dataset = st.file_uploader(' ', type=['.csv'])
		uploaded = load_data(uploaded_dataset)

		if uploaded is not None:
			st.success('successfully uploaded!')
			st.write('Select featurs and target')
			new_data=st.multiselect("Select feature columns first. NB: Make Sure Target column is selected LAST",uploaded.columns)
			data_file=uploaded[new_data]
			#st.sidebar.dataframe(df1)
			data_file_data=data_file.iloc[:,0:-1]
			data_file_target=data_file.iloc[:,-1]

# Choose type of analysis
st.sidebar.header("2. MODELLING")
# Preprocessing
with st.sidebar.expander("Pre-processing", False):
	normalize_method = st.selectbox(
			"Features Scaling Method",
			(
				"None",
				"Normalizer",
				"StandardScaler",
				"MinMaxScaler",
			)
		)

# Splitting dataset
with st.sidebar.expander("Split the dataset", False):
	test_size = st.number_input("Set the Test datase size", 0.20, 0.40, 0.2, 0.1)
	seed=st.slider('Seed',1,200)

# Model and its parameters
with st.sidebar.expander("Train a classification model", False):
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

# # Cross-validation
# with st.sidebar.expander("Cross-Validation", False):
# 	cv_method = st.selectbox(
# 			"Specify CV method", 
# 			(
# 				"RepeatedStratifiedKFold",
# 				"StratifiedKFold",
# 				"StratifiedShuffleSplit",
# 			)
# 		)
# 	cv_split = st.number_input("CV Splits", 2, 10, 2, 1)
# 	if cv_method == "RepeatedStratifiedKFold":
# 		cv_repeat = st.number_input("CV Repeats", 1, 50, 10, 1)


step = st.radio("", 
	['Home','EDA', 'Model Result'])

if step == 'Home':
	home.home()
elif step == 'EDA':
	eda.eda(data_file, data_file_data, data_file_target, prj_name)
elif step == 'Model Result':
	train_model.build_model(data_file_data, data_file_target, normalize_method, test_size, seed, prj_name, model, algorithm)






