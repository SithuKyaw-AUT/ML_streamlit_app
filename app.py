import streamlit as st

st.set_page_config(layout="wide", page_title='An Interactive ML app')

import pandas as pd
from load_css import local_css

# Models import
from models.DecisionTree import dt_param_selector
from models.KNearestNeighbors import knn_param_selector
from models.RandomForest import rf_param_selector
from models.NaiveBayes import nb_param_selector
from models.SVC import svc_param_selector
from models.NeuralNetwork import nn_param_selector
from models.LogisticsRegression import lr_param_selector

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


# Name the project
prj_name = None
data_file = None

dataset_container = st.sidebar.expander("Configure your dataset", True)
with dataset_container:
	prj_name = st.text_input('Name your project: ')

	# Sidebar options and file upload
	st.write('Load your csv file to start the app')
	uploaded_dataset = st.file_uploader(' ', type=['.csv'])
	data_file = load_data(uploaded_dataset)

	if data_file is not None:
		st.success('successfully uploaded!') 

# Choose type of analysis
obj_prj = None

model_train_container = st.sidebar.expander("Train a model", True)
with model_train_container:
	obj_prj = st.radio('Choose objective of the project: ', ['Regression', 'Classification', 'Clustering'])

	if obj_prj=='Regression':
		algorithm = st.selectbox(
			"Choose a model",
			(
				"Linear Regression",
				"Random Forest",
				"Decision Tree",
				"Hierarchical",
				"Ensembles",
				"Neural Networks",
			),	
		)
	elif obj_prj == 'Classification':
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
				"Discriminant Analysis"
			),	
		)
	elif obj_prj == 'Clustering':
		algorithm = st.selectbox(
			"Choose a model",
			(
				"K-Means",
				"Hierarchical",
				"Neural Networks",
			),	
		)


	if algorithm == "Decision Tree":
		model = dt_param_selector()
	elif algorithm == "K-Nearest Neighbors":
		model = knn_param_selector()
	elif algorithm == "Random Forest":
		model = rf_param_selector()
	elif algorithm == "Support Vector Machine":
		model = svc_param_selector()
	elif algorithm == "Neural Networks":
		model = nn_param_selector()
	elif algorithm == "Logistics Regression":
		model = lr_param_selector()





# sidebar navigation
st.sidebar.title('Steps')
step = st.sidebar.radio('Select a page:', 
	['Home','EDA', 'Model','Result', 'Conclusion'])

if step == 'Home':
	home.home()
elif step == 'EDA':
	eda.eda(data_file, prj_name)
elif step == 'Model':
	train_model.build_model(data_file, prj_name)
elif step == 'Result':
	result.result(data_file, prj_name)
elif step == 'Conclusion':
	conclusion.conclusion(data_file, prj_name)

