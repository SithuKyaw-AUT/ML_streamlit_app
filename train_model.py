import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

def build_model(df_data, df_target, normal, tst_sz, seed, prj, model, model_type):
	st.title(prj)
	st.header("Result of the machine learning model")

	if df_data is None:
		st.warning('No file has been uploaded.')
	else:
		x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=tst_sz, random_state = seed)

		if normal == "Normalizer":
			normalized_x_train = Normalizer().fit_transform(x_train)
			normalized_x_test = Normalizer().transform(x_test)
			
		elif normal == "StandardScalar":
			normalized_x_train = StandardScaler().fit_transform(x_train)
			normalized_x_test = StandardScaler().transform(x_test)
			
		elif normal == "MinMaxScalar":
			normalized_x_train = MinMaxScaler().fit_transform(x_train)
			normalized_x_test = MinMaxScaler().transform(x_test)

		else: 
			normalized_x_train = x_train
			normalized_x_test = x_test

		model.fit(normalized_x_train, y_train) # 80% for training
		y_pred = model.predict(normalized_x_test) # 20% for testing
		accuracy = accuracy_score(y_test, y_pred)

		st.write('Classifier Name: ', model_type)
		st.write('Accuracy for your model: ', accuracy)
		st.markdown('The ' + model_type + ' model is predicting at {0:.2g}%.'.format(accuracy*100))

		with st.expander("Classification report", True):
			df_classification_report = pd.DataFrame(classification_report(y_test,y_pred, output_dict=True)).transpose()
			st.dataframe(df_classification_report)

		with st.expander("Confusion Matrix", True):
				# df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred))
				# st.dataframe(df_cm)
				plot_confusion_matrix(model, x_test, y_test, values_format='d')
				st.pyplot()

	