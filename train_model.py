import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def build_model(df_data, df_target, tst_sz, seed, prj, model, model_type):
	st.title(prj)
	st.write("Building machine learning model: ", model_type)
	
	x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=tst_sz, random_state = seed)
	
	model.fit(x_train, y_train) # 80% for training

	y_pred = model.predict(x_test) # 20% for testing

	st.write(y_pred)

	accuracy = accuracy_score(y_test, y_pred)

	st.write('Classifier Name: ', model_type)
	st.write('Accuracy for your model: ', accuracy)

	