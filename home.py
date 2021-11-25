import streamlit as st

def home():
    pweb = """<a href='http://andymcdonald.scot' target="_blank">http://andymcdonald.scot</a>"""
    sm_li = """<a href='https://www.linkedin.com/in/sithu-kyaw-66523334/' target="_blank"><img src='https://cdn.exclaimer.com/Handbook%20Images/linkedin-icon_32x32.png'></a>"""
    sm_tw = """<a href='https://twitter.com/geoandymcd' target="_blank"><img src='https://cdn.exclaimer.com/Handbook%20Images/twitter-icon_32x32.png'></a>"""
    sm_med = """<a href='https://medium.com/@andymcdonaldgeo/' target="_blank"><img src='https://cdn.exclaimer.com/Handbook%20Images/Medium_32.png'></a>"""

    st.title('A Machine Learning Classifier - Ver 0.1')
    st.write('### Created by Sithu Kyaw',f'{sm_li}', unsafe_allow_html=True)
    st.write('''This app is designed by using Python and Streamlit to compare and contrast of the accuracy of the Machine Learning Classifiers.''')
    st.write('To begin using the app, sample datasets can be used or you can load your \'.csv\' file using the file upload option on the sidebar. Once you have done this, you can navigate to the relevant tools using the Navigation menu.')
    st.write('\n')
    st.write('### Business Problem')
    st.write('''It is observed that the analytics team had trouble communicating their findings to the firm's management. It is also noticed that management receives just a small portion of what the analytics team provides to them (perhaps about 32%). The analytics team delivers their results to management in a raw Jupyter Notebook alongside the code and tries to explain it to them in a way that is difficult for them to understand. 
''')
    st.write('### Objective')
    st.write('''This aims to help the analytics team communicate better with management. Using Streamlit, this interactive web application would help them better demonstration of their findings to the management of the business.
''')
    st.write('\n')
    st.write('### Sections')
    st.write('**Home:** Information about the project.')
    st.write('**EDA:** Exploratory data analysis on the dataset.')
    st.write('**Model Result:** Result of the machine learning Classifier.')

    st.write('\n')
    st.write('### Source Code')
    githublink = """<a href='https://github.com/SithuKyaw-AUT/Interactive-machine-learning-app' target="_blank">https://github.com/SithuKyaw-AUT/Interactive-machine-learning-app</a>"""
    st.write(f'\n\nSource code at the GitHub Repo: {githublink}. \n\nMost welcome for the feedback and suggestions.', unsafe_allow_html=True)

