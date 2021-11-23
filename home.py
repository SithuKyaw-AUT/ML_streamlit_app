import streamlit as st

def home():
    pweb = """<a href='http://andymcdonald.scot' target="_blank">http://andymcdonald.scot</a>"""
    sm_li = """<a href='https://www.linkedin.com/in/andymcdonaldgeo/' target="_blank"><img src='https://cdn.exclaimer.com/Handbook%20Images/linkedin-icon_32x32.png'></a>"""
    sm_tw = """<a href='https://twitter.com/geoandymcd' target="_blank"><img src='https://cdn.exclaimer.com/Handbook%20Images/twitter-icon_32x32.png'></a>"""
    sm_med = """<a href='https://medium.com/@andymcdonaldgeo/' target="_blank"><img src='https://cdn.exclaimer.com/Handbook%20Images/Medium_32.png'></a>"""

    st.title('An Interactive Machine Learning App - Ver 0.1')
    st.write('## Welcome to the simple ML app')
    st.write('### Created by Sithu Kyaw')
    st.write('''This app is designed by using Python and Streamlit to help you understand the machine learning process.''')
    st.write('To begin using the app, load your \'.csv\' file using the file upload option on the sidebar. Once you have done this, you can navigate to the relevant tools using the Navigation menu.')
    st.write('\n')
    st.write('## Steps')
    st.write('**EDA:** Information from the LAS file header.')
    st.write('**Model:** Information about the curves contained within the LAS file, including names, statisics and raw data values.')
    st.write('**Result:** Visualisation tools to view las file data on a log plot, crossplot and histogram.')
    st.write('**Conclusion:** Visualisation tools understand data extent and identify areas of missing values.')
    st.write('## Get in Touch')
    st.write(f'\nIf you want to get in touch, you can find me on Social Media at the links below or visit my website at: {pweb}.', unsafe_allow_html=True)
    
    st.write(f'{sm_li}  {sm_med}  {sm_tw}', unsafe_allow_html=True)

    st.write('## Source Code, Bugs, Feature Requests')
    githublink = """<a href='https://github.com/SithuKyaw-AUT/Interactive-machine-learning-app' target="_blank">https://github.com/SithuKyaw-AUT/Interactive-machine-learning-app</a>"""
    st.write(f'\n\nCheck out the GitHub Repo at: {githublink}. If you find any bugs or have suggestions, please open a new issue and I will look into it.', unsafe_allow_html=True)
