import streamlit as st
from streamlit_option_menu import option_menu as om
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle


df = pd.read_csv('data//hearing_test.csv')
loaded_model = pickle.load(open('lr_hearing_loss.sav', 'rb'))


with st.sidebar:
    selected = om(menu_title = "Main Menu",
                  options = ["Home", 'Graphs','Project']
        )
    
    
if selected =='Home':
    st.title('Hearing Loss Web App')
    st.image('asset//hearing_loss.png', width = 800)
    
    if st.checkbox('Check Data'):
        st.dataframe(df, width = 800)
        
        
if selected =='Graphs':
    st.header('Graphs of Data')
    
    graph = st.selectbox('What Kind of Graphs', ['None','Interactive', 'Non Interactive'])
    if graph =='None':
        pass
    if graph == 'Interactive':
        
        if graph =='Interactive':  
            fig = px.scatter(df,
                            x=df.age,
                            y = df.physical_score, 
                            labels = {'x' : 'Age', 'y': 'physical_score'},
                            color= 'test_result')
            st.plotly_chart(fig)
            
        st.write('---------------------------------------------------')
        
        
        fig = px.scatter_3d(df, x=df['age'], y=df['physical_score'], z=df['test_result'], 
                    opacity=1, color=df['test_result'].astype(str), 
                    color_discrete_sequence=['black']+px.colors.qualitative.Plotly,
                    width=500, height=500)
        st.plotly_chart(fig)
    
    if graph == 'Non Interactive':
        #plt.style.use(['science','notebook','grid'])
        plt.figure(figsize=(6,4), dpi=100)
        ax = sns.countplot(x='test_result', data=df, palette= 'spring')
        for p in ax.patches:
            height = p.get_height()
            ax.text(x = p.get_x() + p.get_width()/2, 
                    y = height + 20, 
                    s = '{:.0f}'.format(height), 
                    ha ='center'
                    )
        plt.title('Test Result')
        plt.ylim(0, 3500)
        st.pyplot(ax.get_figure())
        
        st.write('---------------------------------------------------')
        
        plt.figure(figsize=(6,4), dpi=100)
        ax = sns.scatterplot(x='age', y='physical_score', hue ='test_result', data= df, palette='spring')
        plt.title('Distribution Data')
        st.pyplot(ax.get_figure())
        
        st.write('---------------------------------------------------')
        
        metric = df.corr()
        mask = np.zeros_like(metric)
        mask[np.triu_indices_from(mask)]= True
        ax = sns.heatmap(metric, mask=mask, lw=1, annot=True, cmap='spring')
        plt.title('Corelation Data')
        st.pyplot(ax.get_figure())
        
        st.write('---------------------------------------------------')
        
    
if selected =='Project':
    st.header('Test Hearing Loss')
    age = st.number_input('your age', min_value =0, max_value= 100, step =1)
    physical_score = st.number_input('physical score', min_value = 00.00, max_value = 100.00, step =0.01)
    
    def test_result(input_data):
        
        input_data_np_array = np.asarray(input_data)
        
        input_data_reshape = input_data_np_array.reshape(1, -1)
        
        prediction = loaded_model.predict(input_data_reshape)[0]
        
        print(prediction) 
        
        if prediction== 0:
            return'The person is Not Pass'
        else :
            return'The person is Pass'
        
    diagnosis = ''
        
    if st.button('Test Result'):
        diagnosis = test_result([age,physical_score])
        st.success(diagnosis)  
    
