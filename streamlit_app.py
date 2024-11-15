#Importing the libraries
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import pandas as pd 
import pandas.api.types as ptypes
#the above is to get to know the type of column
import numpy as np
from streamlit_pandas_profiling import st_profile_report
#  #this will allow us to use pandas profiling in the streamlit application
import matplotlib.pyplot as plt # this will import the pyplot from the matplotlib.pyplot
import pdfkit # this will import the pdfkit
import tempfile # this will import the tempfile
from datetime import datetime# this will import the datetime module for the streamlit webapp
#also installed wkhtmltopdf
from keras import preprocessing
from keras import applications
import torch
import torch.nn as nn
import torch.nn.functional as F


#<------------------------------------------------------->





#Sidebar for navigation
with st.sidebar:
    selected =option_menu('Choose from the given options',
                          ["Data Analysis","Pneumonia Prediction","Diabetes Prediction"],default_index=0)
#<------------------------------------------------------->

#creating the ANN_Model class
#Now we are creating our model with the help of Pytorch
#First thing for this is to create a class
#for an ANN we require input features , hidden layers , output layers
class ANN_Model(nn.Module):#here the class inherited the nn.Module
    #here we will create the functions
    def __init__(self,input_features=8,hidden1=20,hidden2=20,out_features=2):
      #here we are storing 8 as the value for the input features and 20 nodes for each hidden layer
      #and 2 nodes for the output layer
      #here we will be using 'categorical cross entropy ' so that we can see how the probability
      #values are coming up
        super().__init__()#this is written to inherit the parent class ANN_Model
        #now below is the fully connected layer

        #the input layer and the first hidden layer will be constituting the fully connected layer
        #in order to create a fully connected layer we will be creating a dense layer
        self.f_connected1=nn.Linear(input_features,hidden1) #this is the first fully connected layer
        self.f_connected2=nn.Linear(hidden1,hidden2)#second fully connected layer
        self.out=nn.Linear(hidden2,out_features)
    def forward(self,x):#this is for forward propogation
    #the parameter x will drag the "Gradient Descent" and all the "Backpropogation"
        x=F.relu(self.f_connected1(x)) #here 'x' in the parameter of 'self.f_connected1(x)' is
        #basically the all the information related to that layer
        x=F.relu(self.f_connected2(x))
        x=self.out(x)
        return x

#<------------------------------------------------------->


#Loading the saved models
model = pickle.load(open("model_pickle",'rb'))
# m1=pickle.load(open("model1.pkl",'rb'))
m1 = torch.load('diabetes.pt')
#<------------------------------------------------------->


#Data Analysis Page
if (selected =="Data Analysis"):
    #page title
    st.title("Extracting information from the given dataset")
    uploaded_file = st.file_uploader("Upload your input CSV")
    #Pandas Profiling Report
    if uploaded_file is not None:
        @st.cache_data
        
    # In Streamlit, the @st.cache decorator is used to cache the output of a function, so that when the function is called with the same inputs, the results are reused rather than recalculated. This is especially useful when working with operations that are computationally expensive or time-consuming, such as loading large datasets or running complex computations.

    # In your code, the @st.cache decorator is applied to the load_csv function:

    # python
    # Copy code
    # @st.cache
    # def load_csv():
    #     csv = pd.read_csv(uploaded_file)
    #     return csv
    # Purpose of @st.cache in Your Code:
    # Efficiency: If the file uploaded_file remains the same, load_csv() won’t need to reload the CSV from disk repeatedly; it will use the cached DataFrame instead.
    # Performance: This caching improves performance in Streamlit apps, where reruns occur whenever a user interacts with widgets. Without caching, the file would be reloaded each time the app reruns, potentially slowing it down.
    # Reducing Redundant Computation: If users toggle between views or perform other actions, Streamlit re-runs the code. Cached functions ensure that redundant computations are avoided.
















        def load_csv():
            csv = pd.read_csv(uploaded_file)
            return csv
        df = load_csv()
        # pr = ProfileReport(df,explorative =True)
        st.header('**Input Dataframe**')
        st.write(df)
        st.write('---')
        cl = df.columns.tolist()
        st.write("Please select any of the columns to get insights of the data in it ")
        st.write(cl)
        name = st.text_input("Please enter the required column:-")
        if name is not None:
            
            if name in cl:
                if ptypes.is_numeric_dtype(df[name]):
                    #this will get passed if the type of data in the name column is of numeric type
                    st.write(f"The columns name is {name}") #this will wrtie the name of the column
                    mn = df[name].mean() # this will calculate the mean 
                    med = df[name].median() # this will calculate the median
                    md = df[name].mode()[0] if not df[name].mode().empty else 'No Mode' 
                    # this will tell the mode of the column
                    st.write(f"**Median** is :- {mn} ")
                    #this will write the mean 
                    st.write(f"The **Median** is :- {md}")
                    #this will write the median
                    st.write(f"The **Mode** is :- {md} ")
                    #this will write the mode
                    sum= df[name].describe()# this will describe the data
                    st.write(f"The summary is {sum}") #this will write the summary


                    #Grouped the numeric data into bins example intervals of 20
                    bin_labels = ['0-20','20-40','40-60','60-80','80-100']
                    df[name+" "]= pd.cut(df[name],bins=[0,20,40,60,80,100],labels=bin_labels)

                    #counting the occurrences in each bin
                    c_counts = df[name+" "].value_counts()



                    

                    st.title(f"Pie chart for the data in column {name}")
                    #this will set the title for the pie chart 


                    # now plotting the pie chart
                    fig,ax = plt.subplots()
                    ax.pie(c_counts,labels = c_counts.index,autopct = '%1.1f%%',startangle = 90)
                    ax.axis('equal') # Equal aspect ratio ensures the pie chart is circular 

                    st.pyplot(fig) #displaying the pie chart in streamlit 
                    









                    #Generating the report as HTML
                    report_html  = f"""
                    <h1> Analysis Report for {name}</h1>
                    <p> <strong>Mean: </strong>{mn}</p>
                    <p><strong>Median:</strong>{med}</p>
                    <p><strong>Mode: </strong>{md} </p>

                    """
                    #Converting the pie chart to HTML image 
                    with tempfile.NamedTemporaryFile(delete=False,suffix=".png") as tmpfile:
                        fig.savefig(tmpfile.name)
                        pie_chart_html = f'<img src ="file://{tmpfile.name}" width = "400"'
                    report_html +=f"<h2>Pie Chart for {name}</h2>{pie_chart_html}"
                    #PDF generation button
                    if st.button("Download Analysis as PDF "):
                        #Save HTML to PDF 
                        config = pdfkit.configuration(wkhtmltopdf=r'"C:\Users\Tarun\Downloads\wkhtmltox-0.12.6-1.msvc2015-win64.exe"')
                        pdf_file =pdfkit.from_string(report_html,False,configuration=config)
                        c_time = datetime.now.strftime("%y-%m-%d %H:%M:%S")#this get the current time and will be used to distinguish between files
                        st.download_button(label="Download  PDF ", data=pdf_file, file_name=f"{name}_analysis_{c_time}.pdf",mime = "application/pdf")




                else:
                    st.write(f"The columns name is {name}") # this will write the name of the column
                
            if name not in cl:
                st.write("The entered column is not in the list!!!")
        else:
            pass
    else:
        st.info('Awaiting for CSV file to be uploaded')
        if st.button('Press to use the already uploaded dataset'):
            # @st.cache
            #this is for reading the csv file 
            df1 = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Streamlit\HospitalManagement\archive\healthcare_dataset.csv')#here r is placed before for raw string 
            #here we droped the hospital column from the dataframe
            df= df1.drop('Hospital', axis = 1)
            st.write(df)  

#<------------------------------------------------------->

#Pneumonia Detection Page
if(selected =="Pneumonia Prediction"):
    #setting the title
    st.title("Pneumonia Prediction")
    #uploading the the image
    # up_f=st.sidebar.file_uploader("Upload the image")
    up_f=st.file_uploader("Upload the image")
    if up_f is not None:
        img = preprocessing.image.load_img(up_f,target_size=(224,224))
        x= preprocessing.image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        image_data = applications.vgg16.preprocess_input(x)
        classes = model.predict(image_data)
        if(classes[0][0]>classes[0][1]):
            st.write("Normal")
        else:
            st.write("Pneumonia Detected")


#<------------------------------------------------------->

#Diabetes Prediction Page

if(selected=="Diabetes Prediction"):
    #setting the title
    st.title("Diabetes Prediction")
   

    #below are the input fields
    p=st.number_input("Number of Pregnancies",value=None,placeholder="Type a Number",key='1')
    g=st.number_input("Glucose Level",value=None,placeholder="Type a Number",key='2')
    b=st.number_input("Blood Pressure Level",value=None,placeholder="Type a Number",key='3')
    s=st.number_input("Skin Thickness",value=None,placeholder="Type a Number",key='4')
    i=st.number_input("Insulin",value=None,placeholder="Type a Number",key='5')
    bm=st.number_input("BMI",value=None,placeholder="Type a Number",key='6')
    d=st.number_input("Diabetes Pedigree Function",value=None,placeholder="Type a Number",key='7')
    a=st.number_input("Age",value=None,placeholder="Type a Number",key='8')
    #All these number_inputs should have different key values



    #Now creating the submit button
    if st.button("Submit"):
        l=[p,g,b,s,i,bm,d,a] #All the entered values are inserted in a list
        #Converting the list l into a tensor
        new_data= torch.tensor(l)
        #now making the prediction on the new_data
        prediction = m1(new_data).argmax().item() #this prediciton will contain the index of the class with the maximum value
        #clearing the values in the input field
        # st.session_state['1']=""
        # st.session_state['2']=""
        # st.session_state['3']=""
        # st.session_state['4']=""
        # st.session_state['5']=""
        # st.session_state['6']=""
        # st.session_state['7']=""
        # st.session_state['8']=""
        if (prediction==1):
            st.write("The person is Diabetic")
        else:
            st.write("The person is Non-Diabetic")