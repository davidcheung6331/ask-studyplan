import subprocess
import streamlit as st
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd
import os
from PIL import Image


st.set_page_config(
    page_title="📝 Summary Study Plan",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
image = Image.open("study.png")
st.image(image, caption='created by MJ')




st.title("📝 :blue[Summary your Study Plan]")

# Set API keys
system_openai_api_key = os.environ.get('OPENAI_API_KEY')
system_openai_api_key = st.text_input(":key: Enter your OpenAI Key :", value=system_openai_api_key)
os.environ["OPENAI_API_KEY"] = system_openai_api_key


q1 =  "Question: How many subjects in study plan ?"
q2  = "Question: How many stack Type in study plan ?"
q3 = "Question: list out the subjects by stack type in table format? "
q4 = "Question: Calucate the total week of study-week  by stack type in table format?"
q5 = "Question: calucate the total week of study-week ?"

openai_key = os.getenv("OPENAI_API_KEY")



col1, col2  = st.columns(2)

with col1:
    try:
        st.subheader("Step 1. 📤 Upload Sample study.csv")
        uploaded_file = st.file_uploader('Upload a file')
        df = pd.read_csv(uploaded_file)
        
        # st.write(df.head(5))
        st.write(df)
        agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
    except OSError as err:
        st.warning("OS error:", err)
    except ValueError:
        st.warning("Could not convert data to an integer.")
    except Exception as err:
        st.warning(f"Unexpected {err=}, {type(err)=}")


with col2:
    st.subheader("Option 1. 👇🏻 Select Prompt Text to query: ")
    if st.button(q1):
        st.subheader("2. Click button below for enquiry the CSV")
        A1 = agent.run(q1)
        subprocess.call(["say", q1])
        st.info(A1)
        subprocess.call(["say", A1])

    if st.button(q2):
        A2 = agent.run(q2)
        subprocess.call(["say", q2])
        st.info(A2)
        subprocess.call(["say", A2])

    if st.button(q3):
        A3 = agent.run(q3)
        subprocess.call(["say", q3])
        st.info(A3)
        subprocess.call(["say", A3])

    if st.button(q4):
        A4 = agent.run(q4)
        subprocess.call(["say", q2])
        st.info(A4)
        subprocess.call(["say", A4])


    if st.button(q5):
        A5 = agent.run(q5)
        subprocess.call(["say", q5])
        st.info(A5)
        subprocess.call(["say", A5])

    st.subheader("Option 2. 👇🏻 Enter your Prompt Text to query: ")
    query = st.text_input("Your Prompt Text", value=q5)  
    if st.button("Query"):
        result = agent.run(query)
        st.info(result)
        subprocess.call(["say", result])
    

log = """

 try:
        st.subheader("1. Upload study.csv file to dataFrame agent ")
        uploaded_file = st.file_uploader('Upload a file')
        df = pd.read_csv(uploaded_file)
        st.write(df)
        agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
        ....
    
    
    query = st.text_input("enter your query", value="how many row in this file?")  
    if st.button("Submit Query"):
        result = agent.run(query)
        st.info(result)
        subprocess.call(["say", result])

> Entering new AgentExecutor chain...
Thought: I need to count the number of subjects
Action: python_repl_ast
Action Input: len(df['subject'].unique())
Observation: 31
Thought: I now know the final answer
Final Answer: There are 31 subjects in the study plan.

> Finished chain.


> Entering new AgentExecutor chain...
Thought: I need to count the stack-type
Action: python_repl_ast
Action Input: df['stack-type'].nunique()
Observation: 4
Thought: I now know the final answer
Final Answer: There are 4 stack-type in the study plan.

> Finished chain.


> Entering new AgentExecutor chain...
Thought: I need to group the data by stack-type and then sum the study-week
Action: python_repl_ast
Action Input: df.groupby('stack-type')['Study-Week'].sum()
Observation: stack-type
All                 4
Backend            40
Front              83
Front & Backend    32
Name: Study-Week, dtype: int64
Thought: I now know the final answer
Final Answer: The total week of study-week by stack-type is 4 for All, 40 for Backend, 83 for Front, and 32 for Front & Backend.

> Finished chain.


> Entering new AgentExecutor chain...
Thought: I need to sum up the values in the study-week column
Action: python_repl_ast
Action Input: df['Study-Week'].sum()
Observation: 159
Thought: I now know the final answer
Final Answer: The total week of study-week is 159.

> Finished chain.
"""

with st.expander("explanation"):
    st.code(log)





 
