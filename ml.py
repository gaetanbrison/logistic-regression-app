import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt
import plotly.express as px
import streamlit as st
import random
from PIL import Image
import altair as alt
from htbuilder import HtmlElement, div, hr, a, p, img, styles
from htbuilder.units import percent, px
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report



data_url = "http://lib.stat.cmu.edu/datasets/boston" 


# data = "C:\Users\DELL\Desktop\streamlit\images\data-processing.png"

# setting up the page streamlit

st.set_page_config(
    page_title="Logistic Regression App ", layout="wide", page_icon="./images/nyu.png"
)


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded



def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()

image_nyu = Image.open('images/nyu.png')
st.image(image_nyu, width=100)

st.title("Logistic Regression Laboratory 🧪")

# navigation dropdown

st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('🔎 Select Page',['Introduction 🚀','Data 💽','Data Cleanup 🧹','Visualization 📊','Prediction 📈'])
select_dataset =  st.sidebar.selectbox('💽 Select Dataset',["Advertising"])
select_model = st.sidebar.selectbox('🦾 Select Model',["Logistic Regression"])
df = pd.read_csv("advertising.csv")

list_variables = df.columns
select_variable =  st.sidebar.selectbox('🎯 Select Variable to Predict',list_variables)
# page 1 
if app_mode == 'Introduction 🚀':
    image_header = Image.open('./images/logistic_regression.png')
    st.image(image_header, width=400)

    st.markdown("### 00 What is logistic regression used for ❓")
    
    if st.button('Answer1'):
        image_binary = Image.open('./images/binary.png')
        st.image(image_binary, width=100)
        st.write('The logistic regression is used for classification tasks 💡')

    st.markdown("### 01 Applications for logistic regression ❓")
    if st.button('Answer2'):
        image_credit = Image.open('./images/credit-scoring.png')
        st.write('**Credit Industry**')
        st.image(image_credit, width=150)


        image_health = Image.open('./images/cancer-detection.png')
        st.write('**Healthcare Industry**')
        st.image(image_health, width=150)


        image_hotel = Image.open('./images/cancel-booking.png')
        st.write('**Hotel Industry**')
        st.image(image_hotel, width=150)


    st.markdown("### 02 Difference between logistic and linear regression ❓")
    if st.button('Answer3'):
        st.write('**Linear Regression**')
        st.markdown("![Alt Text](https://miro.medium.com/v2/resize:fit:4800/1*QHqi36j1qdNYHtTAsQcxhw.gif)")
        #image_lin = Image.open('')
        #st.image(image_lin, width=150)

        st.write('**Logistic Regression**')
        st.markdown("![Alt Text](https://miro.medium.com/v2/resize:fit:640/1*n2Yhin53lFn-7xloKg_sfQ.gif)")


if app_mode == 'Data 💽':    
    image_header = Image.open('./images/log.png')
    st.image(image_header, width=200)


    st.markdown("### 00 - Show  Dataset")
    if select_dataset == "Advertising":


        st.markdown("#### Advertising Dataset")
        st.write("This data set contains the following features:")
        st.write("---")
        st.write("**Daily Time Spent on Site**: consumer time on site in minutes")
        st.write("**Age**: cutomer age in years")
        st.write("**Area Income**: Avg. Income of geographical area of consumer")
        st.write("**Daily Internet Usage**: Avg. minutes a day consumer is on the internet")
        st.write("**Ad Topic Line**: Headline of the advertisement")
        st.write("**City**: City of consumer")
        st.write("**Male**: Whether or not consumer was male")
        st.write("**Country**: Country of consumer")
        st.write("**Timestamp**: Time at which consumer clicked on Ad or closed window")
        st.write("**Clicked on Ad**: 0 or 1 indicated clicking on Ad")
        st.write("---")
    num = st.number_input('No. of Rows', 5, 10)
    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
    if head == 'Head':
        st.dataframe(df.head(num))
    else:
        st.dataframe(df.tail(num))

    code = '''

    df.head(5)


    '''
    st.code(code, language='python')
    
    st.markdown("Number of rows and columns helps us to determine how large the dataset is.")
    st.text('(Rows,Columns)')
    st.write(df.shape)


    st.markdown("### 01 - Description")
    st.dataframe(df.describe())

    code2 = '''

    df.describe()

    '''
    st.code(code2, language='python')

    st.markdown("### 02 - Missing Values")
    st.markdown("Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")
    dfnull = df.isnull().sum()/len(df)*100
    totalmiss = dfnull.sum().round(2)
    st.write("Percentage of total missing values:",totalmiss)
    st.write(dfnull)
    if totalmiss <= 30:
        st.success("Looks good! as we have less then 30 percent of missing values.")
    else:
        st.warning("Poor data quality due to greater than 30 percent of missing value.")
        st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")

    code3 = '''
    ## calculate missing values
    dfnull = df.isnull().sum()/len(df)*100
    totalmiss = dfnull.sum().round(2) 
    '''
    st.code(code3, language='python')


    st.markdown("### 03 - Completeness")
    st.markdown(" Completeness is defined as the ratio of non-missing values to total records in dataset.") 
    # st.write("Total data length:", len(df))
    nonmissing = (df.notnull().sum().round(2))
    completeness= round(sum(nonmissing)/len(df),2)
    st.write("Completeness ratio:",completeness)
    st.write(nonmissing)
    if completeness >= 0.80:
        st.success("Looks good! as we have completeness ratio greater than 0.85.")
           
    else:
        st.success("Poor data quality due to low completeness ratio( less than 0.85).")
        pr = df.profile_report()
        st_profile_report(pr)
    code5 = '''
    ## data completeness
    nonmissing = (df.notnull().sum().round(2))
    completeness= round(sum(nonmissing)/len(df),2)
    '''
    st.code(code5, language='python')
    st.markdown("### 04 - Complete Report")
    if st.button("Generate Report"):

        pr = df.profile_report()
        st_profile_report(pr)
        code5 = '''
        ## data report 
        pr = df.profile_report()
        st_profile_report(pr)
        '''
        st.code(code5, language='python')

if app_mode == 'Data Cleanup 🧹':
    st.markdown("## Data Cleanup 🧹")

    st.write("Remove Duplicates ➡️")
    code6 = '''
    df.drop_duplicates()
    '''
    st.code(code6, language='python')

    st.write("Remove Missing Values ➡️")
    code7 = '''
    df.dropna()
    '''
    st.code(code7, language='python')

    st.write("Convert text to categorical integers ➡️")
    code6 = '''
    df['col1'] = df['col1'].factorize()[0]
    '''
    st.code(code6, language='python')


if app_mode == 'Visualization 📊':
    st.markdown("## Visualization")
    df_new = df.copy()
    df_new['Country'] = df_new['Country'].factorize()[0]
    df_new['Ad Topic Line'] = df_new['Ad Topic Line'].factorize()[0]
    df_new = df_new[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage', 'Ad Topic Line', 'Male', 'Country', 'Clicked on Ad']]
    list_variables2 = df_new.columns
    symbols = st.multiselect("Select two variables",list_variables2,["Age","Area Income"] )
    width1 = st.sidebar.slider("plot width", 1, 25, 10)
    #symbols = st.multiselect("", list_variables, list_variables[:5])
    tab1, tab2= st.tabs(["Line Chart","📈 Correlation"])    

    tab1.subheader("Line Chart")
    st.line_chart(data=df, x=symbols[0],y=symbols[1], width=0, height=0, use_container_width=True)
    st.write(" ")
    st.bar_chart(data=df, x=symbols[0], y=symbols[1], use_container_width=True)

    tab2.subheader("Correlation Tab 📉")
    fig,ax = plt.subplots(figsize=(width1, width1))
    sns.heatmap(df.corr(),cmap= sns.cubehelix_palette(8),annot = True, ax=ax)
    tab2.write(fig)


    st.write(" ")
    st.write(" ")
    st.markdown("### Pairplot")

    df2 = df[[list_variables[0],list_variables[1],list_variables[2],list_variables[3],list_variables[4]]]
    fig3 = sns.pairplot(df2)
    st.pyplot(fig3)




if app_mode == 'Prediction 📈':
    st.markdown("## Prediction")
    df_new = df.copy()
    df_new['Country'] = df_new['Country'].factorize()[0]
    df_new['Ad Topic Line'] = df_new['Ad Topic Line'].factorize()[0]
    df_new = df_new[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage', 'Ad Topic Line', 'Male', 'Country', 'Clicked on Ad']]
    list_variables2 = df_new.columns
    train_size = st.sidebar.number_input("Train Set Size", min_value=0.00, step=0.01, max_value=1.00, value=0.70)
    new_df= df.drop(labels=select_variable, axis=1)  #axis=1 means we drop data by columns
    list_var = new_df.columns
    output_multi = st.multiselect("Select Explanatory Variables", list_variables2,["Age","Area Income"])

        #independent variables / explanatory variables
        #choosing column for target
    new_df2 = df_new[output_multi]
    x =  new_df2
    y = df_new["Clicked on Ad"]
    col1,col2 = st.columns(2)
    col1.subheader("Feature Columns top 25")
    col1.write(x.head(25))
    col2.subheader("Target Column top 25")
    col2.write(y.head(25))
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=train_size)
    lm = LogisticRegression()
    lm.fit(X_train,y_train)
    predictions = lm.predict(X_test)
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    # Calculate accuracy score of the model
    accuracy = np.round(accuracy_score(y_test, predictions)*100,3)



    # Print the accuracy score
    st.write("🎯Accuracy % :", accuracy)


    st.markdown("### Reminder on how we calculate accuracy: ")
    image_conf1 = Image.open('./images/conf1.png')
    st.image(image_conf1, width=600)
    image_conf2 = Image.open('./images/conf2.png')
    st.image(image_conf2, width=600)




    #st.write("1) The model explains,", np.round(mt.explained_variance_score(y_test, predictions)*100,2),"% variance of the target feature")
    #st.write("2) The Mean Absolute Error of model is:", np.round(mt.mean_absolute_error(y_test, predictions ),2))
    #st.write("3) MSE: ", np.round(mt.mean_squared_error(y_test, predictions),2))
    #st.write("4) The R-Square score of the model is " , np.round(mt.r2_score(y_test, predictions),2))




if __name__=='__main__':
    main()

st.markdown(" ")
st.markdown("### 👨🏼‍💻 **App Contributors:** ")
st.image(['images/gaetan.png'], width=100,caption=["Gaëtan Brison"])

st.markdown(f"####  Link to Project Website [here]({'https://github.com/NYU-DS-4-Everyone/Linear-Regression-App'}) 🚀 ")
st.markdown(f"####  Feel free to contribute to the app and give a ⭐️")


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;background - color: white}
     .stApp { bottom: 80px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1,

    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer2():
    myargs = [
        "👨🏼‍💻 Made by ",
        link("https://github.com/NYU-DS-4-Everyone", "NYU - Professor Gaëtan Brison"),
        "🚀"
    ]
    layout(*myargs)


if __name__ == "__main__":
    footer2()
