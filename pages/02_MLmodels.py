import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve

import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.metrics import max_error
from sklearn.ensemble import RandomForestRegressor

# 1. Read data
data = pd.read_csv("data_dropDate.csv")
data=data.drop(['Unnamed: 0'],axis=1)

#--------------
# GUI
st.title("Data Science Project")
st.write("## USA’s Avocado AveragePrice Prediction")

# Hiển thị dữ liệu sau tiền xử lý
df_cleaned=pd.read_csv("df_final_scaled_encoded.csv")

data_new=df_cleaned.drop(['Unnamed: 0'],axis=1)
X=data_new.drop(['AveragePrice'],axis=1)
y=data_new.AveragePrice

# 3. Build model

# Splitting data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

rd_model = RandomForestRegressor()
rd_model.fit(X_train, y_train)

#4. Evaluate model
score_train=rd_model.score(X_train, y_train)
score_test=rd_model.score(X_test, y_test)


#5. Save models
# luu model classication
pkl_filename = "rd_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(rd_model, file)


#6. Load models 
# Đọc model
# import pickle
with open(pkl_filename, 'rb') as file:  
    rd_model = pickle.load(file)

# GUI
menu = ["Business Objective", 'Build Project', 'New Prediction']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write('#### Học viên: Phạm Thuỷ Tú - Nguyễn Thị Trần Lộc',justify="center")
    st.write("""
    ###### Hiện tại: Công ty kinh doanh quả bơ ở rất nhiều vùng của nước Mỹ với 2 loại bơ là bơ thường và bơ hữu cơ, được đóng gói theo nhiều quy chuẩn (Small/Large/XLarge Bags), và có 3 PLU (Product Look Up) khác nhau (4046, 4225, 4770). Nhưng họ chưa có mô hình để dự đoán giá bơ cho việc mở rộng.
    """,justify="center")  
    st.write("""###### Mục tiêu/ Vấn đề: Xây dựng mô hình dự đoán giá trung bình của bơ “Hass” ở Mỹ. Từ đó xem xét việc mở rộng sản xuất, kinh doanh.""",justify="center")
    st.image("avocado.jpeg")
elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Dữ liệu ban đầu")
    st.dataframe(data.head(3))
    st.dataframe(data.tail(3))  
    st.write("##### 2. Dữ liệu sau xử lý - đã scaled và encoded")
    st.dataframe(data_new.head(3))
    st.dataframe(data_new.tail(3))

    st.write("##### 3. Build model...")
    st.write("##### 4. Evaluation")
    st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
        
    y_train_pred = rd_model.predict(X_train)
    y_test_pred = rd_model.predict(X_test)
        
    st.code('Train Set:')
    st.code('RMSE :'+ str( mean_squared_error(y_train, y_train_pred, squared = False)))
    st.code('Max Error: '+ str(max_error(y_train, y_train_pred)))
    st.code('R2-score: '+ str(r2_score(y_train, y_train_pred)))

    st.code('Test Set:')
    st.code('RMSE :'+ str(mean_squared_error(y_test, y_test_pred, squared = False)))
    st.code('Max Error: '+ str( max_error(y_test, y_test_pred)))
    st.code('R2-score: '+ str(r2_score(y_test, y_test_pred)))

    st.write("##### 5. Summary: This model is good enough for AVOCADO classification.")
elif choice=='New Prediction':
    st.subheader("Select data")
        
    lines = None
    st.write("##### Hãy upload file cần dự báo giá")
    # Upload file
    uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
    if uploaded_file_1 is not None:
        lines = pd.read_csv(uploaded_file_1, header=None)
        st.dataframe(lines)
        lines = lines[0]     

    st.write("Content:")
    y_pred_new = rd_model.predict(X_test)

    y_pred_new=pd.DataFrame(y_pred_new, columns=['y_pred_new'])
    df_result=(pd.concat([y_test.reset_index(),y_pred_new],axis=1))
    st.dataframe(df_result[['AveragePrice','y_pred_new']])