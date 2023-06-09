import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from datetime import datetime

from itertools import combinations
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA as ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
pd.options.display.float_format = '{:.2f}'.format

import pickle
import streamlit as st
from sklearn import metrics

# 1. Read data
data = pd.read_csv("df_remove_region.csv")
data["Date"] = pd.to_datetime(data["Date"])
data=data.drop(['Unnamed: 0'],axis=1)

#--------------
# GUI
st.title("Data Science Project")
st.write("## USA’s Avocado AveragePrice Prediction")

# Chọn LosAngeles để dự đoán
df_LosAngeles = data[data.region=='LosAngeles']
# Tách thành hai tập con theo từng loại Conventional và Organic
df_organic = df_LosAngeles[(df_LosAngeles.type=="organic")].groupby(['Date'])['AveragePrice'].mean()
df_organic = df_organic.resample('W').mean()
df_organic = df_organic.to_frame(name='AveragePrice')

df_conventional = df_LosAngeles[(df_LosAngeles.type=="conventional")].groupby(['Date'])['AveragePrice'].mean()
df_conventional = df_conventional.resample('W').mean()
df_conventional = df_conventional.to_frame(name='AveragePrice')

# Xử lý tính dừng cho bộ dữ liệu
df_organic_diff= df_organic['AveragePrice'].diff()
df_organic_diff = df_organic_diff.dropna()
df_organic_diff=df_organic_diff.to_frame()

df_conventional_diff= df_conventional['AveragePrice'].diff()
df_conventional_diff = df_conventional_diff.dropna()
df_conventional_diff=df_conventional_diff.to_frame()

# 3. Build model

# Splitting data into training and test set
train_size = int(len(df_organic_diff) * 0.75)
train_organic, test_organic = df_organic_diff[0:train_size], df_organic_diff[train_size:]

train_size = int(len(df_conventional_diff) * 0.75)
train_conventional, test_conventional = df_conventional_diff[0:train_size], df_conventional_diff[train_size:]

import time
start_time_organic=time.time()

model_sarimax_organic = sm.tsa.statespace.SARIMAX(train_organic,order = (1,1,0),seasonal_order = (1,1,1,52))
model_fit_sarimax_organic = model_sarimax_organic.fit()

end_time_organic=time.time()
total_time_organic=end_time_organic-start_time_organic

start_time_conventional=time.time()

model_sarimax_conventional = sm.tsa.statespace.SARIMAX(train_conventional,order = (1,1,0),seasonal_order = (1,1,1,52))
model_fit_sarimax_conventional = model_sarimax_conventional.fit()

end_time_conventional=time.time()
total_time_conventional=end_time_conventional-start_time_conventional

#4. Evaluate model

# GUI
menu = ["Business Objective", 'Build Project', 'New Prediction']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write('#### Học viên: Phạm Thuỷ Tú - Nguyễn Thị Trần Lộc')
    st.write("""
    ###### Hiện tại: Công ty kinh doanh quả bơ ở rất nhiều vùng của nước Mỹ với 2 loại bơ là bơ thường và bơ hữu cơ, được đóng gói theo nhiều quy chuẩn (Small/Large/XLarge Bags), và có 3 PLU (Product Look Up) khác nhau (4046, 4225, 4770). Nhưng họ chưa có mô hình để dự đoán giá bơ cho việc mở rộng.
    """)  
    st.write("""###### Mục tiêu/ Vấn đề: Xây dựng mô hình dự đoán giá trung bình của bơ “Hass” ở Mỹ. Từ đó xem xét việc mở rộng sản xuất, kinh doanh.""")
    st.image("avocado.jpeg")
elif choice == 'Build Project':

    st.subheader("Build Project")
    st.write("##### 1. Dữ liệu ban đầu")
    st.dataframe(data.head(3))
    st.dataframe(data.tail(3))  

    st.write("##### 2. Dữ liệu LosAngeles")
    st.dataframe(df_LosAngeles.head(3))
    st.dataframe(df_LosAngeles.tail(3))

    st.write("##### 3.1. Dữ liệu LosAngeles loại Organic")
    st.dataframe(df_organic.head(3))
    st.write("##### 3.2. Dữ liệu LosAngeles loại Conventional")
    st.dataframe(df_conventional.head(3))

    # Xem thông tin tổng quan
    st.write('Thống kê mô tả của dữ liệu Bơ Organic tại TP LosAngeles')
    st.write(df_organic.describe().T)
    st.write('Thống kê mô tả của dữ liệu Bơ Conventional tại TP LosAngeles')
    st.write(df_organic.describe().T)

    st.write("##### 4. Build model...")
    st.code('Thời gian thực hiện model trên bơ Organic tại LosAngeles '+str(total_time_organic))
    st.write('Tham số tối ưu trong mô hình dự đoán trên bơ Organic tại LosAngeles')
    st.write(model_fit_sarimax_organic.summary())
    st.code('Thời gian thực hiện model trên bơ Conventional tại LosAngeles '+str(total_time_conventional))

    st.write("##### 5. Evaluation")
    pred_organic = model_fit_sarimax_organic.predict()
    pred_conventional = model_fit_sarimax_conventional.predict()
    
    st.write('Độ chính xác dự đoán trên dữ liệu Bơ Organic tại LosAngeles') 
    st.code('RMSE :'+ str( np.sqrt(mean_squared_error(train_organic, pred_organic))))
    st.code('MSE :'+ str( mean_squared_error(train_organic, pred_organic)))
    mae_p = mean_absolute_error(train_organic, pred_organic)
    st.code('MAE: %.3f' % mae_p)

    st.write('Độ chính xác dự đoán trên dữ liệu Bơ Conventional tại LosAngeles')
    st.code('RMSE :'+ str( np.sqrt(mean_squared_error(train_conventional, pred_conventional))))
    st.code('MSE :'+ str( mean_squared_error(train_conventional, pred_conventional)))
    mae_p = mean_absolute_error(train_conventional, pred_conventional)
    st.code('MAE: %.3f' % mae_p)

    fig, ax = plt.subplots()       
    ax.plot(train_organic)
    ax.plot(model_fit_sarimax_organic.fittedvalues, color='red');
    st.pyplot(fig)

    st.write("##### 5. Summary: This model is good enough for AVOCADO in LosAngeles")

elif choice=='New Prediction':
    st.write("Get forecast of 20 weeks ahead in future:")

    pred_organic = model_fit_sarimax_organic.get_prediction(start = 169, end = 189)
    pred_ci_organic = pred_organic.conf_int()
    pred_ci_organic

    pred_conventional = model_fit_sarimax_conventional.get_prediction(start = 169, end = 189)
    pred_ci_conventional = pred_conventional.conf_int()
    pred_ci_conventional

    forecast_organic = model_fit_sarimax_organic.get_forecast(steps=20)
    forecast_ci_organic = forecast_organic.conf_int()
    forecast_ci_organic

    forecast_conventional = model_fit_sarimax_conventional.get_forecast(steps=20)
    forecast_ci_conventional = forecast_conventional.conf_int()
    forecast_ci_conventional
    