import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Generate More Realistic Synthetic Data with COVID-19 Impact for Top Restaurants
def generate_data():
    np.random.seed(42)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5 * 365)]  # Data from 2020 to 2024
    cities = ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot', 'Gandhinagar']
    restaurants = ['Starbucks', 'McDonald', 'Dominos', 'KFC', 'Pizza Hut', 'Subway', 'Barbeque Nation', 'Havmor', 'Burger King', 'Cafe Coffee Day']
    data = []
    
   for city in cities:
        for restaurant in restaurants:
            base_traffic = np.random.randint(500, 2000)  # Base foot traffic for restaurants
            for date in dates:
                temp = np.random.normal(30, 5)  # Temperature centered around 30°C
                rainfall = np.random.choice([0, np.random.uniform(0, 50)], p=[0.7, 0.3])  # 30% chance of rain
                holiday = np.random.choice(["No", "Yes"], p=[0.85, 0.15])
                day_of_week = date.weekday()
                weekend_multiplier = 1.5 if day_of_week >= 5 else 1.0  # Higher foot traffic on weekends
                holiday_multiplier = 1.8 if holiday == "Yes" else 1.0
                weather_impact = -rainfall * 5 + (35 - temp) * 5  # Weather impact on foot traffic
                covid_multiplier = 0.2 if date.year in [2020, 2021] else 1.0  # Reduced traffic during COVID years
                foot_traffic = base_traffic * weekend_multiplier * holiday_multiplier * covid_multiplier + weather_impact
                foot_traffic = max(100, int(foot_traffic))  # Ensure minimum foot traffic
                data.append([date, city, restaurant, round(temp, 2), round(rainfall, 2), holiday, foot_traffic])
    
   df = pd.DataFrame(data, columns=['Date', 'City', 'Restaurant', 'Temperature', 'Rainfall', 'Holiday', 'Foot_Traffic'])
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date column is in datetime format
    return df

data = generate_data()

# Streamlit UI
st.title("Top Restaurant Foot Traffic Analysis in Gujarat (2020-2024, COVID Impact)")

# Sidebar Filters
st.sidebar.header("Filters")
selected_city = st.sidebar.selectbox("Select City", data['City'].unique())
selected_restaurant = st.sidebar.selectbox("Select Restaurant", data['Restaurant'].unique())
start_date = pd.to_datetime(st.sidebar.date_input("Start Date", data['Date'].min()))
end_date = pd.to_datetime(st.sidebar.date_input("End Date", data['Date'].max()))

# Filtered Data
filtered_data = data[(data['City'] == selected_city) & (data['Restaurant'] == selected_restaurant) & (data['Date'] >= start_date) & (data['Date'] <= end_date)]
st.dataframe(filtered_data.head())

# Time Series Analysis
st.subheader("Restaurant Foot Traffic Over Time")
st.write("**Note:** Significant decline in 2020-2021 due to COVID-19 lockdowns and restrictions.")
fig = px.line(filtered_data, x='Date', y='Foot_Traffic', title=f'Foot Traffic in {selected_restaurant} ({selected_city})')
st.plotly_chart(fig)

# Foot Traffic vs. Temperature
st.subheader("Impact of Temperature on Restaurant Foot Traffic")
st.write("Higher temperatures generally reduce foot traffic, but other factors also play a role.")
fig = px.scatter(filtered_data, x='Temperature', y='Foot_Traffic', color='Holiday',
                 title=f'Temperature vs. Foot Traffic in {selected_restaurant} ({selected_city})')
st.plotly_chart(fig)

# Foot Traffic vs. Rainfall
st.subheader("Impact of Rainfall on Restaurant Foot Traffic")
st.write("Increased rainfall can negatively impact foot traffic due to unfavorable conditions.")
fig = px.scatter(filtered_data, x='Rainfall', y='Foot_Traffic', color='Holiday',
                 title=f'Rainfall vs. Foot Traffic in {selected_restaurant} ({selected_city})')
st.plotly_chart(fig)

# Foot Traffic Distribution
st.subheader("Restaurant Foot Traffic Distribution")
st.write("This distribution helps understand how foot traffic varies over time.")
fig = px.histogram(filtered_data, x='Foot_Traffic', nbins=30, title=f'Foot Traffic Distribution in {selected_restaurant} ({selected_city})')
st.plotly_chart(fig)

# Yearly Trends (Highlighting COVID-19 Impact)
st.subheader("Yearly Restaurant Foot Traffic Trends")
filtered_data['Year'] = filtered_data['Date'].dt.year
yearly_traffic = filtered_data.groupby('Year')['Foot_Traffic'].mean().reset_index()
st.write("**Note:** 2020-2021 saw lower traffic due to COVID-19 impact.")
fig = px.line(yearly_traffic, x='Year', y='Foot_Traffic', title='Yearly Foot Traffic Trends')
st.plotly_chart(fig)

# Monthly Trends
st.subheader("Monthly Restaurant Foot Traffic Trends")
filtered_data['Month'] = filtered_data['Date'].dt.month_name()
monthly_traffic = filtered_data.groupby('Month')['Foot_Traffic'].mean().reset_index()
fig = px.line(monthly_traffic, x='Month', y='Foot_Traffic', title='Monthly Foot Traffic Trends')
st.plotly_chart(fig)

# Predictive Modeling
st.subheader("Restaurant Foot Traffic Prediction")
X = filtered_data[['Temperature', 'Rainfall']]
y = filtered_data['Foot_Traffic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display Metrics
st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
st.write(f"R-squared Score: {r2_score(y_test, y_pred):.2f}")

# Predicted vs Actual
prediction_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
st.subheader("Actual vs. Predicted Restaurant Foot Traffic")
fig = px.scatter(prediction_df, x='Actual', y='Predicted', title='Actual vs. Predicted Foot Traffic')
st.plotly_chart(fig)

st.success("Analysis Completed! COVID-19 impact clearly highlighted.")



