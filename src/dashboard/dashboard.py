# import streamlit as st
# import pandas as pd
# import folium
# from streamlit_folium import st_folium
# import joblib
# import os

# main_dir = os.getcwd().rsplit("\\", 0)[0]
# print('main_dir:', main_dir)

# # Only load necessary columns
# cols = ['date', 'energy_source', 'power_MW', 'latitude', 'longitude', 'output_efficiency', 'plantname']
# df = pd.read_csv(f"{main_dir}\\data\\processed\\cleaned_data_final.csv", usecols=cols)
# df['date'] = pd.to_datetime(df['date'], format='mixed')
# prophet_model = joblib.load(f"{main_dir}\\models\\prophet_wind.pkl")

# st.title("Renewable Energy Predictive Maintenance Dashboard")
# energy_source = st.selectbox("Select Energy Source", ['wind', 'solar'])
# date_range = st.date_input("Select Date Range", [df['date'].min(), df['date'].max()])

# # Filter data
# filtered_df = df[(df['energy_source'] == energy_source) & 
#                  (df['date'].dt.date >= date_range[0]) & 
#                  (df['date'].dt.date <= date_range[1])]

# # Limit number of rows for plotting and mapping
# MAX_ROWS = 1000
# if len(filtered_df) > MAX_ROWS:
#     filtered_df = filtered_df.sample(MAX_ROWS, random_state=42)

# st.line_chart(filtered_df.set_index('date')['power_MW'])

# # Geospatial map (limit markers)
# m = folium.Map(location=[31.9686, -99.9018], zoom_start=6)
# for idx, row in filtered_df.drop_duplicates(['latitude', 'longitude']).iterrows():
#     folium.CircleMarker(
#         [row['latitude'], row['longitude']],
#         radius=row['output_efficiency'] * 10,
#         popup=f"{row['plantname']} - {row['energy_source']}",
#         color='blue' if row['energy_source'] == 'wind' else 'orange',
#         fill=True
#     ).add_to(m)

# st_folium(m, width=700, height=500)


import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib
import os
from prophet import Prophet
from datetime import timedelta

# Set main directory
main_dir = os.getcwd().rsplit("\\", 0)[0]

# Load necessary columns (include regressors)
cols = ['date', 'energy_source', 'power_MW', 'latitude', 'longitude', 'output_efficiency', 'plantname', 
        'wind_speed', 'precipitation', 'temperature_avg', 'wind_volatility']
df = pd.read_csv(f"{main_dir}\\data\\processed\\cleaned_data_final.csv", usecols=cols)
df['date'] = pd.to_datetime(df['date'], format='mixed')

# Load Prophet models
prophet_wind = joblib.load(f"{main_dir}\\models\\prophet_wind.pkl")
prophet_solar = None
try:
    prophet_solar = joblib.load(f"{main_dir}\\models\\prophet_solar.pkl")
except FileNotFoundError:
    st.warning("Solar Prophet model not found. Solar predictions will be skipped.")

# Streamlit app
st.title("Renewable Energy Predictive Maintenance Dashboard")

# Filters
energy_source = st.selectbox("Select Energy Source", ['wind', 'solar'])
date_range = st.date_input("Select Date Range", [df['date'].min(), df['date'].max()])

# Filter historical data
filtered_df = df[(df['energy_source'] == energy_source) & 
                (df['date'].dt.date >= date_range[0]) & 
                (df['date'].dt.date <= date_range[1])]

# Limit rows for plotting
MAX_ROWS = 1000
if len(filtered_df) > MAX_ROWS:
    filtered_df = filtered_df.sample(MAX_ROWS, random_state=42)

# Historical time series
st.subheader(f"{energy_source.capitalize()} Power Output (Historical)")
st.line_chart(filtered_df.set_index('date')['power_MW'])

# Geospatial map
st.subheader(f"{energy_source.capitalize()} Plant Locations")
m = folium.Map(location=[31.9686, -99.9018], zoom_start=6)  # Texas center
for idx, row in filtered_df.drop_duplicates(['latitude', 'longitude']).iterrows():
    folium.CircleMarker(
        [row['latitude'], row['longitude']],
        radius=row['output_efficiency'] * 10,
        popup=f"{row['plantname']} - {row['energy_source']}",
        color='blue' if row['energy_source'] == 'wind' else 'orange',
        fill=True
    ).add_to(m)
st_folium(m, width=700, height=500)

# Prediction horizon selector
st.subheader("Power Output Predictions")
horizon = st.selectbox("Select Prediction Horizon", ["30 days", "60 days", "90 days", "6 months"])
horizon_days = {"30 days": 30, "60 days": 60, "90 days": 90, "6 months": 180}
days = horizon_days[horizon]

# Generate future predictions
if energy_source == 'wind':
    prophet_model = prophet_wind
elif energy_source == 'solar' and prophet_solar:
    prophet_model = prophet_solar
else:
    st.error("No model available for solar predictions.")
    prophet_model = None

if prophet_model:
    # Create future dataframe
    future = prophet_model.make_future_dataframe(periods=days, freq='D')
    # Add regressors using historical averages
    for col in ['wind_speed', 'precipitation', 'temperature_avg', 'wind_volatility']:
        if col in df.columns:
            future[col] = df[df['energy_source'] == energy_source][col].mean()
        else:
            st.warning(f"Regressor {col} missing from data. Using 0 as fallback.")
            future[col] = 0
    forecast = prophet_model.predict(future)
    
    # Filter to future dates only
    future_start = df['date'].max() + timedelta(days=1)
    forecast = forecast[forecast['ds'] >= future_start][['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'power_MW'})
    
    # Aggregate to monthly
    forecast['date'] = pd.to_datetime(forecast['date'])
    monthly_forecast = forecast.groupby(forecast['date'].dt.to_period('M')).agg({'power_MW': 'mean'}).reset_index()
    monthly_forecast['date'] = monthly_forecast['date'].dt.to_timestamp()

    # Display monthly predictions
    st.subheader(f"Monthly Power Predictions ({horizon})")
    st.dataframe(monthly_forecast.rename(columns={'power_MW': 'Predicted Power (MW)'}))
    
    # Plot monthly predictions
    st.line_chart(monthly_forecast.set_index('date')['power_MW'])

# Save filtered data for debugging
#filtered_df.to_csv(f"{main_dir}\\data\\processed\\filtered_dashboard_data.csv", index=False)cl