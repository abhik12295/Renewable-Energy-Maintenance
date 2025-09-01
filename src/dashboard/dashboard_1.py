# import streamlit as st
# import pandas as pd
# import joblib
# import os
# from prophet import Prophet
# from datetime import timedelta
# from sklearn.preprocessing import StandardScaler
# import numpy as np

# # Set main directory
# main_dir = os.getcwd().rsplit("\\", 0)[0]

# # Yearly directory
# yearly_dir = f"{main_dir}/data/processed/yearly"

# # Optimize data types based on schema
# dtypes = {
#     'date': 'str',
#     'energy_source': 'category',
#     'power_MW': 'float32',
#     'maintenance_status': 'int32',
#     'respondent': 'category',
#     'latitude': 'float32',
#     'longitude': 'float32',
#     'capacity_MW': 'float32',
#     'year': 'int32',
#     'plantcode': 'float32',
#     'plantname': 'category',
#     'nearest_station': 'category',
#     'wind_speed': 'float32',
#     'precipitation': 'float32',
#     'temperature_avg': 'float32',
#     'wind_volatility': 'float32',
#     'sentiment_score': 'float32',
#     'site_density': 'float32',
#     'output_efficiency': 'float32'
# }

# # All columns from schema
# cols = [
#     'date', 'energy_source', 'power_MW', 'maintenance_status', 'respondent',
#     'latitude', 'longitude', 'capacity_MW', 'year', 'plantcode',
#     'plantname', 'nearest_station', 'wind_speed', 'precipitation',
#     'temperature_avg', 'wind_volatility', 'sentiment_score', 'site_density',
#     'output_efficiency'
# ]

# # Cache yearly data loading
# @st.cache_data
# def load_yearly_data(years, energy_source, cols, dtypes, chunksize=50000):
#     filtered_chunks = []
#     for year in years:
#         year_file = f"{yearly_dir}/cleaned_data_{year}.csv"
#         if os.path.exists(year_file):
#             for chunk in pd.read_csv(year_file, usecols=cols, dtype=dtypes, chunksize=chunksize):
#                 chunk['date'] = pd.to_datetime(chunk['date'], format='mixed')
#                 filtered_chunk = chunk[chunk['energy_source'] == energy_source]
#                 if not filtered_chunk.empty:
#                     filtered_chunks.append(filtered_chunk)
#     if filtered_chunks:
#         return pd.concat(filtered_chunks, ignore_index=True)
#     return pd.DataFrame(columns=cols)

# # Load Prophet models and scaler
# prophet_wind = joblib.load(f"{main_dir}\\models\\prophet_wind.pkl")
# scaler = joblib.load(f"{main_dir}\\models\\scaler.pkl")
# prophet_solar = None
# try:
#     prophet_solar = joblib.load(f"{main_dir}\\models\\prophet_solar.pkl")
# except FileNotFoundError:
#     st.warning("Solar Prophet model not found. Solar predictions will be skipped.")

# # Streamlit app
# st.title("Renewable Energy Power Prediction Dashboard")

# # Filters
# energy_source = st.selectbox("Select Energy Source", ['wind', 'solar'])
# horizon = st.selectbox("Select Prediction Horizon", ["30 days", "60 days", "90 days", "6 months"])
# horizon_days = {"30 days": 30, "60 days": 60, "90 days": 90, "6 months": 180}
# days = horizon_days[horizon]

# # Load minimal data for predictions (latest year for regressor averages)
# df = load_yearly_data([2025], energy_source, cols, dtypes)

# # Cache predictions
# @st.cache_data
# def get_predictions(_prophet_model, days, energy_source, df):
#     future = _prophet_model.make_future_dataframe(periods=days, freq='D')
#     for col in ['wind_speed', 'precipitation', 'temperature_avg', 'wind_volatility']:
#         if col in df.columns:
#             future[col] = df[df['energy_source'] == energy_source][col].mean()
#         else:
#             future[col] = 0
#     forecast = _prophet_model.predict(future)
#     future_start = df['date'].max() + timedelta(days=1) if not df.empty else pd.to_datetime('2025-12-31') + timedelta(days=1)
#     forecast = forecast[forecast['ds'] >= future_start][['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'power_MW'})
#     # Inverse-transform power_MW
#     forecast_scaled = pd.DataFrame({
#         'wind_speed': 0, 'precipitation': 0, 'temperature_avg': 0, 'wind_volatility': 0,
#         'power_MW': forecast['power_MW'], 'site_density': 0, 'output_efficiency': 0
#     })
#     forecast['power_MW'] = scaler.inverse_transform(forecast_scaled)[:, 4]
#     return forecast

# # Generate future predictions
# if energy_source == 'wind':
#     prophet_model = prophet_wind
# elif energy_source == 'solar' and prophet_solar:
#     prophet_model = prophet_solar
# else:
#     st.error("No model available for solar predictions.")
#     prophet_model = None

# if prophet_model:
#     forecast = get_predictions(prophet_model, days, energy_source, df)
    
#     # Display daily predictions
#     st.subheader(f"Daily Power Predictions ({horizon})")
#     forecast['date'] = pd.to_datetime(forecast['date'])
#     st.dataframe(forecast.rename(columns={'power_MW': 'Predicted Power (MW)'}), height=300)
    
#     # Plot daily predictions
#     st.line_chart(forecast.set_index('date')['power_MW'])

#     # Download button
#     st.download_button(
#         label="Download Daily Predictions",
#         data=forecast.to_csv(index=False),
#         file_name="daily_forecast.csv",
#         mime="text/csv"
#     )


import streamlit as st
import pandas as pd
import joblib
import os
from prophet import Prophet
from datetime import timedelta, datetime
from sklearn.preprocessing import StandardScaler
import numpy as np

# Set main directory
main_dir = os.getcwd().rsplit("\\", 0)[0]

# MEGA URLs for yearly files and models (replace with your MEGA links)
mega_yearly_urls = {
    2025: "https://mega.nz/file/qB11BBBR#BVHLxZkz8t9p7u7LDHIMd_VLuYP1iSRF8I9eOLdsWKc",
    2024: "https://mega.nz/file/zRVVxbAZ#-fsL0vKTGELkp4_BDEBA7a7zIBKNhQseCaCUiNYjCy0",
    2023: "https://mega.nz/file/LddAQDiY#piVONfhFpKO-S4VzisUPRci7fAWPlmuqUFqk1f3kyk0",
    2022: "https://mega.nz/file/bEFRGIjB#uYWPBdefVCreoksWDUWLCYLzuTVGsmU7sAOPpk0f9ic",
    2021: "https://mega.nz/file/jZt0HapR#6vizmhM3v7flZFnhSpY2vGQDZ3IR9SK7-cjuTo3po6M",
    2020: "https://mega.nz/file/2B8H1K7D#U7eowd9rwrYd1WRtMEMirJGy41GXMOim0s3PNzH_X3I",
    2019: "https://mega.nz/file/vJUACYaJ#vDs8M8WlvESuMEgbjPdNDGYorOyaPoSL_UmjMnV3Lrg",

    # Add other years if needed
}
mega_models = {
    'prophet_wind': "https://mega.nz/file/PV0BVaga#pHLO5hHdzQv2w2dXOa6rgIXHA_OHTUxOUkW-AH7AS-E",
    'scaler': "https://mega.nz/file/6JkSCSiL#OFTtyxh3xqrByxoB70pNkPkGKoZkgj2oWBVOV8blHHM",
    'prophet_solar': "https://mega.nz/file/rM0BWBTD#EzWH0r9lzwQSlibcuwzvPw7qqpL7TM7Y5MlP62LQFX8"
}

# Optimize data types based on schema
dtypes = {
    'date': 'str',
    'energy_source': 'category',
    'power_MW': 'float32',
    'maintenance_status': 'int32',
    'respondent': 'category',
    'latitude': 'float32',
    'longitude': 'float32',
    'capacity_MW': 'float32',
    'year': 'int32',
    'plantcode': 'float32',
    'plantname': 'category',
    'nearest_station': 'category',
    'wind_speed': 'float32',
    'precipitation': 'float32',
    'temperature_avg': 'float32',
    'wind_volatility': 'float32',
    'sentiment_score': 'float32',
    'site_density': 'float32',
    'output_efficiency': 'float32'
}

# All columns from schema
cols = [
    'date', 'energy_source', 'power_MW', 'maintenance_status', 'respondent',
    'latitude', 'longitude', 'capacity_MW', 'year', 'plantcode',
    'plantname', 'nearest_station', 'wind_speed', 'precipitation',
    'temperature_avg', 'wind_volatility', 'sentiment_score', 'site_density',
    'output_efficiency'
]

# Cache yearly data loading from MEGA
@st.cache_data
def load_yearly_data(years, energy_source, cols, dtypes, chunksize=50000):
    filtered_chunks = []
    for year in years:
        url = mega_yearly_urls.get(year)
        if url:
            for chunk in pd.read_csv(url, usecols=cols, dtype=dtypes, chunksize=chunksize):
                chunk['date'] = pd.to_datetime(chunk['date'], format='mixed')
                filtered_chunk = chunk[chunk['energy_source'] == energy_source]
                if not filtered_chunk.empty:
                    filtered_chunks.append(filtered_chunk)
    if filtered_chunks:
        return pd.concat(filtered_chunks, ignore_index=True)
    return pd.DataFrame(columns=cols)

# Load Prophet models and scaler from MEGA
prophet_wind = joblib.load(mega_models['prophet_wind'])
scaler = joblib.load(mega_models['scaler'])
prophet_solar = None
try:
    prophet_solar = joblib.load(mega_models['prophet_solar'])
except Exception as e:
    st.warning("Solar Prophet model not found. Solar predictions will be skipped.")

# Streamlit app
st.title("Renewable Energy Power Prediction Dashboard")

# Filters
energy_source = st.selectbox("Select Energy Source", ['wind', 'solar'])
horizon = st.selectbox("Select Prediction Horizon", ["30 days", "60 days", "90 days", "6 months"])
horizon_days = {"30 days": 30, "60 days": 60, "90 days": 90, "6 months": 180}
days = horizon_days[horizon]

# Current date (September 01, 2025)
current_date = pd.to_datetime('2025-09-01')

# Historical data from past 2 months (July 01, 2025 to August 31, 2025)
past_start = current_date - timedelta(days=60)
past_end = current_date - timedelta(days=1)

# Load historical data for past 2 months (2025 only)
df = load_yearly_data([2025], energy_source, cols, dtypes)
df = df[(df['date'] >= past_start) & (df['date'] <= past_end)]

# Cache predictions
@st.cache_data
def get_predictions(prophet_model, days, energy_source, df):
    future = prophet_model.make_future_dataframe(periods=days, freq='D')
    for col in ['wind_speed', 'precipitation', 'temperature_avg', 'wind_volatility']:
        if col in df.columns:
            future[col] = df[df['energy_source'] == energy_source][col].mean()
        else:
            future[col] = 0
    forecast = prophet_model.predict(future)
    future_start = current_date
    forecast = forecast[forecast['ds'] >= future_start][['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'power_MW'})
    # Inverse-transform power_MW
    forecast_scaled = pd.DataFrame({
        'wind_speed': 0, 'precipitation': 0, 'temperature_avg': 0, 'wind_volatility': 0,
        'power_MW': forecast['power_MW'], 'site_density': 0, 'output_efficiency': 0
    })
    forecast['power_MW'] = scaler.inverse_transform(forecast_scaled)[:, 4]
    return forecast

# Generate future predictions
if energy_source == 'wind':
    prophet_model = prophet_wind
elif energy_source == 'solar' and prophet_solar:
    prophet_model = prophet_solar
else:
    st.error("No model available for solar predictions.")
    prophet_model = None

if prophet_model:
    forecast = get_predictions(prophet_model, days, energy_source, df)
    
    # Combine historical and predictions
    historical = df[['date', 'power_MW']].rename(columns={'power_MW': 'power_MW'})
    forecast = forecast[['date', 'power_MW']].rename(columns={'power_MW': 'predicted_power_MW'})
    combined = pd.concat([historical, forecast], ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])
    
    # Display combined data
    st.subheader(f"Daily Power Predictions from {current_date.date()} ({horizon})")
    st.dataframe(combined.rename(columns={'power_MW': 'Historical Power (MW)', 'predicted_power_MW': 'Predicted Power (MW)'}))
    
    # Plot combined
    st.line_chart(combined.set_index('date'))

    # Download button
    st.download_button(
        label="Download Predictions",
        data=combined.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )