# running best
# import streamlit as st
# import pandas as pd
# import joblib
# import sqlite3
# import os
# from prophet import Prophet
# from datetime import timedelta, datetime
# from sklearn.preprocessing import StandardScaler
# import numpy as np

# # Set main directory and database path
# main_dir = os.getcwd().rsplit("\\", 0)[0]
# db_path = f"{main_dir}\\data\\energy_data.db"

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

# # Cache data loading from SQLite
# @st.cache_data
# def load_yearly_data(energy_source, start_date, end_date, cols, dtypes, _scaler):
#     try:
#         conn = sqlite3.connect(db_path)
#         query = """
#         SELECT *
#         FROM energy
#         WHERE energy_source = ? AND date >= ? AND date <= ?
#         """
#         df = pd.read_sql_query(query, conn, params=(energy_source, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
#         conn.close()
#         if df.empty:
#             st.error(f"No data found for {energy_source} between {start_date.date()} and {end_date.date()}. Check {db_path}.")
#             return pd.DataFrame(columns=cols)
#         # Apply dtypes dynamically
#         for col, dtype in dtypes.items():
#             if col in df.columns:
#                 df[col] = df[col].astype(dtype, errors='ignore')
#         # Fill NaN in numerical columns
#         for col in ['wind_speed', 'precipitation', 'temperature_avg', 'wind_volatility']:
#             if col in df.columns:
#                 df[col] = df[col].fillna(df[col].mean() if not df[col].isna().all() else 0)
#         # Destandardize power_MW
#         if 'power_MW' in df.columns and _scaler is not None:
#             try:
#                 scaled_data = pd.DataFrame({
#                     'wind_speed': df['wind_speed'].fillna(0),
#                     'precipitation': df['precipitation'].fillna(0),
#                     'temperature_avg': df['temperature_avg'].fillna(0),
#                     'wind_volatility': df['wind_volatility'].fillna(0),
#                     'power_MW': df['power_MW'],
#                     'site_density': df['site_density'].fillna(0),
#                     'output_efficiency': df['output_efficiency'].fillna(0)
#                 })
#                 df['power_MW'] = _scaler.inverse_transform(scaled_data)[:, 4]
#             except Exception as e:
#                 st.error(f"Failed to destandardize historical power_MW: {str(e)}")
#         # Log diagnostics
#         # st.write(f"Sample data columns: {df.columns.tolist()}")
#         # st.write(f"NaN counts: {df.isna().sum().to_dict()}")
#         # st.write(f"Row count: {len(df)}")
#         return df
#     except Exception as e:
#         st.error(f"Failed to load data from SQLite: {str(e)}")
#         return pd.DataFrame(columns=cols)

# # Load Prophet models and scaler
# try:
#     prophet_wind = joblib.load(f"{main_dir}\\models\\prophet_wind.pkl")
# except FileNotFoundError:
#     st.error("Wind Prophet model not found. Please ensure prophet_wind.pkl exists.")
#     prophet_wind = None
# try:
#     scaler = joblib.load(f"{main_dir}\\models\\scaler.pkl")
# except FileNotFoundError:
#     st.error("Scaler model not found. Please ensure scaler.pkl exists.")
#     scaler = None
# prophet_solar = None
# try:
#     prophet_solar = joblib.load(f"{main_dir}\\models\\prophet_solar.pkl")
# except FileNotFoundError:
#     st.warning("Solar Prophet model not found. Solar predictions will be skipped.")

# # Streamlit app
# st.title("Renewable Energy Power Prediction Dashboard")

# # Filters
# energy_source = st.selectbox("Select Energy Source", ['wind', 'solar'])
# view_type = st.selectbox("Select View", ["Historical Data", "Prediction"])
# horizon = st.selectbox("Select Prediction Horizon", ["30 days", "60 days", "90 days", "6 months"]) if view_type == "Prediction" else None
# horizon_days = {"30 days": 30, "60 days": 60, "90 days": 90, "6 months": 180} if horizon else {}
# days = horizon_days.get(horizon, 30) if horizon else 30

# # Define date range for historical data
# today = pd.to_datetime('2025-09-02')  # Current date
# historical_start = pd.to_datetime('2025-05-01')  # Start of historical data
# historical_end = pd.to_datetime('2025-07-31')  # Data until July 31, 2025

# # Load historical data
# df = load_yearly_data(energy_source, historical_start, historical_end, cols, dtypes, scaler)

# # Cache predictions
# @st.cache_data
# def get_predictions(_prophet_model, days, energy_source, df, _scaler):
#     if df.empty:
#         st.error("Cannot generate predictions: No historical data available.")
#         return pd.DataFrame(columns=['date', 'power_MW'])
#     # Prepare historical data for Prophet to ensure alignment
#     prophet_df = df[['date', 'power_MW']].rename(columns={'date': 'ds', 'power_MW': 'y'})
#     prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], errors='coerce')
#     prophet_df = prophet_df.dropna(subset=['ds', 'y'])
#     if prophet_df.empty:
#         st.error("No valid historical data for Prophet.")
#         return pd.DataFrame(columns=['date', 'power_MW'])
#     # Create future dataframe starting from today
#     future = _prophet_model.make_future_dataframe(periods=days, freq='D', include_history=False)
#     for col in ['wind_speed', 'precipitation', 'temperature_avg', 'wind_volatility']:
#         if col in df.columns:
#             future[col] = df[df['energy_source'] == energy_source][col].mean()
#         else:
#             future[col] = 0
#     try:
#         forecast = _prophet_model.predict(future)
#     except Exception as e:
#         st.error(f"Prophet prediction failed: {str(e)}")
#         return pd.DataFrame(columns=['date', 'power_MW'])
#     # Filter from today onward
#     future_start = today
#     forecast = forecast[forecast['ds'] >= future_start][['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'power_MW'})
#     if forecast.empty:
#         st.error(f"No predictions generated for {days} days from {today.date()}. Ensure Prophet model is trained with sufficient historical data.")
#         return pd.DataFrame(columns=['date', 'power_MW'])
#     # Inverse-transform power_MW
#     try:
#         forecast_scaled = pd.DataFrame({
#             'wind_speed': 0, 'precipitation': 0, 'temperature_avg': 0, 'wind_volatility': 0,
#             'power_MW': forecast['power_MW'], 'site_density': 0, 'output_efficiency': 0
#         })
#         forecast['power_MW'] = _scaler.inverse_transform(forecast_scaled)[:, 4]
#     except Exception as e:
#         st.error(f"Scaler error: {str(e)}")
#         return pd.DataFrame(columns=['date', 'power_MW'])
#     return forecast

# # Display based on view type
# if view_type == "Historical Data":
#     if not df.empty:
#         st.subheader(f"Historical Data for {energy_source} (May 1, 2025 - July 31, 2025)")
#         df['date'] = pd.to_datetime(df['date'])
#         st.dataframe(df[['date', 'power_MW']].rename(columns={'power_MW': 'Power (MW)'}), height=300)
#         st.line_chart(df.set_index('date')['power_MW'])
#         st.download_button(
#             label="Download Historical Data",
#             data=df[['date', 'power_MW']].to_csv(index=False),
#             file_name="historical_data.csv",
#             mime="text/csv"
#         )
#     else:
#         st.warning("No historical data to display.")

# elif view_type == "Prediction":
#     if energy_source == 'wind':
#         prophet_model = prophet_wind
#     elif energy_source == 'solar' and prophet_solar:
#         prophet_model = prophet_solar
#     else:
#         st.error("No model available for selected energy source.")
#         prophet_model = None

#     if prophet_model and scaler:
#         forecast = get_predictions(prophet_model, days, energy_source, df, scaler)
#         if not forecast.empty:
#             st.subheader(f"Daily Power Predictions ({horizon}) from {today.date()}")
#             forecast['date'] = pd.to_datetime(forecast['date'])
#             st.dataframe(forecast.rename(columns={'power_MW': 'Predicted Power (MW)'}), height=300)
#             st.line_chart(forecast.set_index('date')['power_MW'])
#             st.download_button(
#                 label="Download Daily Predictions",
#                 data=forecast.to_csv(index=False),
#                 file_name="daily_forecast.csv",
#                 mime="text/csv"
#             )
#         else:
#             st.warning("No predictions to display.")


import streamlit as st
import pandas as pd
import joblib
import sqlite3
import os
from prophet import Prophet
from datetime import timedelta, datetime
from sklearn.preprocessing import StandardScaler
import numpy as np

# Set main directory and database path
main_dir = os.getcwd().rsplit("\\", 0)[0]
db_path = f"{main_dir}\\data\\energy_data.db"

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

# Cache data loading from SQLite
@st.cache_data
def load_yearly_data(energy_source, start_date, end_date, cols, dtypes, _scaler):
    try:
        conn = sqlite3.connect(db_path)
        query = """
        SELECT *
        FROM energy
        WHERE energy_source = ? AND date >= ? AND date <= ?
        """
        df = pd.read_sql_query(query, conn, params=(energy_source, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        conn.close()
        if df.empty:
            st.error(f"No data found for {energy_source} between {start_date.date()} and {end_date.date()}. Check {db_path}.")
            return pd.DataFrame(columns=cols)
        # Apply dtypes dynamically
        for col, dtype in dtypes.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype, errors='ignore')
        # Fill NaN in numerical columns
        for col in ['wind_speed', 'precipitation', 'temperature_avg', 'wind_volatility']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean() if not df[col].isna().all() else 0)
        # Destandardize power_MW
        if 'power_MW' in df.columns and _scaler is not None:
            try:
                scaled_data = pd.DataFrame({
                    'wind_speed': df['wind_speed'].fillna(0),
                    'precipitation': df['precipitation'].fillna(0),
                    'temperature_avg': df['temperature_avg'].fillna(0),
                    'wind_volatility': df['wind_volatility'].fillna(0),
                    'power_MW': df['power_MW'],
                    'site_density': df['site_density'].fillna(0),
                    'output_efficiency': df['output_efficiency'].fillna(0)
                })
                df['power_MW'] = _scaler.inverse_transform(scaled_data)[:, 4]
            except Exception as e:
                st.error(f"Failed to destandardize historical power_MW: {str(e)}")
        # Log diagnostics
        # st.write(f"Sample data columns: {df.columns.tolist()}")
        # st.write(f"NaN counts: {df.isna().sum().to_dict()}")
        # st.write(f"Row count: {len(df)}")
        return df
    except Exception as e:
        st.error(f"Failed to load data from SQLite: {str(e)}")
        return pd.DataFrame(columns=cols)

# Load Prophet models and scaler
try:
    prophet_wind = joblib.load(f"{main_dir}\\models\\prophet_wind.pkl")
except FileNotFoundError:
    st.error("Wind Prophet model not found. Please ensure prophet_wind.pkl exists.")
    prophet_wind = None
try:
    scaler = joblib.load(f"{main_dir}\\models\\scaler.pkl")
except FileNotFoundError:
    st.error("Scaler model not found. Please ensure scaler.pkl exists.")
    scaler = None
prophet_solar = None
try:
    prophet_solar = joblib.load(f"{main_dir}\\models\\prophet_solar.pkl")
except FileNotFoundError:
    st.warning("Solar Prophet model not found. Solar predictions will be skipped.")

# Streamlit app
st.title("Renewable Energy Power Prediction Dashboard")

# Filters
energy_source = st.selectbox("Select Energy Source", ['wind', 'solar'])
view_type = st.selectbox("Select View", ["Historical Data", "Prediction"])
horizon = st.selectbox("Select Prediction Horizon", ["30 days", "60 days", "90 days", "6 months"]) if view_type == "Prediction" else None
horizon_days = {"30 days": 30, "60 days": 60, "90 days": 90, "6 months": 180} if horizon else {}
days = horizon_days.get(horizon, 30) if horizon else 30

# Define date range for historical data
today = pd.to_datetime(datetime.today().date())  # Automatically detect current date (date only)
historical_start = pd.to_datetime('2025-01-01')  # Start of historical data
historical_end = pd.to_datetime('2025-07-31')  # Data until July 31, 2025

# Load historical data
df = load_yearly_data(energy_source, historical_start, historical_end, cols, dtypes, scaler)

# Cache predictions
@st.cache_data
def get_predictions(_prophet_model, days, energy_source, df, _scaler):
    if df.empty:
        st.error("Cannot generate predictions: No historical data available.")
        return pd.DataFrame(columns=['date', 'power_MW'])
    # Prepare historical data for Prophet to ensure alignment
    prophet_df = df[['date', 'power_MW']].rename(columns={'date': 'ds', 'power_MW': 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], errors='coerce')
    prophet_df = prophet_df.dropna(subset=['ds', 'y'])
    if prophet_df.empty:
        st.error("No valid historical data for Prophet after cleaning.")
        return pd.DataFrame(columns=['date', 'power_MW'])
    # Log prophet_df diagnostics
    st.write(f"Prophet input data: {len(prophet_df)} rows, date range: {prophet_df['ds'].min().date()} to {prophet_df['ds'].max().date()}")
    # Create future dataframe with exact number of days
    future_dates = pd.date_range(start=today, periods=days, freq='D')
    future = pd.DataFrame({'ds': future_dates})
    for col in ['wind_speed', 'precipitation', 'temperature_avg', 'wind_volatility']:
        if col in df.columns:
            future[col] = df[df['energy_source'] == energy_source][col].mean()
        else:
            future[col] = 0
    # Log future dataframe diagnostics
    st.write(f"Future dataframe: {len(future)} rows, date range: {future['ds'].min().date()} to {future['ds'].max().date()}")
    try:
        forecast = _prophet_model.predict(future)
    except Exception as e:
        st.error(f"Prophet prediction failed: {str(e)}")
        return pd.DataFrame(columns=['date', 'power_MW'])
    # Filter and rename
    forecast = forecast[forecast['ds'] >= today][['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'power_MW'})
    if len(forecast) != days:
        st.error(f"Expected {days} predictions, but got {len(forecast)}. Ensure Prophet model is compatible with daily frequency.")
        return pd.DataFrame(columns=['date', 'power_MW'])
    # Inverse-transform power_MW
    try:
        forecast_scaled = pd.DataFrame({
            'wind_speed': 0, 'precipitation': 0, 'temperature_avg': 0, 'wind_volatility': 0,
            'power_MW': forecast['power_MW'], 'site_density': 0, 'output_efficiency': 0
        })
        forecast['power_MW'] = _scaler.inverse_transform(forecast_scaled)[:, 4]
    except Exception as e:
        st.error(f"Scaler error: {str(e)}")
        return pd.DataFrame(columns=['date', 'power_MW'])
    return forecast

# Display based on view type
if view_type == "Historical Data":
    if not df.empty:
        st.subheader(f"Historical Data for {energy_source} (May 1, 2025 - July 31, 2025)")
        df['date'] = pd.to_datetime(df['date'])
        st.dataframe(df[['date', 'power_MW']].rename(columns={'power_MW': 'Power (MW)'}), height=300)
        st.line_chart(df.set_index('date')['power_MW'])
        st.download_button(
            label="Download Historical Data",
            data=df[['date', 'power_MW']].to_csv(index=False),
            file_name="historical_data.csv",
            mime="text/csv"
        )
    else:
        st.warning("No historical data to display.")

elif view_type == "Prediction":
    if energy_source == 'wind':
        prophet_model = prophet_wind
    elif energy_source == 'solar' and prophet_solar:
        prophet_model = prophet_solar
    else:
        st.error("No model available for selected energy source.")
        prophet_model = None

    if prophet_model and scaler:
        forecast = get_predictions(prophet_model, days, energy_source, df, scaler)
        if not forecast.empty:
            st.subheader(f"Daily Power Predictions ({horizon}) from {today.date()}")
            forecast['date'] = pd.to_datetime(forecast['date'])
            st.dataframe(forecast.rename(columns={'power_MW': 'Predicted Power (MW)'}), height=300)
            st.line_chart(forecast.set_index('date')['power_MW'])
            st.download_button(
                label="Download Daily Predictions",
                data=forecast.to_csv(index=False),
                file_name="daily_forecast.csv",
                mime="text/csv"
            )
        else:
            st.warning("No predictions to display.")