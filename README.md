# Renewable-Energy-Maintenance
Create a scalable, weather-driven AI solution for predictive maintenance in renewable energy systems. This project will cover:

## Workflow Overview:
1. **Data Acquisition**: Gather data from APIs and public datasets.
2. **Data Preprocessing**: Clean and prepare the collected data.
3. **Exploratory Data Analysis (EDA)**: Identify trends and insights within the data.
4. **Model Development**: Train and compare models such as Prophet, LSTM, Random Forest, and Isolation Forest for forecasting energy output.
5. **Model Assessment**: Evaluate models using metrics like RMSE, MAE, and accuracy.
6. **Dashboard Creation**: Develop an interactive dashboard using Streamlit.
7. **API Integration**: Deploy prediction endpoints with Flask or FastAPI.
8. **Testing**: Write unit tests to ensure system robustness.

### Data Sources
- **NCEI Weather Data**: Hourly weather metrics (wind speed, solar radiation, temperature, humidity) from 2015–2025. [NCEI NOAA](https://www.ncei.noaa.gov/data/global-hourly/access/)
- **EIA Energy Data**: Solar and wind farm output, plus maintenance logs. [EIA Open Data](https://www.eia.gov/opendata/) (API key required)
- **OpenStreetMap**: Geospatial information for renewable energy locations. [OpenStreetMap](https://www.openstreetmap.org/)
- **Kaggle**: Wind turbine SCADA datasets. [Example Dataset](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset)
- **X Platform**: Sentiment analysis data related to renewable energy (API access required: [X Developer Platform](https://developer.x.com/))

