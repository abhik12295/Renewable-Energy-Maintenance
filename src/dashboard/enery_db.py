import pandas as pd
import sqlite3
import os

yearly_dir = "C:/Users/stuar/Desktop/Renewable Energy Maintenance/data/processed/yearly"
db_path = "C:/Users/stuar/Desktop/Renewable Energy Maintenance/data/processed/energy_data.db"

conn = sqlite3.connect(db_path)

for fname in os.listdir(yearly_dir):
    if fname.endswith(".csv"):
        year = fname.split("_")[-1].replace(".csv", "")
        df = pd.read_csv(os.path.join(yearly_dir, fname))
        df['year'] = int(year)
        df.to_sql('energy', conn, if_exists='append', index=False)

conn.close()
