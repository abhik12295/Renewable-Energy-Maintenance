import pandas as pd
import os

main_dir = '.'  # Adjust to your main directory path if needed
file_path = f"{main_dir}/data/processed/cleaned_data_final.csv"

# Create yearly directory
yearly_dir = f"{main_dir}/data/processed/yearly"
os.makedirs(yearly_dir, exist_ok=True)

# Split in chunks to save memory
chunksize = 50000
for chunk in pd.read_csv(file_path, chunksize=chunksize):
    chunk['date'] = pd.to_datetime(chunk['date'], format='mixed')
    chunk['year'] = chunk['date'].dt.year
    for year in chunk['year'].unique():
        year_chunk = chunk[chunk['year'] == year]
        year_file = f"{yearly_dir}/cleaned_data_{year}.csv"
        if not os.path.exists(year_file):
            year_chunk.to_csv(year_file, index=False)
        else:
            year_chunk.to_csv(year_file, mode='a', header=False, index=False)

print("CSV split into yearly files successfully.")