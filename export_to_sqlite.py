import pandas as pd
import sqlite3

# Load dataset
df = pd.read_csv('../data/patient_data.csv')

# Create SQLite DB and export data
conn = sqlite3.connect('patient_readmission.db')
df.to_sql('patients', conn, if_exists='replace', index=False)
conn.close()

print("Data exported to patient_readmission.db successfully!")
