import pandas as pd

def detect_overtrading(df):
    # Logic: Group by hour, if count > 10, return 'High Risk'
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    hourly_counts = df.set_index('Timestamp').resample('H').size()
    return hourly_counts[hourly_counts > 10]