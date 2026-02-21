import pandas as pd

def detect_overtrading(df):
    # Logic: Group by hour, if count > 10, return 'High Risk'
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    hourly_counts = df.set_index('timestamp').resample('H').size()
    return hourly_counts[hourly_counts > 10]