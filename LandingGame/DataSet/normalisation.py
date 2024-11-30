import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_csv_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df.to_csv(output_csv, index=False)

input_csv = 'raw_data.csv'
output_csv = 'scaled_data.csv'
scale_csv_data(input_csv, output_csv)
