import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_csv_data(input_csv, output_csv, min_max):
    df = pd.read_csv(input_csv)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df_scaled.to_csv(output_csv, index=False)
    min_max_values = pd.DataFrame({
        'column':df.columns,
        'min':scaler.data_min_,
        'max':scaler.data_max_,
    })
    min_max_values.to_csv(min_max, index=False)

input_csv = 'raw_data.csv'
output_csv = 'scaled_data.csv'
min_max = 'min_max_values.csv'
scale_csv_data(input_csv, output_csv, min_max)
