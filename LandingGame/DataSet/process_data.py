import pandas as pd
import numpy as np

source_df = pd.read_csv('ce889_dataCollection.csv',header=None)
x = len(source_df)//5
testing_df = source_df.sample(n=x) # 随机选取x行
training_df = source_df.drop(testing_df.index)
training_df.to_csv('train.csv', index=False, header=False)
testing_df.to_csv('test.csv', index=False,header=False)

MAX = []
MIN = []
# Normalization:
def normalize_training_data(df):
    result = df.copy()
    for column in df.columns:
        max_value = df[column].max()
        MAX.append(max_value)
        min_value = df[column].min()
        MIN.append(min_value)
        result[column] = (df[column] - min_value) / (max_value - min_value)
    min_max_df = pd.DataFrame({
        'MIN':MIN,
        'MAX':MAX
    })
    print(f'MIN=np.array({MIN}) \nMAX=np.array({MAX})')
    return result, min_max_df

def normalize_testing_data(df):
    result = df.copy()
    i = 0
    for column in df.columns:
        result[column] = (df[column] - MIN[i]) / (MAX[i]-MIN[i])
        i += 1
    return result



train_normalized, min_max_df = normalize_training_data(training_df)
test_normalized = normalize_testing_data(testing_df)
train_normalized.to_csv('train_normalized.csv', index=False,header=False)
test_normalized.to_csv('test_normalized.csv', index=False,header=False)

min_max_df.to_csv('min_max.csv', index=False)
