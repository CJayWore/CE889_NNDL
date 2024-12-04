import pandas as pd

source_df = pd.read_csv('raw_data.csv')
x = len(source_df)//5
testing_df = source_df.sample(n=x) # 随机选取x行
training_df = source_df.drop(testing_df.index)
training_df.to_csv('train.csv', index=False)
testing_df.to_csv('test.csv', index=False)

MAX = []
MIN = []
# Normalization:
def normalize_training_data(df):
    result = df.copy()
    for column in training_df.columns:
        max_value = df[column].max()
        MAX.append(max_value)
        min_value = df[column].min()
        MIN.append(min_value)
        result[column] = (df[column] - min_value) / (max_value - min_value)
    return result

def normalize_testing_data(df):
    result = df.copy()
    i = 0
    for column in testing_df.columns:
        result[column] = (df[column] - MIN[i]) / (MAX[i]-MIN[i])
        i += 1
    return result
#

# def normalize_data(df_train, df_test):
#     train_result = df_train.copy()
#     test_result = df_test.copy()
#     for column in training_df.columns:
#         max_value = df_train[column].max()
#         min_value = df_train[column].min()
#         train_result[column] = (df_train[column] - min_value) / (max_value - min_value)
#         test_result[column] = (df_test[column] - min_value) / (max_value - min_value)
#     return train_result, test_result
# train_normalized, test_normalized = normalize_data(training_df,testing_df)
# print('MAX:', MAX, 'MIN:', MIN)
train_normalized = normalize_training_data(training_df)
test_normalized = normalize_testing_data(testing_df)
train_normalized.to_csv('train_normalized.csv', index=False)
test_normalized.to_csv('test_normalized.csv', index=False)

