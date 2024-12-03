import pandas as pd

source_df = pd.read_csv('raw_data.csv')
x = len(source_df)//5
testing_df = source_df.sample(n=x) # 随机选取x行

# 删除源数据中已选取的行
training_df = source_df.drop(testing_df.index)
training_df.to_csv('train.csv', index=False)
testing_df.to_csv('test.csv', index=False)


# Normalization:
def min_max_normalize(df, columns_to_normalize):
    result = df.copy()
    for column in columns_to_normalize:
        max_value = df[column].max()
        min_value = df[column].min()
        result[column] = (df[column] - min_value) / (max_value - min_value)
    return result

normalization_method = 'min_max_normalize'
columns_to_normalize = training_df.columns.tolist()

normalized_data = min_max_normalize(training_df, columns_to_normalize)
normalized_data.to_csv('normalized_data.csv', index=False)

