import pandas as pd

# 读取CSV文件
data = pd.read_csv(input('Your data to normalize:'))

# one hot encoding
columns_to_encode = []
while True:
    select_column = input(
        "Please select columns to one hot encode"
         " (Enter an empty string to stop or all to normalize all columns):\n")
    if select_column == "all" or select_column == "ALL" or select_column == "All":
        columns_to_encode = data.columns.tolist()
        break
            # columns_to_encode = all the columns in .csv
    elif select_column == "":
        break
    else:
        columns_to_encode.append(select_column)

# 确保只有数值而没有字符
# columns_to_encode = [col for col in columns_to_encode if pd.api.types.is_numeric_dtype(data[col])]
data = pd.get_dummies(data, columns=columns_to_encode, prefix=columns_to_encode, dtype='int')
# 显示前几行数据
# print("original data:\n", data.head())
data.to_csv('get_dummies_data.csv', index=False)



# Normalization:
def min_max_normalize(df, columns_to_normalize):
    result = df.copy()
    for column in columns_to_normalize:
        max_value = df[column].max()
        min_value = df[column].min()
        result[column] = (df[column] - min_value) / (max_value - min_value)
    return result

def mean_normalize(df, columns_to_normalize):
    result = df.copy()
    for column in columns_to_normalize:
        mean = df[column].mean()
        result[column] = (df[column] - mean) / df[column].std()
    return result

print('1. min_max_normalize\n2. mean_normalize')
normalization_method = input('Select normalization method: ')
columns_to_normalize = []
while True:
    select_column = input(
        "Please select columns to normalize"
        " (Enter an empty string to stop or all to normalize all columns):\n")
    if select_column == "all" or select_column == "ALL" or select_column == "All":
        columns_to_normalize = data.columns.tolist()
        break
        # columns_to_normalize = all the columns in .csv
    elif select_column == "":
        break
    else:
        columns_to_normalize.append(select_column)

#确保只有数值而没有字符
columns_to_normalize = [col for col in columns_to_normalize if pd.api.types.is_numeric_dtype(data[col])]

if normalization_method == "1":
    normalized_data = min_max_normalize(data, columns_to_normalize)
elif normalization_method == "2":
    normalized_data = mean_normalize(data, columns_to_normalize)
else:
    raise ValueError("Please select a valid normalization method")

#Save the normalized data
normalized_data.to_csv('normalized_data.csv', index=False)



