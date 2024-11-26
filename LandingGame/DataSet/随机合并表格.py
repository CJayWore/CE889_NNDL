import pandas as pd

source_csv = input('source csv: ')
# target_csv = input('target csv：')

source_df = pd.read_csv(source_csv)
# target_df = pd.read_csv(target_csv)

x = len(source_df)//5
# x = int(input("要合并的总行数: "))
rows_selected = source_df.sample(n=x) # 随机选取x行

# 删除源数据中已选取的行
training_df = source_df.drop(rows_selected.index)
training_df.to_csv('train.csv', index=False)

rows_selected.to_csv('test.csv', index=False)

# Combined_Data_df = pd.concat([target_df, rows_selected], ignore_index=True)
# Combined_Data_df.to_csv('Combined_Data.csv', index=False)

