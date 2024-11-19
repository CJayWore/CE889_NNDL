import pandas as pd

source_csv = input('source csv: ')
target_csv = input('target csv：')

source_df = pd.read_csv(source_csv)
target_df = pd.read_csv(target_csv)

x = int(input("要插入的总行数: "))
rows_insert = source_df.sample(n=x) # 随机选取x行
Combined_Data_df = pd.concat([target_df, rows_insert], ignore_index=True)

Combined_Data_df.to_csv('Combined_Data', index=False)

