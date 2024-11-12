import pandas as pd
import random

source_csv = input('来源csv: ')
target_csv = input('目标csv：')

source_df = pd.read_csv(source_csv)
target_df = pd.read_csv(target_csv)

x = int(input("请输入要插入的行数: "))
random_rows = source_df.sample(n=x)
Combined_Data_df = pd.concat([target_df, random_rows], ignore_index=True)

Combined_Data_df.to_csv('Combined_Data', index=False)

