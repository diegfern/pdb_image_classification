import pandas as pd
import sys

data_path = sys.argv[1]
base_name = sys.argv[2]
export_path = sys.argv[3]
split_num = int(sys.argv[4])
    
    
df_list = []

for i in range(0, split_num):
    df_list.append(pd.read_csv(f"{data_path}{base_name}{i}.csv"))

export_df = pd.concat(df_list,axis=0)
export_df.to_csv(f"{export_path}{base_name}.csv",index=False)
