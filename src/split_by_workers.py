import pandas as pd
import numpy as np
import sys

data_path = sys.argv[1]
base_name = sys.argv[2]
export_path = sys.argv[3]
split_num = int(sys.argv[4])

input_df = pd.read_csv(f"{data_path}{base_name}.csv")

export_dataframes = np.array_split(input_df, split_num)

for idx, dataframe in enumerate(export_dataframes):
    dataframe.to_csv(f"{export_path}{base_name}{idx}.csv", index=False)