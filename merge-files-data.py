
import numpy as np
import pandas as pd
import os


# ================== Combining data files ====================
root = './data'
fileList = []

for path, subdirs, files in os.walk(root):
    subdirs.sort()
    for name in files:
        print(os.path.join(path, name))
        fileList.append(os.path.join(path, name))

print(fileList)

combined_csv = pd.concat([pd.read_csv(f) for f in fileList])

combined_csv.to_csv('combined_data.csv', index=False)






