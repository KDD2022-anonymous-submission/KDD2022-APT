import os
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser("argument for training")
parser.add_argument("--file", type=str, default="test.out", help="filename")
args = parser.parse_args()

filename = args.file
print("load file", filename)
tmp_dic = {}
metrics = ["acc"]
current_dataset = ""
current_state = ""
with open(filename, "r") as f:
    list1 = f.readlines()
    for i in range(len(list1)):
        line = list1[i].strip().split(' ')
        if line[0] == "dataset":
            if line[1] not in tmp_dic.keys():
                tmp_dic[line[1]] = {row: [] for row in (metrics)}
            current_dataset = line[1]
        if line[0] == "mean" and current_state in metrics:
            tmp_dic[current_dataset][current_state].append(float(line[1]))
        if line[0] in metrics:
            current_state = line[0]
        else:
            current_state = "None"
# print(tmp_dic)
records = pd.DataFrame(index=list(tmp_dic.keys()))
record = []
for i in list(tmp_dic.keys()):
    for metric in metrics:
        tmp_array = np.array(tmp_dic[i][metric])
        a = round(np.mean(tmp_array) * 100, 2)
        b = round(np.std(tmp_array) * 100, 2)
        f = str(a) + "(" + str(b) + ")"
        record.append(f)
records["result"] = record
records.T.to_csv("result_" + filename + ".csv", sep=',')
print(list(tmp_dic.keys()))
print(records.T)
