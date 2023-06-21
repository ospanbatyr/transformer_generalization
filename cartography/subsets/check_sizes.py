import pickle5 as pickle
def read_pickle(file_path: str) -> set:
    with open(file_path, "rb") as handle:
        return pickle.load(handle)

sizes = []
import os
import sys

folder = str(sys.argv[1])
for pname in os.listdir(folder):
    if pname.endswith(".pickle"):
        subset = read_pickle(os.path.join(folder, pname))
        sizes.append(len(list(subset)))

print(set(sizes))
