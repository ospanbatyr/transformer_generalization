import os
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import torch
import matplotlib.pyplot as plt
import pickle5 as pickle
import plotly.express as px
import argparse
import scipy.stats
import scipy.special as special
from typing import Dict, List, Any, Tuple

def seed_everything(seed: int):
    import random, os
    import numpy as np
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything(42)

def read_pickle(file_path: str) -> Any:
	with open(file_path, "rb") as handle:
		return pickle.load(handle)
    
def write_pickle(file: Any, file_path: str) -> None:
    with open(file_path, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
STRING_TRUNCATE = 50

def get_scores(dir_path: str, converge_epoch: int, string_truncate: int, min_epoch: int = 3) -> Tuple[Dict[int, Dict[str, List[float]]], Dict[str, List[Any]]]:
    file_list = os.listdir(dir_path)
    idx_to_sentences: Dict[int, Dict[str, str]] = read_pickle(os.path.join(dir_path, "idx_to_sentences.pickle"))

    file_list = [f for f in file_list if f[:5] == "epoch"]
    file_list = [f for f in file_list if int(f.split("_")[0].replace("epoch", "")) > min_epoch and int(f.split("_")[0].replace("epoch", "")) < converge_epoch]
    file_list = sorted(file_list, key= lambda s: int(s.split("_")[1].replace("stepidx", "")))

    print("Loading files in:", dir_path)
    idxs, ppls, chias, bleus = [], [], [], []

    print("read_pickle")
    for file_name in tqdm(file_list):
        file_path = f"{dir_path}/{file_name}"
        # print(file_name)
        if "ppl" in file_path:
            ppls.extend(read_pickle(file_path).tolist())
        elif "chia" in file_path:
            chias.extend(read_pickle(file_path).tolist())
        elif "bleu" in file_path:
            bleus.extend(read_pickle(file_path))
        elif "idx" in file_path:
            idxs.extend(read_pickle(file_path).tolist())
        else:
            output_csv_name = file_path

    items = list(zip(idxs, ppls, chias, bleus))
    items = sorted(items, key=lambda i: i[0])
    idx_dict: Dict[int, Dict[str, List[float]]] = {}

    print("idx_dict[item[0]]")
    for item in tqdm(items):
        if item[0] not in idx_dict:
            idx_dict[item[0]] = {"inv_ppl": [1 / item[1]], "chia": [item[2]], "bleu": [item[3]]}
        else:
            idx_dict[item[0]]["inv_ppl"].append(1 / item[1])
            idx_dict[item[0]]["chia"].append(item[2])
            idx_dict[item[0]]["bleu"].append(item[3])

    i2s = {"Index": [], "In": [], "Out": [], "In abbv.": [], "Out abbv.": [], "In Len": [], "Out Len": []}

    print("idx_to_sentences.items()")
    for k, v in tqdm(idx_to_sentences.items()):
        i2s["Index"].append(k)
        i2s["In"].append(v["in"])
        i2s["Out"].append(v["out"])
        i2s["In abbv."].append(v["in"][:STRING_TRUNCATE])
        i2s["Out abbv."].append(v["out"][:STRING_TRUNCATE])
        i2s["In Len"].append(len(v["in"].split()))
        i2s["Out Len"].append(len(v["out"].split()))

    return idx_dict, i2s

from collections import Counter

def create_vocab(df):
	in_v = Counter()
	out_v = Counter()
    
	for idx, txt in df["In"].items():
		tokens = txt.split()
		in_v.update(tokens)
         
	for idx, txt in df["Out"].items():
		tokens = txt.split()
		out_v.update(tokens)

	return set(in_v.keys()), set(out_v.keys()), in_v, out_v


def calculate_statistics(epoch: int, idx_dict: Dict[int, Dict[str, List[float]]], i2s: Dict[str, List[Any]]) -> pd.DataFrame:
    idx_mean_var_dict: Dict[int, Dict[str, Tuple[float, float]]] = {}
    idx_mean_var_list: List[Tuple[int, float, float, float, float, float, float, float, float]] = []
    score_names = ["inv_ppl", "chia", "bleu"]
    print("calculate_statistics")
    for idx, scores in tqdm(idx_dict.items()):
        scores_list = []
        for score_name in score_names:
            score_arr = np.array(scores[score_name][:epoch])
            mean = score_arr.mean()
            var = score_arr.var()
            scores_list.extend([mean, var])

        idx_mean_var_list.append(tuple((idx, *scores_list)))

    i2s_df = pd.DataFrame.from_dict(i2s)


    df = pd.DataFrame(idx_mean_var_list, columns =['Index', 'Confidence - Inverse PPL', 'Variability - Inverse PPL', \
                                                    'Confidence - CHIA', 'Variability - CHIA', \
                                                    'Confidence - BLEU', 'Variability - BLEU'])

    cartography = pd.merge(df, i2s_df, on="Index")

    return cartography

def load_scores(dir_path: str, plot_path: str, converge_epoch: int) -> None:
	idx_dict = get_scores(dir_path, plot_path, converge_epoch)
	
	for epoch in trange(3, converge_epoch, 2):
		df = calculate_statistics(epoch, idx_dict)

		plot_types = ["inv_ppl", "chia", "bleu"]

		for plot_type in tqdm(plot_types, "Plots"):
			plot(df, plot_path, str(epoch), plot_type)
            
def save_subset(subset_df: pd.DataFrame, ds_name: str, subset_fname: str) -> None:
    subset_idx = subset_df["Index"].tolist()
    subset_idx = [int(i) for i in subset_idx]
    subset_idx = set(subset_idx)
    
    os.makedirs(os.path.join("subsets", ds_name), exist_ok=True)
    write_pickle(subset_idx, os.path.join("subsets", ds_name, subset_fname))
    print(f"subset_idx: {len(subset_idx)}")
    
    
from pprint import pprint

def choose_subset(df: pd.DataFrame, metric: str, criteria: str, ds_name: str, subset_fname:str, ratio:float = 0.33, write=True) -> pd.DataFrame:
    assert metric in ["Inverse PPL", "Neg PPL", "CHIA", "BLEU"]
    assert criteria in ["Easy to Learn", "Ambiguous", "Hard to Learn", "Random"]
    
    if criteria == "Easy to Learn":
        sort_by = f"Confidence - {metric}"
        ascending = False
    elif criteria == "Ambiguous":
        sort_by = f"Variability - {metric}"
        ascending = False
    elif criteria == "Hard to Learn":
        sort_by = f"Confidence - {metric}"
        ascending = True
        
    if criteria == "Random":
        sorted_df = df.sample(frac=1)
    else:
        sorted_df = df.sort_values(by=[sort_by], ascending=ascending)

    sorted_df = sorted_df.reset_index(drop=True)
    subset_df = sorted_df.iloc[:int(len(df)*ratio),:]
    
    all_in_v, all_out_v, _, _ = create_vocab(df)
    subset_in_v, subset_out_v, subset_in_v_counts, subset_out_v_counts = create_vocab(subset_df)

    add_ex_i = []
    remove_ex_i = []
    
    for i in trange(int(len(df)*ratio), len(df)):
        new_in, new_out = sorted_df.iloc[i, 7], sorted_df.iloc[i, 8]
        new_in_tokens, new_out_tokens = set(new_in.split()), set(new_out.split())
        
        if (new_in_tokens - subset_in_v) or (new_out_tokens - subset_out_v):
            # print(f"In vocab dif: {(new_in_tokens - subset_in_v)}")
            # print(f"Out vocab dif: {(new_out_tokens - subset_out_v)}")
            add_ex_i.append(i)
            subset_in_v = subset_in_v.union(new_in_tokens)
            subset_out_v = subset_out_v.union(new_out_tokens)
            subset_in_v_counts.update(new_in.split())
            subset_out_v_counts.update(new_out.split())
            
    in_counter = subset_in_v_counts
    out_counter = subset_out_v_counts
    
    removed_amount = 0
    for i in trange(int(len(df)*ratio)-1, -1, -1):
        if len(remove_ex_i) == len(add_ex_i):
            break
            
        ex_in, ex_out = sorted_df.iloc[i, 7], sorted_df.iloc[i, 8]
        ex_in_counter, ex_out_counter = Counter(ex_in.split()), Counter(ex_out.split())
        
        upd_in_counter = in_counter - ex_in_counter
        upd_out_counter = out_counter - ex_out_counter
        
        ex_in_words, ex_out_words = list(set(ex_in.split())), list(set(ex_out.split()))
        
        remove = True
        for word in ex_in_words:
            if upd_in_counter[word] <= 1:
                remove = False
        
        for word in ex_out_words:
            if upd_out_counter[word] <= 1:
                remove = False
                
        if remove:
            in_counter = upd_in_counter
            out_counter = upd_out_counter
            remove_ex_i.append(i)
    
    subset_df = pd.concat([subset_df, df.iloc[add_ex_i]])
    subset_df = subset_df.drop(remove_ex_i, axis=0)
    subset_df = subset_df.reset_index(drop=True)
    
    assert all_in_v == set(in_counter.keys()), "The process is wrong"
    assert all_out_v == set(out_counter.keys()), "The process is wrong 2"
    
    if write:
        save_subset(subset_df, ds_name, subset_fname)
    
    print(len(remove_ex_i), len(add_ex_i))
    
    return subset_df


STRING_TRUNCATE = 120

mtrc2abv = {"Inverse PPL": "inv_ppl", "Neg PPL": "neg_ppl", "CHIA": "chia", "BLEU": "bleu"}
crit2abv = {"Easy to Learn": "easy_to_learn", "Ambiguous": "ambiguous", "Hard to Learn": "hard_to_learn", "Random": "random"}
create_fname = lambda m, cr, c_e: f"{mtrc2abv[m]}_{crit2abv[cr]}_{c_e}.pickle"
outputs_path = lambda x: f"../scores/{x}"

DATASET_NAMES = ["pcfg"] #["cogs", "cfq", "scan_length", "scan_jump"]
METRICS = ["BLEU"] # "Inverse PPL", "CHIA", 
CRITERIA = ["Easy to Learn", "Ambiguous", "Hard to Learn"]
CONVERGE_EPOCHS = [140] #[10, 20, 30, 30]

for CONVERGE_EPOCH, DATASET_NAME in zip(CONVERGE_EPOCHS, DATASET_NAMES):
    OUTPUTS_PATH = outputs_path(DATASET_NAME)
    idx_dict, i2s = get_scores(OUTPUTS_PATH, CONVERGE_EPOCH, STRING_TRUNCATE)
    for METRIC in METRICS:
        for CRITERION in CRITERIA:
            df = calculate_statistics(CONVERGE_EPOCH, idx_dict, i2s)
            idx_fname = create_fname(METRIC, CRITERION, CONVERGE_EPOCH)
            subset_df = choose_subset(df, METRIC, CRITERION, DATASET_NAME, idx_fname)