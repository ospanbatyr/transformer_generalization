import os
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import torch
import matplotlib.pyplot as plt
import pickle5 as pickle
import plotly.express as px
import itertools
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
        
def get_word_freqs(i2s):
    in_v = Counter()
    out_v = Counter()
    
    for txt in i2s["In"]:
        tokens = txt.split()
        in_v.update(tokens)

    for txt in i2s["Out"]:
        tokens = txt.split()
        out_v.update(tokens)
    
    total = sum(in_v.values())
    for k in in_v:
        in_v[k] /= total

    total = sum(out_v.values())
    for k in out_v:
        out_v[k] /= total
        
    return in_v, out_v
  

def get_rarity(in_txt, out_txt, in_v, out_v):
    in_toks = in_txt.split()
    out_toks = out_txt.split()
    
    in_rarity, out_rarity = 0, 0
    in_len, out_len = len(in_toks), len(out_toks)
    
    for tok in in_toks:
        in_rarity += in_v[tok]
        
    in_rarity /= in_len
    
    for tok in out_toks:
        out_rarity += out_v[tok]
    
    out_rarity /= out_len
    
    return -np.log(in_rarity), -np.log(out_rarity)
  

STRING_TRUNCATE = 50

def get_scores(dir_path: str, converge_epoch: int, string_truncate: int, min_epoch: int = 3) -> Tuple[Dict[int, Dict[str, List[float]]], Dict[str, List[Any]]]:
    file_list = os.listdir(dir_path)
    idx_to_sentences: Dict[int, Dict[str, str]] = read_pickle(os.path.join(dir_path, "idx_to_sentences.pickle"))

    file_list = [f for f in file_list if f[:5] == "epoch"]
    file_list = [f for f in file_list if int(f.split("_")[0].replace("epoch", "")) > min_epoch and int(f.split("_")[0].replace("epoch", "")) < converge_epoch]
    file_list = sorted(file_list, key= lambda s: int(s.split("_")[1].replace("stepidx", "")))

    # print("Loading files in:", dir_path)
    idxs, ppls, chias, bleus = [], [], [], []
    print("Read pickles")
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
    
    print("Process items")
    for item in tqdm(items):
        if item[0] not in idx_dict:
            idx_dict[item[0]] = {"inv_ppl": [1 / item[1]], "chia": [item[2]], "bleu": [item[3]]}
        else:
            idx_dict[item[0]]["inv_ppl"].append(1 / item[1])
            idx_dict[item[0]]["chia"].append(item[2])
            idx_dict[item[0]]["bleu"].append(item[3])

    i2s = {"Index": [], "In": [], "Out": [], "In abbv.": [], "Out abbv.": [], "In Len": [], "Out Len": [], "In Rarity": [], "Out Rarity": []}

    print("Create items list")
    for k, v in tqdm(idx_to_sentences.items()):
        i2s["Index"].append(k)
        i2s["In"].append(v["in"])
        i2s["Out"].append(v["out"])
        i2s["In abbv."].append(v["in"][:STRING_TRUNCATE])
        i2s["Out abbv."].append(v["out"][:STRING_TRUNCATE])
        i2s["In Len"].append(len(v["in"].split()))
        i2s["Out Len"].append(len(v["out"].split()))

    in_v, out_v = get_word_freqs(i2s)
    
    print("Process item rarity")
    for k, v in tqdm(idx_to_sentences.items()):
        in_rarity, out_rarity = get_rarity(v["in"], v["out"], in_v, out_v)
        i2s["In Rarity"].append(in_rarity)
        i2s["Out Rarity"].append(out_rarity)

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

    print("Calculate statistics")
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
    
    os.makedirs(os.path.join("subsets", "curriculum", ds_name), exist_ok=True)
    write_pickle(subset_idx, os.path.join("subsets", "curriculum", ds_name, subset_fname))
    print(f"subset_idx: {len(subset_idx)}")
    

from pprint import pprint

def choose_subset(df: pd.DataFrame, metric: str, criteria: str, ds_name: str, subset_fname:str, write=True) -> pd.DataFrame:
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
    
    sorted_idx = sorted_df["Index"].tolist()
    sorted_idx = [int(i) for i in sorted_idx]
    
    if write:
        save_subset(sorted_df, ds_name, subset_fname)
    
    return sorted_df
  
  
def combine_subsets(df: pd.DataFrame, subset_dfs: List[pd.DataFrame], ds_name: str, subset_fname: str) -> pd.DataFrame:
    
    combined_set = pd.concat(subset_dfs)
    combined_set = combined_set.drop_duplicates(keep="first")
    
    if len(combined_set) > (len(df) / 2):
        combined_set = combined_set.iloc[:int(len(df) / 2)]
    else:
        count = 0
        while len(combined_set) < (len(df) / 2):
            example = df.sample(n=1)
            if not example.iloc[0]["In"] in combined_set['In'].tolist():
                combined_set = combined_set.append(example)
                
    save_subset(combined_set, ds_name, subset_fname)
    
    return combined_set
  
def plot(df, plot_type="inv_ppl", color_column="_merge"):
	if plot_type == "inv_ppl":
            # print(df["_merge"].unique())
            #df["_merge"] = df["_merge"].cat.remove_categories("right_only")
            # print(df["_merge"].unique())
            #assert '_merge' in df.columns, "_merge not in columns"
            fig = px.scatter(df, x="Variability - Inverse PPL", y="Confidence - Inverse PPL", custom_data=['In abbv.', 'Out abbv.', 'In Len', 'Out Len'], color=color_column) # , range_color=[0,1]
            fig.update_layout(yaxis_range=[0, 1])
            fig.update_traces(
                hovertemplate="<br>".join([
                    "Variability - Inverse PPL: %{x}",
                    "Confidence - Inverse PPL: %{y}",
                    "In: %{customdata[0]}",
                    "Out: %{customdata[1]}",
                    "In Len: %{customdata[2]}",
                    "Out Len: %{customdata[3]}", 
                ])
            )
	elif plot_type == "chia":
		fig = px.scatter(df, x="Variability - CHIA", y="Confidence - CHIA", custom_data=['In abbv.', 'Out abbv.', 'In Len', 'Out Len'], color='Confidence - BLEU', range_color=[0,1])
		fig.update_layout(yaxis_range=[0, 1])
		fig.update_traces(
			hovertemplate="<br>".join([
				"Variability - CHIA: %{x}",
				"Confidence - CHIA: %{y}",
				"In: %{customdata[0]}",
				"Out: %{customdata[1]}",
                "In Len: %{customdata[2]}",
                "Out Len: %{customdata[3]}", 
			])
		)
	elif plot_type == "bleu":
		fig = px.scatter(df, x="Variability - BLEU", y="Confidence - BLEU", custom_data=['In abbv.', 'Out abbv.', 'In Len', 'Out Len'], color='Confidence - BLEU', range_color=[0,1])
		fig.update_layout(yaxis_range=[0, 1])
		fig.update_traces(
			hovertemplate="<br>".join([
				"Variability - BLEU: %{x}",
				"Confidence - BLEU: %{y}",
				"In: %{customdata[0]}",
				"Out: %{customdata[1]}",
                "In Len: %{customdata[2]}",
                "Out Len: %{customdata[3]}", 
			])
		)	
	fig.update_traces(marker=dict(size=3), selector=dict(mode='markers'))
	fig.update_layout(
		autosize=False,
		width=800,
		height=900
	)
	fig.show()
    

STRING_TRUNCATE = 120

mtrc2abv = {"Inverse PPL": "inv_ppl", "Neg PPL": "neg_ppl", "CHIA": "chia", "BLEU": "bleu"}
crit2abv = {"Easy to Learn": "easy_to_learn", "Ambiguous": "ambiguous", "Hard to Learn": "hard_to_learn", "Random": "random"}
create_fname = lambda m, cr, c_e: f"{mtrc2abv[m]}_{crit2abv[cr]}_{c_e}.pickle"
create_ratio_fname = lambda m, cr, c_e, rto: f"{mtrc2abv[m]}_{crit2abv[cr]}_{c_e}_{rto}.pickle"
create_comb_fname = lambda m, cr1, cr2, c_e: f"{mtrc2abv[m]}_{crit2abv[cr1]}_{crit2abv[cr2]}_{c_e}.pickle"
outputs_path = lambda x: f"../scores/{x}"

DATASET_NAMES = ["pcfg"] # "cfq", "cogs", "scan_length", "scan_jump", "pcfg"
METRICS = ["Inverse PPL", "BLEU"] #  "CHIA",
CRITERIA = ["Ambiguous", "Easy to Learn", "Hard to Learn"]
CONVERGE_EPOCHS = [110] # 20, 10, , 30, 30, 140


for CONVERGE_EPOCH, DATASET_NAME in zip(CONVERGE_EPOCHS, DATASET_NAMES):
    OUTPUTS_PATH = outputs_path(DATASET_NAME)
    idx_dict, i2s = get_scores(OUTPUTS_PATH, CONVERGE_EPOCH, STRING_TRUNCATE, min_epoch=44)
    df = calculate_statistics(CONVERGE_EPOCH, idx_dict, i2s)
    for METRIC in METRICS:
        for CRITERION in CRITERIA:
            print(METRIC, CRITERION)
            idx_fname = create_fname(METRIC, CRITERION, CONVERGE_EPOCH)
            subset_df = choose_subset(df, METRIC, CRITERION, DATASET_NAME, idx_fname)