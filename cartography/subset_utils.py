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
from collections import Counter
import random, os


STRING_TRUNCATE = 50
MTRC2ABV = {"Inverse PPL": "inv_ppl", "Neg PPL": "neg_ppl", "CHIA": "chia", "BLEU": "bleu"}
ABV2MTRC = {"inv_ppl": "Inverse PPL", "neg_ppl": "Neg PPL", "chia": "CHIA", "bleu": "BLEU"}
CRIT2ABV = {"Easy to Learn": "easy_to_learn", "Ambiguous": "ambiguous", "Hard to Learn": "hard_to_learn", "Random": "random"}
create_fname = lambda m, cr, c_e: f"{MTRC2ABV[m]}_{CRIT2ABV[cr]}_{c_e}.pickle"
create_ratio_fname = lambda m, cr, c_e, rto: f"{MTRC2ABV[m]}_{CRIT2ABV[cr]}_{c_e}_{rto}.pickle"
create_comb_fname = lambda m, cr1, cr2, c_e: f"{MTRC2ABV[m]}_{CRIT2ABV[cr1]}_{CRIT2ABV[cr2]}_{c_e}.pickle"
outputs_path = lambda x: f"../scores/{x}"

# These seeds are necessary as each seed corresponds to a different training dynamics set
# These seeds are randomly generated and used during each 100% training, and saved for later usage
DATASET2SEEDS = {
    "cfq": ["1685001853", "3960970220", "2895201892"],
    "cogs": ["83256541", "4190663204", "3926193344"],
    "pcfg": ["42"],
}

DATASET2CEPOCHS = {
    "cfq": 20,
    "cogs": 10,
    "pcfg": 140,
}


def wandb_command(config_name, config_path):
    config_path = "/".join(config_path.split("/")[1:])
    return f"wandb sweep --name {config_name} {config_path}"


def wandb_config_name(seed, dname, metric, criterion, conv_epoch, ratio, mode="single"):
    if mode == "single":
        return f"cartography/subsets/{seed}/{dname}/{MTRC2ABV[metric]}_{CRIT2ABV[criterion]}_{conv_epoch}_{ratio}.pickle"


def create_wandb_config(dname, metric, criterion, conv_epoch, ratio, indices_paths, mode="single", **kwargs):    
    if dname == "cfq":
        config_dir = "../sweeps/templates/cfq_mcd_subset_tmp.yaml"
        config_name = "cfq_mcd_subset.yaml"
    elif dname == "cogs":
        config_dir = "../sweeps/templates/cogs_trafo_subset_tmp.yaml"
        config_name = "cogs_trafo_subset.yaml"
    else:
        assert False, "Dataset name is different from cfq or cogs"

    with open(config_dir, "r") as f:
        config_template = f.read()
        
    if mode == "single":
        new_config_dir = f"../sweeps/{MTRC2ABV[metric]}/{CRIT2ABV[criterion]}/{conv_epoch}/{ratio}"
        new_config_path = f"../sweeps/{MTRC2ABV[metric]}/{CRIT2ABV[criterion]}/{conv_epoch}/{ratio}/{config_name}"
        
    for i, indices_path in enumerate(indices_paths):
        config_template = config_template.replace(f"<INDICES_PATH_{i}>", indices_path)
        
    run_command = wandb_command(config_name, new_config_path)
    config_template = config_template.replace("<WANDB_COMMAND>", run_command)
    
    os.makedirs(new_config_dir, exist_ok=True)
    
    with open(new_config_path, "w") as f:
        f.write(config_template)
        
    return run_command


def dnames_cepochs():
    dnames, cepochs = [], []
    for dataset_name, seeds in DATASET2SEEDS.items():
        dirs = [f"{seed}/{dataset_name}" for seed in seeds]
        dataset_cepochs = [DATASET2CEPOCHS[dataset_name] for seed in seeds]
        dnames.extend(dirs)
        cepochs.extend(dataset_cepochs)
    
    return zip(dnames, cepochs)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    
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
        in_rarity -= np.log(in_v[tok])
        
    in_rarity /= in_len
    
    for tok in out_toks:
        out_rarity -= np.log(out_v[tok])
    
    out_rarity /= out_len
    
    return in_rarity, out_rarity


def get_scores(dir_path: str, converge_epoch: int, string_truncate: int, min_epoch: int = 3) -> Tuple[Dict[int, Dict[str, List[float]]], Dict[str, List[Any]]]:
    file_list = os.listdir(dir_path)
    idx_to_sentences: Dict[int, Dict[str, str]] = read_pickle(os.path.join(dir_path, "idx_to_sentences.pickle"))

    file_list = [f for f in file_list if f[:5] == "epoch"]
    file_list = [f for f in file_list if int(f.split("_")[0].replace("epoch", "")) > min_epoch and int(f.split("_")[0].replace("epoch", "")) < converge_epoch]
    file_list = sorted(file_list, key= lambda s: int(s.split("_")[1].replace("stepidx", "")))

    idxs, ppls, chias, bleus = [], [], [], []
    
    print("Starting reading of training dynamics...")
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
    print("Finished reading of training dynamics.")


    items = list(zip(idxs, ppls, chias, bleus))
    items = sorted(items, key=lambda i: i[0])
    idx_dict: Dict[int, Dict[str, List[float]]] = {}
    
    print("Starting processing scores for Pandas DataFrames...")
    for item in tqdm(items):
        if item[0] not in idx_dict:
            idx_dict[item[0]] = {"inv_ppl": [1 / item[1]], "chia": [item[2]], "bleu": [item[3]]}
        else:
            idx_dict[item[0]]["inv_ppl"].append(1 / item[1])
            idx_dict[item[0]]["chia"].append(item[2])
            idx_dict[item[0]]["bleu"].append(item[3])
    print("Finished processing scores for Pandas DataFrames.")

    i2s = {"Index": [], "In": [], "Out": [], "In abbv.": [], "Out abbv.": [], "In Len": [], "Out Len": [], "In Rarity": [], "Out Rarity": []}

    print("Further processing of examples for length and rarity stats...")
    for k, v in tqdm(idx_to_sentences.items()):
        i2s["Index"].append(k)
        i2s["In"].append(v["in"])
        i2s["Out"].append(v["out"])
        i2s["In abbv."].append(v["in"][:STRING_TRUNCATE])
        i2s["Out abbv."].append(v["out"][:STRING_TRUNCATE])
        i2s["In Len"].append(len(v["in"].split()))
        i2s["Out Len"].append(len(v["out"].split()))

    in_v, out_v = get_word_freqs(i2s)
    for k, v in tqdm(idx_to_sentences.items()):
        in_rarity, out_rarity = get_rarity(v["in"], v["out"], in_v, out_v)
        i2s["In Rarity"].append(in_rarity)
        i2s["Out Rarity"].append(out_rarity)
        
    print("Finished calculating rarity and length.")

    return idx_dict, i2s


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
    
    bins = np.linspace(0.0, 1.0, 10)
    score_names = ["inv_ppl", "chia", "bleu"]
    
    print("Starting calculating confidence and variability stats over epochs...")
    for idx, scores in tqdm(idx_dict.items()):
        scores_list = []
        for score_name in score_names:
            score_arr = np.array(scores[score_name][:epoch])
            mean = score_arr.mean()
            var = score_arr.var()
            scores_list.extend([mean, var])
            if score_name == "bleu":
                correctness = np.isclose(score_arr, 1.0).mean()
                correctness = np.digitize(correctness, bins, right=False) * 0.1
                scores_list.append(correctness)
        
        
        idx_mean_var_list.append(tuple((idx, *scores_list)))
        
    print("Finished calculating statistics.")

    i2s_df = pd.DataFrame.from_dict(i2s)

    df = pd.DataFrame(idx_mean_var_list, columns =['Index', 'Confidence - Inverse PPL', 'Variability - Inverse PPL', \
                                                    'Confidence - CHIA', 'Variability - CHIA', \
                                                    'Confidence - BLEU', 'Variability - BLEU', 'Correctness'])

    cartography = pd.merge(df, i2s_df, on="Index")
    df["Correctness"] = df["Correctness"].apply(lambda x: round(x, 1))
    
    return cartography


def save_subset(subset_idx: set, ds_name: str, subset_fname: str) -> None:    
    os.makedirs(os.path.join("subsets", ds_name), exist_ok=True)
    write_pickle(subset_idx, os.path.join("subsets", ds_name, subset_fname))
    print(f"subset_idx: {len(subset_idx)}")
    

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
        ascending = True # we need examples with low confidence, thus confidence must increase as we go to next experiments
        
    if criteria == "Random":
        sorted_df = df.sample(frac=1)
    else:
        sorted_df = df.sort_values(by=[sort_by], ascending=ascending)

    sorted_df = sorted_df.reset_index(drop=True)
    subset_df = sorted_df.iloc[:int(len(df)*ratio),:]
    
    subset_idx = subset_df["Index"].tolist()
    subset_idx = [int(i) for i in subset_idx]
    subset_idx = set(subset_idx)
    old_subset_len = len(subset_idx)
    
    all_in_v, all_out_v, _, _ = create_vocab(df)
    subset_in_v, subset_out_v, subset_in_v_counts, subset_out_v_counts = create_vocab(subset_df)

    add_ex_i = []
    remove_ex_i = []
    
    for i in trange(int(len(df)*ratio), len(df)):
        item_index, new_in, new_out = sorted_df.iloc[i, 0], sorted_df.iloc[i, 8], sorted_df.iloc[i, 9]
        new_in_tokens, new_out_tokens = set(new_in.split()), set(new_out.split())
        
        if (new_in_tokens - subset_in_v) or (new_out_tokens - subset_out_v):
            add_ex_i.append(item_index)
            subset_in_v = subset_in_v.union(new_in_tokens)
            subset_out_v = subset_out_v.union(new_out_tokens)
            subset_in_v_counts.update(new_in.split())
            subset_out_v_counts.update(new_out.split())
            
    in_counter = subset_in_v_counts
    out_counter = subset_out_v_counts
        
    for i in trange(int(len(df)*ratio)-1, -1, -1):
        if len(remove_ex_i) == len(add_ex_i):
            break
            
        item_index, ex_in, ex_out = sorted_df.iloc[i, 0], sorted_df.iloc[i, 8], sorted_df.iloc[i, 9]
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
            remove_ex_i.append(item_index)
            
    subset_idx = subset_idx - set(remove_ex_i)
    subset_idx = subset_idx.union(set(add_ex_i))
    new_subset_len = len(subset_idx)
    
    assert all_in_v == set(in_counter.keys()), "Not all source vocabulary is covered"
    assert all_out_v == set(out_counter.keys()), "Not all target vocabulary is covered"
    assert old_subset_len == new_subset_len, "Subset size changed"
    
    if write:
        save_subset(subset_idx, ds_name, subset_fname)
        
    return subset_df


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