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


def read_pickle(file_path: str) -> Any:
	with open(file_path, "rb") as handle:
		return pickle.load(handle)


def get_scores(dir_path: str, converge_epoch: int) -> Tuple[Dict[int, Dict[str, List[float]]], Dict[str, List[Any]]]:
	file_list = os.listdir(dir_path)
	idx_to_sentences: Dict[int, Dict[str, str]] = read_pickle(os.path.join(dir_path, "idx_to_sentences.pickle"))

	file_list = [f for f in file_list if f[:5] == "epoch"]
	file_list = [f for f in file_list if int(f.split("_")[0].replace("epoch", "")) > 3 and int(f.split("_")[0].replace("epoch", "")) < converge_epoch]
	file_list = sorted(file_list, key= lambda s: int(s.split("_")[1].replace("stepidx", "")))

	print("Loading files in:", dir_path)
	idxs, ppls, chias, bleus = [], [], [], []
	for file_name in file_list:
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
	for item in items:
		if item[0] not in idx_dict:
			idx_dict[item[0]] = {"inv_ppl": [1 / item[1]], "neg_ppl": [np.exp(-item[1])], "chia": [item[2]], "bleu": [item[3]]}
		else:
			idx_dict[item[0]]["inv_ppl"].append(1 / item[1])
			idx_dict[item[0]]["neg_ppl"].append(np.exp(-item[1]))
			idx_dict[item[0]]["chia"].append(item[2])
			idx_dict[item[0]]["bleu"].append(item[3])

	i2s = {"Index": [], "In": [], "Out": []}
	for k, v in idx_to_sentences.items():
		i2s["Index"].append(k)
		i2s["In"].append(v["in"])
		i2s["Out"].append(v["out"])

	return idx_dict, i2s


def calculate_statistics(epoch: int, idx_dict: Dict[int, Dict[str, List[float]]], i2s: Dict[str, List[Any]]) -> pd.DataFrame:
	idx_mean_var_dict: Dict[int, Dict[str, Tuple[float, float]]] = {}
	idx_mean_var_list: List[Tuple[int, float, float, float, float, float, float, float, float]] = []
	score_names = ["inv_ppl", "neg_ppl", "chia", "bleu"]
	for idx, scores in idx_dict.items():
		scores_list = []
		for score_name in score_names:
			score_arr = np.array(scores[score_name][:epoch])
			mean = score_arr.mean()
			var = score_arr.var()
			scores_list.extend([mean, var])
		
		idx_mean_var_list.append(tuple((idx, *scores_list)))

	i2s_df = pd.DataFrame.from_dict(i2s)

	print(f"i2s_df.head()")
	print(i2s_df.head())

	df = pd.DataFrame(idx_mean_var_list, columns =['Index', 'Confidence - Inverse PPL', 'Variability - Inverse PPL', \
													'Confidence - Neg PPL', 'Variability - Neg PPL', \
													'Confidence - CHIA', 'Variability - CHIA', \
													'Confidence - BLEU', 'Variability - BLEU'])

	print(f"df.head()")
	print(df.head())

	cartography = pd.merge(df, i2s_df, on="Index")

	return cartography


def load_scores(dir_path: str, plot_path: str, converge_epoch: int) -> None:
	idx_dict = get_scores(dir_path, plot_path, converge_epoch)
	
	for epoch in trange(3, converge_epoch, 2):
		df = calculate_statistics(epoch, idx_dict)

		plot_types = ["inv_ppl", "neg_ppl", "chia", "bleu"]

		for plot_type in tqdm(plot_types, "Plots"):
			plot(df, plot_path, str(epoch), plot_type)


def choose_subsets(df: pd.DataFrame, criteria: str, ratio:float = 0.33) -> None:
	assert criteria in ["Inverse PPL", "Neg PPL", "CHIA", "BLEU"]
	sorted_df = df.sort_values(by=[f'Variability - {criteria}'], ascending=False)
	sorted_df = sorted_df.reset_index(drop=True)
	subset_df = sorted_df.iloc[:int(len(df)*ratio),:]
	return subset_df


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('-outputs_path', required=True, type=str)
	parser.add_argument('-converge_epoch', required=True, type=str)
	args = parser.parse_args()
	outputs_path = args.outputs_path
	converge_epoch = int(args.converge_epoch)
	idx_dict, i2s = get_scores(outputs_path, converge_epoch)
	df = calculate_statistics(converge_epoch // 2, idx_dict, i2s)
	
	for ratio in [0.25, 0.33, 0.5]:
		print(f"Ratio: {ratio}")
		inv_ppl_df = choose_subsets(df, "Inverse PPL", ratio=ratio)
		chia_df = choose_subsets(df, "CHIA", ratio=ratio)
		print(f"len(inv_ppl_df), len(chia_df): {len(inv_ppl_df), len(chia_df)}")

		int_df = pd.merge(inv_ppl_df, chia_df, how='inner', on=['Index'])
		print(f"len(int_df): {len(int_df)}")
		print(f"Intersection Score: {(len(int_df) / len(inv_ppl_df)):.3f}")
		print()


if __name__ == "__main__":
	main()
	print("Done!")

