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


def load_scores(dir_path: str, plot_path: str, converge_epoch: int) -> List[Tuple[int, float, float, float]]:
	file_list = os.listdir(dir_path)
	# TODO BLOCKED FOR COGS! - idx_to_sentences: Dict[int, Dict[str, str]] = read_pickle(os.path.join(dir_path, "idx_to_sentences.pickle"))

	file_list = [f for f in file_list if f[:5] == "epoch"]
	file_list = [f for f in file_list if int(f.split("_")[0].replace("epoch", "")) > 3 and int(f.split("_")[0].replace("epoch", "")) < converge_epoch]
	file_list = sorted(file_list, key= lambda s: int(s.split("_")[1].replace("stepidx", "")))

	print("Loading files in:", dir_path)
	idxs, ppls, chias, bleus = [], [], [], []
	for file_name in file_list:
		file_path = f"{dir_path}/{file_name}"
		print(file_name)
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

	# longest_list_len = 0
	# for idx, scores in idx_dict.items():
	# 	for score_name, scores_list in scores.items():
	# 		longest_list_len = max(longest_list_len, len(scores_list))
			

	for epoch in trange(3, converge_epoch, 2):
		# idx_dict_epoch: Dict[int, Dict[str, List[float]]] = {}
		# for idx, scores in idx_dict.items():
		# 	idx_dict_epoch[idx] = {}
		# 	for score_name, scores_list in scores.items():
		# 		idx_dict_epoch[idx][score_name] = scores_list[:epoch]
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
			
			# print(tuple((idx, *scores_list)))
			
			idx_mean_var_list.append(tuple((idx, *scores_list)))

		df = pd.DataFrame(idx_mean_var_list, columns =['Index', 'Confidence - Inverse PPL', 'Variability - Inverse PPL', \
														'Confidence - Neg PPL', 'Variability - Neg PPL', \
														'Confidence - CHIA', 'Variability - CHIA', \
														'Confidence - BLEU', 'Variability - BLEU'])


		plot_types = ["inv_ppl", "neg_ppl", "chia", "bleu"]

		for plot_type in tqdm(plot_types, "Plots"):
			plot(df, plot_path, str(epoch), plot_type)


def plot(df, path_name, extra_path_info, plot_type="inv_ppl"):
	if plot_type == "inv_ppl":
		fig = px.scatter(df, x="Variability - Inverse PPL", y="Confidence - Inverse PPL", color='Confidence - BLEU', range_color=[0,1])
		fig.update_layout(yaxis_range=[0, 1])
	elif plot_type == "neg_ppl":
		fig = px.scatter(df, x="Variability - Neg PPL", y="Confidence - Neg PPL", color='Confidence - BLEU', range_color=[0,1])
		fig.update_layout(yaxis_range=[0, 1])
	elif plot_type == "chia":
		fig = px.scatter(df, x="Variability - CHIA", y="Confidence - CHIA", color='Confidence - BLEU', range_color=[0,1])
		fig.update_layout(yaxis_range=[0, 1])
	elif plot_type == "bleu":
		fig = px.scatter(df, x="Variability - BLEU", y="Confidence - BLEU", color='Confidence - BLEU', range_color=[0,1])
		fig.update_layout(yaxis_range=[0, 1])
	
	fig.update_traces(marker=dict(size=2), selector=dict(mode='markers'))
	fig.write_image(f"{path_name}/cartography_plot_{plot_type}_{extra_path_info}.png", scale=5)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('-outputs_path', required=True, type=str)
	parser.add_argument('-plot_path', required=True, type=str)
	parser.add_argument('-converge_epoch', required=True, type=str)
	args = parser.parse_args()
	outputs_path = args.outputs_path
	plot_path = args.plot_path
	converge_epoch = int(args.converge_epoch)
	df = load_scores(outputs_path, plot_path, converge_epoch)

if __name__ == "__main__":
	main()
	print("Done!")