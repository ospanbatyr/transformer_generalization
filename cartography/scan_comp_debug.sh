nohup python -u cartography_plot.py -outputs_path ../scores/scan -plot_path ../scores/scan/plots -converge_epoch 60 > scan.log &

nohup python -u cartography_plot.py -outputs_path ../scores/cogs -plot_path ../scores/cogs/plots -converge_epoch 30 > cogs.log &

nohup python -u cartography_plot.py -outputs_path ../scores/cfq -plot_path ../scores/cfq/plots -converge_epoch 50 > cfq.log &

nohup python -u cartography_plot.py -outputs_path ../scores/pcfg -plot_path ../scores/pcfg/plots -converge_epoch 120 > pcfg.log &


# python cartography_plot.py -outputs_path ../scores/cogs -plot_path ../scores/cogs &

# nohup python -u create_subset.py -outputs_path ../scores/scan -converge_epoch 60 > scan_subsets.log &

# nohup python -u create_subset.py -outputs_path ../scores/cfq -converge_epoch 30 > cfq_subsets_30.log &

# nohup python -u create_subset.py -outputs_path ../scores/cogs -converge_epoch 7 > cogs_subsets_7.log &
# nohup python -u create_subset.py -outputs_path ../scores/cogs -converge_epoch 13 > cogs_subsets_13.log &
# nohup python -u create_subset.py -outputs_path ../scores/cogs -converge_epoch 19 > cogs_subsets_19.log &

# nohup python -u create_subset.py -outputs_path ../scores/pcfg -converge_epoch 90 > pcfg_subsets_90.log &
# nohup python -u create_subset.py -outputs_path ../scores/pcfg -converge_epoch 119 > pcfg_subsets_119.log &