nohup python -u cartography_plot.py -outputs_path ../scores/scan -plot_path ../scores/scan/plots -converge_epoch 60 > scan.log &

nohup python -u cartography_plot.py -outputs_path ../scores/cogs -plot_path ../scores/cogs/plots -converge_epoch 30 > cogs.log &

nohup python -u cartography_plot.py -outputs_path ../scores/cfq -plot_path ../scores/cfq/plots -converge_epoch 50 > cfq.log &

nohup python -u cartography_plot.py -outputs_path ../scores/pcfg -plot_path ../scores/pcfg/plots -converge_epoch 120 > pcfg.log &


# python cartography_plot.py -outputs_path ../scores/cogs -plot_path ../scores/cogs &