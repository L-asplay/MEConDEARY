# 1.Train the model
run.py --从头训练
zbrun.py --加载后单独训练 

在 run.py/zbrun.py Line ~18 通过注释切换第一层选择器是DQN还是RianBow

nohup python run.py --problem 'mec' --graph_size xx --select_size xx --n_epochs 100 --baseline rollout >./run.log 2>&1 &

nohup python zbrun.py --problem 'mec' --graph_size xx --select_size xx --baseline rollout --load_path 'outputs/mec_x/xxx/epoch-xx.pt' >./zbrun.log 2>&1 &

# 2.Plot the alg

nohup python -u plot_alg.py --problem 'mec' --graph_size xx --result name1.npy --pic name2.png  
--load_path1 (加载用DQN的model)
--load_path2 (加载用RianBow的model)
>./ployalg.log     2>&1 &
