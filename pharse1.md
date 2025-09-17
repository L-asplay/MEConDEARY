record_pharse1

# 模型简化

## 1. 去选择器    
1. 修改 forword   
2. 修改 loss   
3. 取消全连接层

## 2. 减少参数/从TSP到MEC  
0. TSP

1. task_postion, UAV_start_position  

run_20250226T102326

2. timewindow 

run_20250227T163541

3. --priority 10 5 14 17 7 2  # 待修正

run_20250303T210248

4. task_data ,CPU_circles ,IoT_resource ,UAV_resource  

run_20250307T231049

在 graph_size = 5,   uav : fly  ~ 20 : 16   cost ~ 36
在 graph_size = 20,   cost ~ 90

run_20250316T153725

## 3. 分层强化学习
1. Selectlayer.py

    feature_dim = 7 
    lr = 1e-5, 
    gamma = 0.99

2. add HRL
run.py 
   reinforce_baselines/RolloutBaseline()
   train/rollout()
   reinforce_baselines/WarmupBaseline()
   train/validate()
train.py
   train/train_epoch()
   train/train_batch()
selectlayer.py 
attention_model.py 
state_mec.py 

## 4. 对比算法

## 5. list

改 selectlayer 为师姐 FullCollectLayer
改 dqn 为 rainbow dqn
改 dqn cost 为 att return

## 待优化
eval 模式
其它 TSP

# usage
```
python run.py --batch_size 512 --epoch_size 12800 --val_size 1000 --n_epochs 50 --baseline rollout --run_name 'tsp5_rollout'  --problem 'tsp' --graph_size 5
python run.py --batch_size 512 --epoch_size 12800 --val_size 1000 --n_epochs 50 --baseline rollout --run_name 'mec5_rollout' --problem 'mec' --graph_size 5 --lr_model 1e-5

python run.py --problem 'mec' --graph_size 20  --baseline rollout --lr_model 1e-5 --priority 10 5 14 17 7 2 --run_name 'test' --sub_encode_layers 1

nohup python run.py --problem 'mec' --graph_size 30 --select_size 20 --baseline rollout  --run_name 'test' 

nohup python run.py --problem 'mec' --graph_size 20  --baseline rollout --priority 10 5 14 17 7 2 --sub_encode_layers 1 >./output8.log 2>&1 &

nohup python run.py --problem 'mec' --graph_size 20  --baseline rollout --lr_model 1e-5 --priority 10 5 14 17 7 2 --sub_encode_layers 1 >./output7.log 2>&1 &

nohup python run.py --problem 'mec' --graph_size 20  --baseline rollout --lr_model 1e-5 --priority 10 5 14 17 7 2 --run_name 'test' >./output6.log 2>&1 &

nohup python run.py --problem 'mec' --graph_size 20  --baseline rollout --lr_model 1e-5  >./output4.log 2>&1 &

nohup python run.py --problem 'tsp' --graph_size 20  --baseline rollout --lr_model 1e-5 >./trycheck/output20.log 2>&1 &

nohup python run.py --problem 'mec' --graph_size 30 --select_size 20 --baseline rollout  >./trycheck/output30.log 2>&1 &

nohup python run.py --problem 'mec' --graph_size 30 --select_size 20 --n_epochs 23 --baseline rollout  >./trycheck/output300.log 2>&1 & 
```
```
ps -ef | grep python

sudo dmesg | tail -7

tensorboard --logdir=./MEC/logs/mec_30 --port 8001

python run.py --problem 'mec' --graph_size 30 --select_size 10 --n_epochs 1 --baseline rollout --run_name 'test' 
nohup python run.py --problem 'mec' --graph_size 30 --select_size 10 --n_epochs 50 --baseline rollout  >./temlogs/output1.log 2>&1 &  
nohup python run.py --problem 'mec' --graph_size 30 --select_size 10 --n_epochs 100 --baseline rollout >./&
单独优化
nohup python zbrun.py --problem 'mec' --graph_size 30 --select_size 10 --baseline rollout --load_path 'outputs/mec_30_1/run_20250418T010053/epoch-99.pt' >./man1.log 2>&1 &
nohup python zbrun.py --problem 'mec' --graph_size 30 --select_size 10 --baseline rollout --load_path 'outputs/mec_30/run_20250428T132621/epoch-99.pt' >./man1.log 2>&1 &
```


run --从头训练
zarun --加载后协同训练
zbrun --加载后单独训练 

run_20250418T010053 dqn run < greed att 7 8 9 11 12 
for 1
run_20250422T163817 dqn zarun < greed att
run_20250428T132621 rainbow run < greed att 10
run_20250506T082618 dqn zbrun < greed att 6
run_20250508T060312 rainbow zbrun < greed att

160963
run_20250525T011447 rainbow run 100 32-pt
run_20250601T233956 rainbow bun 100 
run_20250606T164403 dqn run 100 
run_20250609T161529

nohup python zbrun.py --problem 'mec' --graph_size 30 --select_size 10 --baseline rollout --load_path 'outputs/mec_30/run_20250606T164403/epoch-99.pt' >./pa.log 2>&1 &

248565