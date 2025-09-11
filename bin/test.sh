#!/bin/bash

window_length=10
window_period=1
seed=0

feature_type='ohlcv'
activation='softmax'
actor_predictor='lstm'
critic_predictor='lstm'

rebalancing=10
buy_fee=0.0025
sell_fee=0.0025
time_cost=0

mdd_alpha=0.0
beta=0.0
action_bound=1

random_start=false
episode=1500
max_step=3000
target_date='20250102'

train_period=10
action_show=true

actor_hidden=16
actor_num_layer=1
critic_hidden=32
critic_num_layer=1

# actor_hidden=128
# actor_num_layer=2
# critic_hidden=16
# critic_num_layer=1

# name='us_etf1'
# file_path="data/us_etf1_20250427.pkl"
# name='us_etf2'
# file_path="data/us_etf2_20250428.pkl"

# name='us_stocks1'
# file_path="data/us_stocks1_20250428.pkl"
# name='us_stocks2_bn_off'
name='us_stocks2'
file_path="data/us_stocks2_20250429.pkl"
# file_path="data/us_stocks2_20250513.pkl"

# name='kr_stocks1'
# file_path="data/kr_stocks1_20250428.pkl"

name='us_stocks3'
file_path="data/us_stocks3_20250821.pkl"

inverse=False

performance_check=True

python test.py --window_length $window_length --window_period $window_period --seed $seed --feature_type $feature_type --activation $activation --actor_predictor $actor_predictor --critic_predictor $critic_predictor --rebalancing $rebalancing --buy_fee $buy_fee --sell_fee $sell_fee --time_cost $time_cost --mdd_alpha $mdd_alpha --beta $beta --action_bound $action_bound --random_start $random_start --episode $episode --max_step $max_step --target_date $target_date --train_period $train_period --name $name --file_path $file_path --action_show $action_show --inverse $inverse --critic_num_layer $critic_num_layer --critic_hidden $critic_hidden --actor_num_layer $actor_num_layer --actor_hidden $actor_hidden --performance_check $performance_check