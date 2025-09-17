#!/bin/bash

feature_type='ohlcv'
target_date='None' # None | YYYY-MM-DD
train_period=30
country="us2" # us | kr
task="load_df" # load_df | download
save_numpy=True # True | False

cd /home/jmhong20/regime-based-portfolio/src
cmd="python data_handler.py --feature_type $feature_type --target_date $target_date --train_period $train_period --country $country --task $task --save_numpy $save_numpy"
echo $cmd
$cmd
