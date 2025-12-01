data_size=10000
python train_springs_hidden.py --name springs_region_5_2_states_variable --logdir ./runs --state-decoder region --train_root datasets/springs_5_var_region_80_strong/ --num-edge-types 2 --num-states 2 --suffix _springs5_inter0.5_l8100_s10000 --num-atoms 5 -b 100 --epochs 1000 --step_size 300 --device cuda:0  --prediction_steps 10 --init_temperature 5 --data-size $data_size
