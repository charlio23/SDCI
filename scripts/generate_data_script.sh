# Here are some examples to generate data
# 5 balls, collision, 2 states, 3 types, fixed graph (should take ~20 mins)
python datasets/generate_data_spring.py --state-type collision --num-train 1000 --num-valid 1 --num-test 100 --n_balls 5 --length 8100 --box-size 1 --n_states 2 --n_edge_types 3 --temperature 0.5 --datadir springs_3_var_collision_80_strong_2_states_fixed --fixed_connectivity --seed 24
# 5 balls, region, 2 states, variable graphs (it takes a long time (+10 hours) due to random seeding to control state proportions. consider generating less samples and use --agument flag in training code.)
python datasets/generate_data_spring.py --state-type region --num-train 2000 --num-valid 1 --num-test 100 --n_balls 5 --length 8100 --box-size 1 --n_states 2 --temperature 0.5 --datadir springs_5_var_region_100_strong_2_states --seed 24
