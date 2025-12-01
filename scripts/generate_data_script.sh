# Here are some examples to generate data
# 5 balls, region, 4 states, variable graphs (it takes a long time due to random seeding. consider generating less samples and use --agument flag in training code.)
python generate_data_spring.py --state-type region --num-train 2000 --num-valid 1 --num-test 100 --n_balls 5 --length 10100 --box-size 1 --n_states 4 --temperature 0.5 --datadir springs_5_var_region_100_strong_4_states --seed 24
# 5 balls, collision, 2 states, 3 types, fixed graph
python generate_data_spring.py --state-type collision --num-train 1000 --num-valid 1 --num-test 100 --n_balls 5 --length 8100 --box-size 1 --n_states 2 --num-max-edges 3 --temperature 0.5 --datadir springs_3_var_collision_80_strong_2_states_fixed --fixed_connectivity --seed 24