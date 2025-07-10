#!/bin/bash
for seed in $(seq 1 10 500)
do
	echo "Evaluation with seed ${seed}"
	python eval_sim.py eval.policy.run_name=test_27052025 eval.eval_mode=multi_agent eval.visualize=True eval.policy.veh_veh_tilt=-20 eval.policy.veh_edge_tilt=10 eval.policy.goal_tilt=-20 eval.seed=$seed 
done

