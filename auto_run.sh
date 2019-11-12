#! /bin/bash
config="config.yaml"
#python train.py $config
~/anaconda3/envs/python3/bin/python test.py $config
python post_processing.py output/result output/result_proposal.json
