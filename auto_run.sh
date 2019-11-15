#! /bin/bash
config="config.yaml"
<<<<<<< HEAD
#python train.py $config
~/anaconda3/envs/python3/bin/python test.py $config
python post_processing.py output/result output/result_proposal.json
=======
python test.py $config
python post_processing.py output/result output/result_proposal.json
python eval.py output/result_proposal.json
>>>>>>> update DBG
