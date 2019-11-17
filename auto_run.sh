#! /bin/bash
config="config.yaml"
python test.py $config
python post_processing.py output/result output/result_proposal.json
python eval.py output/result_proposal.json
