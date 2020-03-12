export PYTHONPATH=/data/home/action_detection/temporal-segment-networks
bash extract_optical_flow.sh ./data/test_in/ ./data/test_out 7
bash extract_optical_flow.sh /data/home/action_detection/data/v1-3/test/ /data/home/action_detection/data/A_R_F_a 1

#kinetics using zip data
python ./eval_net_zip.py activitynet_1.3 1 rgb /data/home/action_detection/data/A_R_F_a /data/home/action_detection/temporal-segment-networks/models/kinetics_models/kinetics/inception_v3_kinetics_rgb_pretrained/inception_v3_rgb_deploy.prototxt /data/home/action_detection/temporal-segment-networks/models/kinetics_models/kinetics/inception_v3_kinetics_rgb_pretrained/inception_v3_kinetics_rgb_pretrained.caffemodel --num_worker 1 --save_scores ./data/tsn_kinetics_score_5fps/spatial/

python ./eval_net_zip.py activitynet_1.3 1 flow /data/home/action_detection/data/A_R_F_a /data/home/action_detection/temporal-segment-networks/models/kinetics_models/kinetics/inception_v3_kinetics_flow_pretrained/inception_v3_flow_deploy.prototxt /data/home/action_detection/temporal-segment-networks/models/kinetics_models/kinetics/inception_v3_kinetics_flow_pretrained/inception_v3_flow_kinetics.caffemodel --num_worker 1 --save_scores ./data/tsn_kinetics_score_5fps/temporal/

### kinetics bn_inception
python ./eval_net_zip.py activitynet_1.3 1 rgb /data/home/action_detection/data/A_R_F_a /data/home/action_detection/temporal-segment-networks/models/kinetics_models/kinetics/bn_inception_kinetics_rgb_pretrained/bn_inception_rgb_deploy.prototxt /data/home/action_detection/temporal-segment-networks/models/kinetics_models/kinetics/bn_inception_kinetics_rgb_pretrained/bn_inception_kinetics_rgb_pretrained.caffemodel --num_worker 1 --save_scores ./data/tsn_kinetics_score_bninception_1024/spatial/

python ./eval_net_zip.py activitynet_1.3 1 flow /data/home/action_detection/data/A_R_F_a /data/home/action_detection/temporal-segment-networks/models/kinetics_models/kinetics/bn_inception_kinetics_flow_pretrained/bn_inception_flow_deploy.prototxt /data/home/action_detection/temporal-segment-networks/models/kinetics_models/kinetics/bn_inception_kinetics_flow_pretrained/bn_inception_kinetics_flow_pretrained.caffemodel --num_worker 1 --save_scores ./data/tsn_kinetics_score_bninception_1024/temporal/

# anet2016
/data/home/anaconda3/envs/python2/bin/python ./eval_net_zip.py activitynet_1.3 1 rgb /data/home/action_detection/data/A_R_F_a /data/home/action_detection/temporal-segment-networks/models/anet2016_models/resnet200_anet_2016_deploy.prototxt /data/home/action_detection/temporal-segment-networks/models/anet2016_models/resnet200_anet_2016.caffemodel --num_worker 1 --save_scores ./data/tsn_anet2016_score_5fps/spatial/

python ./eval_net_zip.py activitynet_1.3 1 flow /data/home/action_detection/data/A_R_F_a  /data/home/action_detection/temporal-segment-networks/models/anet2016_models/bn_inception_anet_2016_temporal_deploy.prototxt /data/home/action_detection/temporal-segment-networks/models/anet2016_models/bn_inception_anet_2016_temporal.caffemodel.v5 --num_worker 1 --save_scores ./data/tsn_anet2016_score_5fps/temporal/


