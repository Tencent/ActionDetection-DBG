import argparse
import os
import sys
import math
import cv2
import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix
import pdb
import pandas as pd
sys.path.append('.')
from pyActionRecog import parse_directory
from pyActionRecog import parse_split_file
from pyActionRecog.utils.video_funcs import default_aggregation_func
import time
from PIL import Image
from io import StringIO , BytesIO
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=['activitynet_1.3','ucf101', 'hmdb51'])
parser.add_argument('split', type=int, choices=[1, 2, 3],
                    help='on which split to test the network')
parser.add_argument('modality', type=str, choices=['rgb', 'flow'])
parser.add_argument('frame_path', type=str, help="root directory holding the frames")
parser.add_argument('net_proto', type=str)
parser.add_argument('net_weights', type=str)
parser.add_argument('--rgb_prefix', type=str, help="prefix of RGB frames", default='img_')
parser.add_argument('--flow_x_prefix', type=str, help="prefix of x direction flow images", default='x_')
parser.add_argument('--flow_y_prefix', type=str, help="prefix of y direction flow images", default='y_')
parser.add_argument('--num_frame_per_video', type=int, default=25,
                    help="prefix of y direction flow images")
parser.add_argument('--save_scores', type=str, default=None, help='the filename to save the scores in')
parser.add_argument('--num_worker', type=int, default=1)
parser.add_argument("--caffe_path", type=str, default='./lib/caffe-action/', help='path to the caffe toolbox')
parser.add_argument("--gpus", type=int, nargs='+', default=None, help='specify list of gpu to use')
args = parser.parse_args()

print (args)
sys.path.append(os.path.join(args.caffe_path, 'python'))
from pyActionRecog.action_caffe import CaffeNet

# build neccessary information
print (args.dataset)
gpu_list = args.gpus

# gen video list
video_list = os.listdir(args.frame_path)
video_list.sort()
done_list = [ i.replace(".csv", "") for i in os.listdir(args.save_scores) ]
evl = [i for i in video_list if i not in done_list]

print len(evl)
eval_video_list = evl
print( len(eval_video_list) )
#*****************************
# kinetics 400
#score_name = 'fc_action'
#score_name = 'global_pool'
#m_i_s = 299  # model input size  inceptionv3
#m_i_s = 224  # model input size  bn inception

# anet200
score_name = 'fc-action'
m_i_s = 224  # model input size
#******************************
time_start = time.time()

def build_net():
    global net
    my_id = multiprocessing.current_process()._identity[0] \
        if args.num_worker > 1 else 1
    if gpu_list is None:
        net = CaffeNet(args.net_proto, args.net_weights, my_id-1)
    else:
        net = CaffeNet(args.net_proto, args.net_weights, gpu_list[my_id - 1])


def eval_video(video):
    global net
    vid = video
    print ('video {} start'.format(vid))
    zip_f = os.path.join(args.frame_path , vid)
    if args.modality == 'rgb':
        imgzip = zipfile.ZipFile( zip_f+'/img.zip' , 'r')
        imglist = imgzip.namelist()
        frame_cnt = len(imglist)
        #frame_win=16
        #frame_index = 3
        frame_win=5
        frame_index = 3


    elif args.modality == 'flow':
        flowxzip =  zipfile.ZipFile(zip_f+'/flow_x.zip' , 'r')
        flowyzip =  zipfile.ZipFile(zip_f+'/flow_y.zip' , 'r')
        flowxlist = flowxzip.namelist()
        frame_cnt = len(flowxlist)
        #frame_win=16
        #stack_depth=5
        frame_win=5
        stack_depth=5
    
    frame_scores = []
    for tick in range(1, frame_cnt+1, frame_win):
        if args.modality == 'rgb':
            name = '{}{:05d}.jpg'.format(args.rgb_prefix, min(frame_cnt, tick+frame_index-1) )
            img_b = imgzip.read(name)
            frame = Image.open(BytesIO(img_b))
            frame = cv2.cvtColor( np.asarray(frame), cv2.COLOR_RGB2BGR ) 
            #frame = cv2.imread( os.path.join(video_frame_path, name), cv2.IMREAD_COLOR )
            scores = net.predict_single_frame([frame,], score_name, frame_size=(m_i_s, m_i_s) , over_sample=False)
            #frame_scores.append( np.mean(scores[:,:,0,0] , axis=0) )
            frame_scores.append( np.mean(scores , axis=0) )

        if args.modality == 'flow':
            frame_idx = [min(frame_cnt, tick+offset) for offset in xrange(stack_depth)]
            #print (frame_idx)
            flow_stack = []
            for idx in frame_idx:
                x_name = '{}{:05d}.jpg'.format(args.flow_x_prefix, idx)
                x_b = flowxzip.read(x_name)
                x_frame = Image.open(BytesIO(x_b)) 
                x_frame =  np.asarray(x_frame) 
                flow_stack.append(x_frame)
  
                y_name = '{}{:05d}.jpg'.format(args.flow_y_prefix, idx)
                y_b = flowyzip.read(y_name)
                y_frame = Image.open(BytesIO(y_b)) 
                y_frame =  np.asarray(y_frame) 
                flow_stack.append(y_frame)
                #flow_stack.append(cv2.imread(os.path.join(video_frame_path, x_name), cv2.IMREAD_GRAYSCALE))
                #flow_stack.append(cv2.imread(os.path.join(video_frame_path, y_name), cv2.IMREAD_GRAYSCALE))
            scores = net.predict_single_flow_stack(flow_stack, score_name, frame_size=(m_i_s, m_i_s) , over_sample=False)
            #pdb.set_trace()
            frame_scores.append( np.mean(scores ,axis=0) )
            #frame_scores.append( np.mean(scores[:,:,0,0] ,axis=0) )

    #except :
    #  print ('error video : {}'.format(vid))
    #else:
    time_end = time.time()
    print ('video {} done, frame number:{}, runing time:{}'.format(vid ,frame_cnt, time_end-time_start))
    sys.stdin.flush()
    df_scores = pd.DataFrame( frame_scores )
    df_scores.to_csv(args.save_scores+vid+'.csv')
    #return np.array(frame_scores)

if args.num_worker > 1:
    pool = multiprocessing.Pool(args.num_worker, initializer=build_net)
    pool.map(eval_video, eval_video_list)
    #video_scores = pool.map(eval_video, eval_video_list)
else:
    build_net()
    map(eval_video, eval_video_list)
    #video_scores = map(eval_video, eval_video_list)

'''
video_pred = [np.argmax(default_aggregation_func(x[0])) for x in video_scores]
video_labels = [x[1] for x in video_scores]
cf = confusion_matrix(video_labels, video_pred).astype(float)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)
cls_acc = cls_hit/cls_cnt
print cls_acc
print 'Accuracy {:.02f}%'.format(np.mean(cls_acc)*100)
if args.save_scores is not None:
    for i , video_name in enumerate(eval_video_list):
        df_scores = pd.DataFrame( video_scores[i] )
    	df_scores.to_csv(args.save_scores+video_name+'.csv')
'''



