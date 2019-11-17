import argparse

import yaml
import os

class DBGConfig():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('config_file', type=str, default='config.yaml', nargs='?')
        args = parser.parse_args()

        print('loading config file: %s' % args.config_file)

        with open(args.config_file, 'r', encoding='utf-8') as f:
            tmp_data = f.read()
            data = yaml.load(tmp_data)
        # print(data)

        """ Parse config info
        """
        dataset_info = data['dataset']
        self.video_info_file = dataset_info['video_info_file']
        self.feat_dir = dataset_info['feat_dir']
        self.tscale = dataset_info['tscale']
        self.feature_dim = dataset_info['feature_dim']

        saver_info = data['saver']
        root_dir = saver_info['root_dir']
        self.checkpoint_dir = os.path.join(root_dir, saver_info['checkpoint_dir'])
        self.result_dir = os.path.join(root_dir, saver_info['result_dir'])
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        testing_info = data['testing']
        self.test_mode = testing_info['mode']
        self.test_batch_size= testing_info['batch_size']


if __name__ == '__main__':
    config = DBGConfig()
