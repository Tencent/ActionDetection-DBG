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
        self.feat_dir = dataset_info['feat_dir']
        self.video_filter = dataset_info['video_filter']
        self.tscale = dataset_info['tscale']
        self.video_info_file = dataset_info['video_info_file']
        self.video_filter_file = dataset_info['video_filter_file']
        self.iou_label_dir = dataset_info['iou_label_dir']
        self.data_aug = dataset_info['data_aug']
        self.feature_dim = dataset_info['feature_dim']

        saver_info = data['saver']
        root_dir = saver_info['root_dir']
        self.checkpoint_dir = os.path.join(root_dir, saver_info['checkpoint_dir'])
        self.result_dir = os.path.join(root_dir, saver_info['result_dir'])
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        training_info = data['training']
        learning_rate = training_info['learning_rate']
        lr_epochs = training_info['lr_epochs']
        assert len(learning_rate) == len(lr_epochs)
        self.learning_rate = []
        for lr, n in zip(learning_rate, lr_epochs):
            self.learning_rate.extend([lr] * n)
        self.epoch_num = len(self.learning_rate)
        self.batch_size = training_info['batch_size']

        testing_info = data['testing']
        self.test_mode = testing_info['mode']
        self.test_batch_size= testing_info['batch_size']


if __name__ == '__main__':
    config = DBGConfig()
