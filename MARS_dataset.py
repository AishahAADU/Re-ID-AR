
from __future__ import print_function, absolute_import
import math
import random
from collections import defaultdict
from scipy.io import loadmat
import os.path as osp
import numpy as np

class Mars(object):
    """
    MARS
    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.
    
    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)
    # cameras: 6
    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
    
    root  = "/MARS" 
    
    train_name_path = osp.join(root, 'info/train_name.txt')
    test_name_path = osp.join(root, 'info/test_name.txt')
    track_train_info_path = osp.join(root, 'info/tracks_train_info.mat')
    track_test_info_path = osp.join(root, 'info/tracks_test_info.mat')
    query_IDX_path = osp.join(root, 'info/query_IDX.mat')

    def __init__(self, min_seq_len=0, ):
        self._check_before_run()
        
       # prepare meta data
        
        train_names ,train_actions= self._get_names("train_3_action.txt")
        test_name,test_actions= self._get_names("test_3_action.txt")

        track_train = loadmat(self.track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
        query_IDX -= 1 # index from 0
        track_query = track_test[query_IDX,:]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]

        train,train_action, num_train_tracklets, num_train_pids, num_train_imgs = self._process_data(train_names,train_actions, track_train,home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)
        video = self._process_train_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)
        query,query_action, num_query_tracklets, num_query_pids, num_query_imgs = self._process_data(test_name,test_actions, track_query,  home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)
        gallery,gallery_action,num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = self._process_data(test_name,test_actions, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)
        
        
        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        # self.train_videos = video
        self.train = train
        self.train_action=train_action
        self.query = query
        self.query_action=query_action
        self.gallery = gallery
        self.gallery_action=gallery_action
        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        actions=[]
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                img,action=new_line.split(' ')
                actions.append(action)
                names.append(img)
                
        return names,actions

    def _process_data(self,names,action, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            img_names = names[start_index-1:end_index]
            actions=action[start_index-1:end_index]
            
            
            
            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                
                tracklets.append((img_paths, pid, camid,actions[0]))
                num_imgs_per_tracklet.append(len(img_paths))
                

        num_tracklets = len(tracklets)
        
        return tracklets, actions, num_tracklets, num_pids, num_imgs_per_tracklet
    def _process_data_withoutclass0intest(self,names,action, meta_data, home_dir=None, relabel=False, min_seq_len=0):
             
         assert home_dir in ['bbox_train', 'bbox_test']
         num_tracklets = meta_data.shape[0]
         pid_list = list(set(meta_data[:,2].tolist()))
         num_pids = len(pid_list)

         if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
         tracklets = []
         num_imgs_per_tracklet = []

         for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            img_names = names[start_index-1:end_index]
            actions=action[start_index-1:end_index]
            
            if int(actions[0])==0:
                continue
            
            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            #assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                actions=int(actions[0])-1
                tracklets.append((img_paths, pid, camid,actions))
                num_imgs_per_tracklet.append(len(img_paths))
                
         num_tracklets = len(tracklets)
         return tracklets, actions, num_tracklets, num_pids, num_imgs_per_tracklet 
        

    def _process_train_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        video = defaultdict(dict)
        
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)
        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            img_names = names[start_index-1:end_index]
            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"
            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                if camid in video[pid] :
                    video[pid][camid].extend(img_paths)  
                else:
                    video[pid][camid] =  img_paths
        return video 