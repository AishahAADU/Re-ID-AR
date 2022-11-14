from __future__ import print_function, absolute_import
import numpy as np
from collections import defaultdict
import os.path as osp
class LPW(object):
    
    
    def __init__(self, min_seq_len=0,train_file,test_file, test_query):
        
       
        root = '/pep_256x128'
   
        train_name_path =  train_file    
        Gallery_name_path =  test_file
        Query_name_path =  test_query

        track_train_info_path =  'tracklet_for_train_scene2_8_action.txt'                 
        track_query_info_path =  'tracklet_test_query.txt'
        track_Gallery_info_path  =  'tracklet_test_gallery.txt'
        # prepare meta data
        train_names ,train_actions= self._get_names(train_name_path)
        Gallery_name,Gallery_actions= self._get_names(Gallery_name_path)
        Query_name,Query_actions= self._get_names(Query_name_path)                       
                               
        with open(track_train_info_path,'r') as f:
               track_train = np.array(f.read().strip().split('\n'))
        track_train= np.genfromtxt(track_train,dtype=int, delimiter=' ')    
        #print(track_train)        
        with open(track_Gallery_info_path ,'r') as f:
               track_test = np.array(f.read().strip().split('\n')) 
        track_gallery= np.genfromtxt(track_test,dtype=int, delimiter=' ')         
        with open(track_query_info_path ,'r') as f:
               track_query= np.array(f.read().strip().split('\n'))  
        track_query= np.genfromtxt(track_query,dtype=int, delimiter=' ')
                               
        
        train,train_action, num_train_tracklets, num_train_pids, num_train_imgs = self._process_data(train_names,train_actions, track_train, relabel=True, min_seq_len=min_seq_len)                                                                                                                                                 
        query,query_action, num_query_tracklets, num_query_pids, num_query_imgs = self._process_data(Query_name,Query_actions, track_query,  relabel=False, min_seq_len=min_seq_len)
        gallery,gallery_action,num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = self._process_data(Gallery_name,Gallery_actions, track_gallery, relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> LPW loaded")
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
        #self. _writetraindata()
    

    def _get_names(self, fpath):
        names = []
        actions=[]
        with open(fpath, 'r') as f:
            for line in f:
                
                new_line = line.rstrip()
                if len(new_line.split(' '))==1:
                    print(new_line)
                img,action=new_line.split(' ')
                actions.append(action)
                names.append(img)
                
        return names,actions

    def _writetraindata(self):
         with open("TrainInfoLPWforCheack.txt", 'w') as f:
                for i in self.train:
                    f.write(str(i)+"\n\n\n")
        
    def _process_data(self,names,action, meta_data, relabel=False, min_seq_len=0):
        #assert home_dir in ['bbox_train', 'bbox_test']
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
            
            img_names = names[start_index:end_index]
            actions=action[start_index:end_index]
            
            # append image names with directory information
            img_paths =[img_name for img_name in img_names] 
            
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                
                tracklets.append((img_paths, pid, camid,actions[0]))
                num_imgs_per_tracklet.append(len(img_paths))
                
        num_tracklets = len(tracklets)
        
        return tracklets, actions, num_tracklets, num_pids, num_imgs_per_tracklet

    def _process_train_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        video = defaultdict(dict)

        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        #print(" meta_data shape",meta_data.shape())
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
