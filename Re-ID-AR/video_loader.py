from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms
import math
random.seed(1)


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, action, seq_len=15, sample='evenly', transform=None , max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid, action= self.dataset[index]
        num = len(img_paths)

        

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            
            if len(indices) < self.seq_len:
                indices=np.array(indices)
                indices = np.append(indices , [indices[-1] for i in range(self.seq_len - len(indices))])
            else:
                indices=np.array(indices)
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            
            return imgs, pid, camid

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            
        
            cur_index=0
            frame_indices = [i for i in range(num)]
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            

            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            

            indices_list.append(last_seq)
            imgs_list=[]
            

            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break 
                imgs = []
                
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                    
                imgs = torch.cat(imgs, dim=0)
                
                imgs_list.append(imgs)
                
            imgs_array = torch.stack(imgs_list)
            actions=int(action)
            return imgs_array, pid, camid, actions

        elif self.sample == 'dense_subset':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.max_length - 1)
            begin_index = random.randint(0, rand_end)
            

            cur_index=begin_index
            frame_indices = [i for i in range(num)]
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            

            indices_list.append(last_seq)
            imgs_list=[]
            
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break 
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid
        
        elif self.sample == 'intelligent_random':
           
            indices = []
            each = max(num//self.seq_len,1)
            for  i in range(self.seq_len):
                if i != self.seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )
            print(len(indices))
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            
            action=int(action)
            return imgs, pid, camid,action
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

class VideoDataset_inderase(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset,action, seq_len=15, sample='evenly', transform=None , max_length=40,errize=False):
        self.dataset = dataset
        self.action= action
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length
        self.erase = RandomErasing3(probability=0.5, mean=[0.485, 0.456, 0.406])
        self.errize=errize
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid,action = self.dataset[index]
        num = len(img_paths)
        if self.sample != "intelligent":
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices1 = frame_indices[begin_index:end_index]
            indices = []
            for index in indices1:
                if len(indices1) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
        else:
            
            indices = []
            each = max(num//self.seq_len,1)
            for  i in range(self.seq_len):
                if i != self.seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )
            
        imgs = []
        labels = []
        actions=[]
        for index in indices:
            index=int(index)
            img_path = img_paths[index]
            
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
                if (self.errize):
                      img , temp  = self.erase(img)
                      labels.append(temp)
            
            img = img.unsqueeze(0)
            imgs.append(img)
            
        labels = torch.tensor(labels)
        imgs = torch.cat(imgs, dim=0)
        actions= int((action))
        
        if (self.errize):
           return imgs, pid, camid , labels, actions
        return    imgs, pid, camid ,  actions


class RandomErasing3(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img , 0 
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img , 1
            return img , 0  
        

