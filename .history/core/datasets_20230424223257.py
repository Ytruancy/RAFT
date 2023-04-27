# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import craig
from pympler import asizeof
import pickle
import sys
import h5py

import os
import math
import random
from glob import glob
import os.path as osp
import time
from skimage.feature import hog
from skimage import color
from sklearn.decomposition import PCA

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from skimage.metrics import mean_squared_error, structural_similarity
from extractor import BasicEncoder, SmallEncoder
from tqdm import tqdm

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()



    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))       
	
        index_images = [str(os.path.basename(file).split("_")[0]) for file in images]
        index_flows = [str(os.path.basename(file).split("_")[0]) for file in flows]
        
        for index in index_images:
            if index not in index_flows:
                print("deleting",index)
                images.remove(osp.join(root, index + '_img1.ppm'))
                images.remove(osp.join(root, index + '_img2.ppm'))
                flows.remove(osp.join(root, index + '_flow.flo'))
			
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1

def fetch_trainingset(args,TRAIN_DS='C+T+K+S+H'):
    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')
    return train_dataset




@torch.no_grad()
def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H', coreset = False, subset_size = 0.4,random = False, cluster_feature = False, model = None):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')



    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)
    
    if coreset:
        print("Trying to get the subset")
        left_image = []
        right_image = []
        optical_flows = []
        B = int(subset_size*len(train_dataset))
        if not random:
            if cluster_feature:
                print("Selecting subset based on cluster feature")
                for i_batch, (image1s, image2s, flows, valid) in tqdm(enumerate(train_loader)):
                    #disparity_matrics.append(structural_similarity(image1.numpy(),image2.numpy(),multichannel = True))
                    #left_image.append(image1)
                    #right_image.append(image2)
                    for flow in flows:
                        hog_features= hog(np.transpose(flow,(1,2,0)), orientations=9, pixels_per_cell=(16,16),cells_per_block=(1,1), visualize=False, channel_axis=-1,feature_vector=True)
                        #Directly write into local file
                        """ with open('flying_chairs_hog.bin', 'ab+') as f:
                            f.write(hog_features.tobytes()) """
                        optical_flows.append(hog_features)
                        #occupied_memory = asizeof.asizeof(optical_flows)/(1024*1024)
                """ with open('hog_Flyingchairs.pkl', 'wb') as f:
                    pickle.dump(optical_flows, f) """
                print("Shape of the array is [{},{}]".format(len(optical_flows),len(optical_flows[0])))
                #Saving array using pickle
                # print("saving the array")
                # #using pickle
                # with open('hog_Flyingchairs.pkl', 'wb') as f:
                #     pickle.dump(optical_flows, f)
                # #Using h5py
                # """with h5py.File("hog_Flyingchairs.h5", "w") as file:
                #     file.create_dataset("large_array", data=optical_flows)"""
                # print("array saved")
                print("subset size is {}".format(B))
                subset, subset_weight, _, _, ordering_time, similarity_time = craig.get_orders_and_weights(B, np.array(optical_flows), 'euclidean', no=0,equal_num=False,smtk=0)
                print("subset index extracted")
            else:
                print("Selecting subset based on model predictions")
                model.eval()
                #Getting training predictions for subset selection
                print("getting current predictions for full training set")
                train_dataset = fetch_trainingset(args)
                predictions = []
                for val_id in tqdm(range(len(train_dataset))):
                    image1, image2, flow_gt, _ = train_dataset[val_id]
                    image1 = image1[None].cuda()
                    image2 = image2[None].cuda()
                    _, flow_pr = model(image1, image2, iters=5, test_mode=True)
                    error_map = (flow_pr[0].cpu()-flow_gt)
                    x_error = error_map[0]
                    y_error = error_map[1]
                    pca = PCA(n_components=50)
                    x_decreased = pca.fit_transform(x_error).reshape(-1)
                    y_decreased = pca.fit_transform(y_error).reshape(-1)
                    compressed_concat = np.concatenate((x_decreased,y_decreased))
                    extracted_features= hog(np.transpose(error_map,(1,2,0)), orientations=9, \
                                      pixels_per_cell=(16,16),cells_per_block=(1,1), \
                                        visualize=False, channel_axis=-1,feature_vector=True)
                    predictions.append(extracted_features)
                    predictions.append(compressed_concat)
                predictions = np.array(predictions)
                print("predictions shape is {}".format(predictions.shape))
                #predictions=np.reshape(predictions,(len(predictions),-1))
                subset, subset_weight, _, _, ordering_time, similarity_time = craig.get_orders_and_weights(B, predictions, 'euclidean', no=0,equal_num=False,smtk=0)
        else:
            print("using random subset")
            subset = np.random.choice(len(train_dataset), size=B, replace=False)
        indexed_subset = torch.utils.data.Subset(train_dataset,indices=subset)
        subset_loader = data.DataLoader(indexed_subset, batch_size=args.batch_size, pin_memory=False, shuffle=True, num_workers=4, drop_last=True)
        print('Training with {} out of {} image pairs'.format(len(subset),len(train_dataset)))
        return subset_loader
    else:
        print('Training with %d image pairs' % len(train_dataset))
        return train_loader

