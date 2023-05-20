# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
import tensorflow as tf
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import craig
from pympler import asizeof
import pickle
from theano.tensor.signal.pool import pool_2d
import theano.tensor as T
import theano
import sys
import h5py
from scipy.ndimage import uniform_filter

from numba import njit, prange

import os
import math
import random
from glob import glob
import os.path as osp
import time
import cv2
from skimage.feature import hog
from skimage import color
from sklearn.decomposition import PCA

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from skimage.metrics import mean_squared_error, structural_similarity
from extractor import BasicEncoder, SmallEncoder
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler,MinMaxScaler

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

        return img1, img2, flow, valid.float(), index



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

#Rewrite the function into parallel mode
def subsetSelection(args,train_dataset,subset_size,model=None,mode = "train"):
    def get_descriptors_surf(channel):
        #hog as descriptors
        # scaler = MinMaxScaler()
        # x_channel = scaler.fit_transform(np.where(np.isnan(channel[0]), np.nanmean(channel[0]), channel[0]))*255
        # y_channel = scaler.fit_transform(np.where(np.isnan(channel[1]), np.nanmean(channel[1]), channel[1]))*255
        # fixed_channel = np.stack((x_channel,y_channel))
        # descriptors= hog(np.transpose(fixed_channel,(1,2,0)), \
        #                       orientations=9, pixels_per_cell=(50,50),\
        #                         cells_per_block=(1,1), visualize=False, \
        #                             channel_axis=-1,feature_vector=True)
        
        
        #Extracting feature using maxpooling
        x_error = channel[0]
        y_error = channel[1]
        downsampling_factor = 10

        # Convert to PyTorch tensors and add two dimensions for compatibility with MaxPool2d: 
        # 1st for batch size and last for number of channels
        x_error = torch.from_numpy(x_error).unsqueeze(0).unsqueeze(0)
        y_error = torch.from_numpy(y_error).unsqueeze(0).unsqueeze(0)

        # Create max pooling layer
        max_pool = torch.nn.MaxPool2d(kernel_size=downsampling_factor, stride=downsampling_factor)

        # Apply max pooling
        downsampled_x = max_pool(x_error)
        downsampled_y = max_pool(y_error)

        # Convert back to numpy and reshape
        downsampled_x = downsampled_x.squeeze(0).squeeze(0).numpy()
        downsampled_y = downsampled_y.squeeze(0).squeeze(0).numpy()

        # Flatten and concatenate the downsampled descriptors
        descriptors = np.concatenate((downsampled_x.reshape(-1), downsampled_y.reshape(-1)))

        #Extracting feature using downsampling
        # x_error = channel[0]
        # y_error = channel[1]
        # downsampling_factor = 10
        # downsampled_x = uniform_filter(x_error,size = downsampling_factor)[::downsampling_factor, ::downsampling_factor]
        # downsampled_y = uniform_filter(y_error,size = downsampling_factor)[::downsampling_factor, ::downsampling_factor]
        # descriptors = np.concatenate((downsampled_x.reshape(-1),downsampled_y.reshape(-1))) 

        """ scaler = MinMaxScaler()
        x_channel = scaler.fit_transform(np.where(np.isnan(channel[0]), np.nanmean(channel[0]), channel[0]))
        y_channel = scaler.fit_transform(np.where(np.isnan(channel[1]), np.nanmean(channel[1]), channel[1]))
        #pca as extracter
        pca = PCA(n_components=10)
        x_decreased = pca.fit_transform(x_channel).reshape(-1)
        y_decreased = pca.fit_transform(y_channel).reshape(-1)
        descriptors = np.concatenate((x_decreased,y_decreased)) """
        # channel = cv2.normalize(np.array(channel), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # surf = cv2.ORB_create(200)
        # keypoints, descriptors = surf.detectAndCompute(channel, None)
        return descriptors
    
    @torch.no_grad()
    def process_data_item(val_id):
        image1, image2, flow_gt, _ , _= train_dataset[val_id]
        if mode != "train": 
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            _, flow_pr = model(image1, image2, iters=3, test_mode=True)
            error_map = (flow_pr[0].cpu() - flow_gt)
            #Use surf as extractor
            extracted_feature = get_descriptors_surf(error_map)
            # Extracting feat ure using HOG as local descriptor
        else:
            #Use surf as extractor
            extracted_feature = get_descriptors_surf(flow_gt)
        if val_id%1000==0:       
            print("Finish processing image {}".format(val_id))
        return extracted_feature
    print("deal with {} images".format(len(train_dataset)))
    dataset_size = len(train_dataset)
    chunk_size = 250
    with ThreadPoolExecutor(max_workers=20) as executor:
        predictions = list(executor.map(process_data_item, range(dataset_size), chunksize=chunk_size))
    predictions = np.array(predictions)
    print("predictions shape is {}".format(predictions.shape))

    chunk_size = 40000
    num_chunks = int(np.ceil(len(predictions) / chunk_size))
    chunks = np.array_split(predictions, num_chunks)
    selected_indices = []
    weights = np.zeros(len(train_dataset))
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1} of {num_chunks}")
        B = int(subset_size*len(chunk))
        subset, subset_weight, _, _, ordering_time, similarity_time = craig.get_orders_and_weights(B, \
                                            chunk, 'euclidean', no=0,equal_num=False,smtk=0)
        #Normalising subset weight
        subset_weight = subset_weight / np.sum(subset_weight) * len(subset_weight)
        # Add the offset to the selected indices to get the correct index in the large array
        offset = len(chunks[i-1]) if i > 0 else 0
        print(f"Offset: {offset}")
        adjusted_indices = [index + offset for index in subset]
        offset_subset = offset + subset
        weights[offset_subset] = subset_weight
        selected_indices.extend(adjusted_indices) 

    return selected_indices, weights


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
                # for i_batch, (image1s, image2s, flows, valid) in tqdm(enumerate(train_loader)):
                #     #disparity_matrics.append(structural_similarity(image1.numpy(),image2.numpy(),multichannel = True))
                #     #left_image.append(image1)
                #     #right_image.append(image2)
                #     # for flow in flows:
                #     #     hog_features= hog(np.transpose(flow,(1,2,0)), orientations=9, pixels_per_cell=(16,16),cells_per_block=(1,1), visualize=False, channel_axis=-1,feature_vector=True)
                #     #     #Directly write into local file
                #     #     """ with open('flying_chairs_hog.bin', 'ab+') as f:
                #     #         f.write(hog_features.tobytes()) """
                #     #     optical_flows.append(hog_features)
                #     #occupied_memory = asizeof.asizeof(optical_flows)/(1024*1024)
                #     flows_np = [flow.numpy() for flow in flows]
                #     optical_flows+=compute_optical_flows(flows_np)
                # print("Shape of the array is [{},{}]".format(len(optical_flows),len(optical_flows[0])))
                # print("subset size is {}".format(B))
                # subset, subset_weight, _, _, ordering_time, similarity_time = craig.get_orders_and_weights(B, np.array(optical_flows), 'euclidean', no=0,equal_num=False,smtk=0)
                subset,weights = subsetSelection(args,train_dataset,subset_size)
                print("subset index extracted")
            else:
                print("Selecting subset based on model predictions")
                model.eval()
                # #Getting training predictions for subset selection
                # print("getting current predictions for full training set")
                # train_dataset = fetch_trainingset(args)
                # predictions = []
                # for val_id in tqdm(range(len(train_dataset))):
                #     image1, image2, flow_gt, _ = train_dataset[val_id]
                #     image1 = image1[None].cuda()
                #     image2 = image2[None].cuda()
                #     _, flow_pr = model(image1, image2, iters=3, test_mode=True)
                #     error_map = (flow_pr[0].cpu()-flow_gt)
                    
                #     #Extracting feature using HOG as local descriptor
                #     extracted_features= hog(np.transpose(error_map,(1,2,0)), orientations=9, \
                #                       pixels_per_cell=(50,50),cells_per_block=(1,1), \
                #                         visualize=False, channel_axis=-1,feature_vector=True)
                #     predictions.append(extracted_features)
                    

                    
                # predictions = np.array(predictions)
                # print("predictions shape is {}".format(predictions.shape))
                # #predictions=np.reshape(predictions,(len(predictions),-1))
                # subset, subset_weight, _, _, ordering_time, similarity_time = craig.get_orders_and_weights(B, predictions, 'euclidean', no=0,equal_num=False,smtk=0)
                subset,weights = subsetSelection(args,train_dataset,subset_size,model=model,mode="pred")
                print("subset extracted")    
        else:
            print("using random subset")
            subset = np.random.choice(len(train_dataset), size=B, replace=False)
            weights = np.ones(len(subset))
        indexed_subset = torch.utils.data.Subset(train_dataset,indices=subset)
        subset_loader = data.DataLoader(indexed_subset, batch_size=args.batch_size, pin_memory=False, shuffle=True, num_workers=4, drop_last=True)
        print('Training with {} out of {} image pairs'.format(len(subset),len(train_dataset)))
        return subset_loader,subset,weights,len(train_dataset)
    else:
        print('Training with %d image pairs' % len(train_dataset))
        return train_loader

