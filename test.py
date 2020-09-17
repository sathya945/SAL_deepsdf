import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import cKDTree
import random
import torch.optim as optim
import tqdm
import argparse
# import open3d as o3dcp 
from skimage import measure
from numpy import linalg as LA
import trimesh
import torch.autograd as autograd

from encoder import DGCNN_cls,SimplePointnet
from utils import load_data,to_mesh,Modelnet,SHREC
import glob
data_files = glob.glob("/home/venkata.sathya/my_code/data/SHREC14/Real/Data/*1.*")[:10]

data_path = "data/scripts/50002_jiggle_on_toes/"

NETWORK = 2 #SAL++




use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# Data loading
class pDataset(Dataset):
    
    def __init__(self,data_files=data_files):
        self.point_clouds = []
        self.ptrees = []
        self.sigmas_list = []
        self.count = 0
        for fil in data_files:
            self.point_clouds.append(load_data( fil))
            self.ptrees.append(cKDTree(self.point_clouds[-1]))
            self.sigmas_list.append(self.ptrees[-1].query(self.point_clouds[-1], np.int(30.0))[0][:,-1])
            self.count = self.count + 1

    def __len__(self):
        return 50
    
    def __getitem__(self, index):
        random_shape_no = np.random.randint(self.count)
        rand_idx = np.random.choice(self.point_clouds[random_shape_no].shape[0],500)
        rand_pnts = self.point_clouds[random_shape_no][rand_idx]
        rand_sig = self.sigmas_list[random_shape_no][rand_idx]
        sample = np.concatenate([ rand_pnts +np.expand_dims(rand_sig, -1) * np.random.normal(0.0, 1.0, size=rand_pnts.shape) ,
                              rand_pnts +  np.random.normal(0.0, 1.0, size=rand_pnts.shape)],
                            axis=0)
        
        distance,indx = self.ptrees[random_shape_no].query(sample)
        derv =2*( self.ptrees[random_shape_no].data[indx] - sample )
        return torch.from_numpy(self.point_clouds[random_shape_no]).float(),torch.from_numpy(sample).float(),torch.from_numpy(distance).float(),torch.from_numpy(derv).float()
#network

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.layer1 = nn.Linear(3+2048,512)
        self.layer2 = nn.Linear(512,512)
        self.layer3 = nn.Linear(512,512) 
        self.layer4 = nn.Linear(512,509) #509
        self.layer5 = nn.Linear(512,512)
        self.layer6 = nn.Linear(512,512)
        self.layer7 = nn.Linear(512,512)
        self.layer8 = nn.Linear(512,1)


    def forward(self,x,latent_vector):
        if x.shape[0] == 1:
            x = x.view(-1, x.shape[-1])
        z = latent_vector.repeat(x.shape[0],1)
        z_x = torch.cat((x.T,z.T)).T
        y = F.relu(self.layer1(z_x)) 
        y = F.relu(self.layer2(y)) 
        y = F.relu(self.layer3(y)) 
        y = F.relu(self.layer4(y)) 
        y = F.relu(self.layer5(torch.cat((y,x),1)))
        y = F.relu(self.layer6(y)) 
        y = F.relu(self.layer7(y)) 
        y = self.layer8(y)
        return y

def SAL_Loss(out,gt):
    return torch.abs(torch.abs(out) - gt).mean()

def derv_loss(pd,gt):
    a = pd-gt
    b = pd+gt
    t1 = a.norm(2,dim=1)
    t2 = b.norm(2,dim=1)
    return torch.min(t1,t2).mean()


def build_network():
    net = Net().to(device)
    for k, v in net.named_parameters():
        if 'weight' in k:
            std = np.sqrt(2) / np.sqrt(v.shape[0])
            nn.init.normal_(v, 0.0, std)
        if 'bias' in k:
            nn.init.constant_(v, 0)
        if k == 'layer8.weight':
            std = np.sqrt(np.pi) / np.sqrt(v.shape[1])
            nn.init.constant_(v, std)
        if k == 'layer8.bias':
            nn.init.constant_(v, -1)            
    return net


# initialize network

trainset = pDataset()
testloader = torch.utils.data.DataLoader(trainset, batch_size=1)


encoder_model = torch.load("models/encoder500")

net = torch.load("models/sdf500")


pointcloud,_,_,_ =  next(iter(testloader))
print(pointcloud)
pointcloud = pointcloud.to(device)
latent_vector1 = encoder_model(pointcloud)


pointcloud,_,_,_ =  next(iter(testloader))
print(pointcloud)
pointcloud = pointcloud.to(device)
latent_vector2 = encoder_model(pointcloud)

print( (latent_vector1-latent_vector2).sum() )

to_mesh(net,device,"temp0",latent_vector1)
to_mesh(net,device,"temp1",latent_vector2)

to_mesh(net,device,"temp0_1",( (latent_vector1+latent_vector2)/2))



# trainset = Modelnet("test")
# testloader = torch.utils.data.DataLoader(trainset, batch_size=1)
# encoder_model = torch.load("modelnet40/encoder")
# net = torch.load("modelnet40/sdf")

# for i in range(10):
#     pointcloud,_,_,_ =  next(iter(testloader))
#     pointcloud = pointcloud.to(device)
#     latent_vector1 = encoder_model(pointcloud)
#     to_mesh(net,device,"modelnet"+str(i),latent_vector1,di="modelnet/shapes/")









