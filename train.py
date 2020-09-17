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
import glob
from skimage import measure
from numpy import linalg as LA
import trimesh
import torch.autograd as autograd

from encoder import DGCNN_cls,SimplePointnet
from utils import load_data,Modelnet,SHREC,to_mesh

import os
# data_files = ['SPRING0082.obj', 'SPRING0050.obj', 'SPRING0008.obj', 'SPRING0079.obj', 'SPRING0047.obj', 'SPRING0001.obj', 'SPRING0058.obj', 'SPRING0063.obj', 'SPRING0009.obj', 'SPRING0071.obj']

# data_files = os.listdir("/home/venkata.sathya/my_code/data/SHREC14/Real/Data")
# data_path = "/home/venkata.sathya/my_code/data/SHREC14/Real/Data/"
NETWORK = 2 #SAL++

data_files = sorted(glob.glob("/home/venkata.sathya/my_code/data/SHREC14/Real/Data/*")) [:10]



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")




#Data loading
class pDataset(Dataset):
    
    def __init__(self):
        self.point_clouds = []
        self.ptrees = []
        self.sigmas_list = []
        self.count = 0
        for fil in data_files:
            self.point_clouds.append(load_data(fil))
            self.ptrees.append(cKDTree(self.point_clouds[-1]))
            self.sigmas_list.append(self.ptrees[-1].query(self.point_clouds[-1], np.int(30.0))[0][:,-1])
            self.count = self.count + 1

    def __len__(self):
        return 100
    
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
# #network

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.layer1 = nn.Linear(3+128,512)
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




def load_data(filename="/home/venkata.sathya/my_code/data/SHREC14/Real/Data/111.obj"):
    print(filename)
    mesh = trimesh.load(filename)
    pts = mesh.vertices.view(np.ndarray)
    # data,_ = load_data_modelnet("test")
    # pts = data[2]
    size = pts.max(axis=0) - pts.min(axis=0)
    pts = 2 * pts / size.max()
    pts -= (pts.max(axis=0) + pts.min(axis=0)) / 2
    # print(pts.min(axis=0),pts.max(axis=0))
    # np.save("temp",pts)
    return pts




#initialize network

trainset = pDataset()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,num_workers=2)

parser = argparse.ArgumentParser(description='Point Cloud Recognition')
args = parser.parse_args()
args.k = 3
args.emb_dims = 1024

# encoder_model = DGCNN_cls(args).to(device)
encoder_model = SimplePointnet().to(device)

net = build_network()

# optimizer1 = optim.SGD(encoder_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
optimizer1 = optim.Adam(encoder_model.parameters(), lr=0.1)         

optimizer2 = optim.Adam(net.parameters(), lr=0.0005)         




# training

assert NETWORK == 2
epochs = 1001
for e in range(epochs):
    running_loss = 0
    count = 0
    prev_latent = None
    for pointcloud,data,gt_usdf,derv in trainloader:
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        pointcloud = pointcloud.to(device)
        latent_vector = encoder_model(pointcloud)
        # print(latent_vector)
        data = data.clone().detach().requires_grad_(True).to(device)
        gt_usdf = gt_usdf.to(device)

        output = net(data,latent_vector)
        loss = SAL_Loss(output.t(), gt_usdf)

        predict_derv = torch.autograd.grad(output.sum(),data,retain_graph=True)[0]

        derv = derv.view(-1, derv.shape[-1]).to(device)
        predict_derv = predict_derv.view(-1,predict_derv.shape[-1])
        # print(derv.shape,predict_derv.shape)
        
        der_loss = derv_loss(derv,predict_derv)
        loss = loss + 0.1* der_loss
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        running_loss += loss.item() 
        count += 1
        # if prev_latent != None:
        #     print( (latent_vector-prev_latent).sum())
        # prev_latent = latent_vector
    # if e%50 == 0 and e>0:
    #     with torch.no_grad():
    #         to_mesh(net,device,"xyz"+str(e),latent_vector)
    if e%50 == 0:
        torch.save(encoder_model,"models/encoder"+str(e))
        torch.save(net,"models/sdf"+str(e))
    print(running_loss/count)
    # if e%100 ==0 and e>0:
    #     to_mesh(net,device,"82testing"+str(e))







