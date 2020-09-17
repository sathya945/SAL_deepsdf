import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import cKDTree
import random
import torch.optim as optim
import tqdm
# import open3d as o3dcp 
from skimage import measure
from numpy import linalg as LA
import trimesh
import torch.autograd as autograd
import sys
sys.path.append('..')
from utils import load_data
# NETWORK = 0 #SAL
# NETWORK = 1 #IGR
NETWORK = 2 #SAL++

def to_mesh(net,device,filename):
    with torch.no_grad():
        res = 256
        print("predict")
        pts, val = predict(net,res,device)
        volume = val.reshape(res,res,res)
        print(volume.min(),volume.max())
        try : 
            verts, faces, normals, values = measure.marching_cubes_lewiner(volume, 0.0)
            mesh = trimesh.Trimesh(vertices=verts,faces=faces,vertex_normals=normals)
            mesh.export('output/'+filename+'.stl')
        except Exception as err:
            print(err)


def load_data(filename="/home/venkata.sathya/my_code/data/SHREC14/Real/Data/111.obj"):
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
print("proc started")
# points = load_data()


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#Data loading
class pDataset(Dataset):
    
    def __init__(self):
        self.points = load_data()
        self.ptree = cKDTree(self.points)
        self.sigmas = self.ptree.query(self.points, np.int(10.0))[0][:,-1]
        

    def __len__(self):
        return 50
    
    def __getitem__(self, index):
        if NETWORK == 0 :
            rand_idx = np.random.choice(self.points.shape[0],2048)
            rand_pnts = self.points[rand_idx]
            rand_sig = self.sigmas[rand_idx]
            sample = np.concatenate([ rand_pnts +np.expand_dims(rand_sig, -1) * np.random.normal(0.0, 1.0, size=rand_pnts.shape) ,
                                  rand_pnts +  np.random.normal(0.0, 1.0, size=rand_pnts.shape)],
                                axis=0)
            return torch.from_numpy(sample).float(),torch.from_numpy(self.ptree.query(sample)[0]).float()
        if NETWORK == 2:
            rand_idx = np.random.choice(self.points.shape[0],500)
            rand_pnts = self.points[rand_idx]
            rand_sig = self.sigmas[rand_idx]
            sample = np.concatenate([ rand_pnts +np.expand_dims(rand_sig, -1) * np.random.normal(0.0, 1.0, size=rand_pnts.shape) ,
                                  rand_pnts +  np.random.normal(0.0, 1.0, size=rand_pnts.shape)],
                                axis=0)
            distance,indx = self.ptree.query(sample)
            derv =2*( self.ptree.data[indx] - sample )
            return torch.from_numpy(sample).float(),torch.from_numpy(distance).float(),torch.from_numpy(derv).float()
#network

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # self.layer1 = nn.Linear(3,512)
        self.layer1 = nn.Linear(3+128,512)
        self.layer2 = nn.Linear(512,512)
        self.layer3 = nn.Linear(512,512) 
        self.layer4 = nn.Linear(512,509) #509
        self.layer5 = nn.Linear(512,512)
        self.layer6 = nn.Linear(512,512)
        self.layer7 = nn.Linear(512,512)
        self.layer8 = nn.Linear(512,1)


    def forward(self, x):
        if x.shape[0] == 1:
            x = x.view(-1, x.shape[-1])
        latent_vector = torch.zeros(1,128,device=device)
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
    # print(t1.shape)
    # print(t2.shape)
    # print(torch.min(t1,t2).shape)
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


def predict(net, res,device):
    x = np.linspace(-1.2, 1.2, res)
    y = np.linspace(-1.2, 1.2, res)
    z = np.linspace(-1.2, 1.2, res)
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)
    pts = np.stack((X, Y, Z), axis=1)
    # print(pts.shape)
    pts = pts.reshape(4096*4, -1, 3)
    # print(pts.shape)
    val = []
    net.eval()
    for p in tqdm.tqdm(pts):
        v = net(torch.Tensor(p).to(device))
        v = v.reshape(-1).detach().cpu().numpy()
        val.append(v)
    pts = pts.reshape((-1, 3))
    val = np.concatenate(val)
    return pts, val

#initialize network

trainset = pDataset()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1)

net = build_network()

optimizer = optim.Adam(net.parameters(), lr=0.0001)         





# training

assert NETWORK == 2
epochs = 1001
for e in range(epochs):
    running_loss = 0
    count = 0
    for data,gt_usdf,derv in trainloader:
        # print(data.shape,gt_usdf.shape,derv.shape)
        data = data.clone().detach().requires_grad_(True).to(device)
        
        gt_usdf = gt_usdf.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = SAL_Loss(output.t(), gt_usdf)

        predict_derv = torch.autograd.grad(output.sum(),data,retain_graph=True)[0]

        derv = derv.view(-1, derv.shape[-1]).to(device)
        predict_derv = predict_derv.view(-1,predict_derv.shape[-1])
        # print(derv.shape,predict_derv.shape)
        der_loss = derv_loss(derv,predict_derv)
        loss = loss + der_loss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() 
        count += 1
    print(running_loss/count)
    if e%100 ==0 and e>0:
        to_mesh(net,device,"dataset"+str(e))

# training

# assert NETWORK == 1
# epochs = 1001
# for e in range(epochs):
#     running_loss = 0
#     count = 0
#     for data,gt_usdf,noise in trainloader:
#         # print(data.shape)
#         data = data.to(device)
#         noise = noise.clone().detach().requires_grad_(True).to(device)
#         gt_usdf = gt_usdf.to(device)
#         optimizer.zero_grad()
#         output = net(data)
#         loss_data = SAL_Loss(output.t(), gt_usdf)
#         loss = loss_data
#         # print(gt_usdf.max())
#         output = net(noise)
#         der  = torch.autograd.grad(output.sum(),noise,retain_graph=True)[0]
#         der = der.view(-1, der.shape[-1])
#         # # print(der.norm(2,dim=1).shape)
#         eikonal_loss = ((der.norm(2, dim=0) - 1) ** 2).mean()
#         loss = loss_data + 0.1 * eikonal_loss
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item() 
#         count += 1
#     print(running_loss/count)
#     if e%50 ==0 :
#         to_mesh(net,device,"testingrand"+str(e))







