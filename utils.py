
import trimesh
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from skimage import measure
from scipy.spatial import cKDTree
import glob 
import os
import h5py
from torch.utils.data import Dataset, DataLoader


def to_mesh(net, device, filename, latent_vector=None,di="output/"):
    with torch.no_grad():
        res = 256
        print("predict")
        pts, val = predict(net, res, device, latent_vector)
        volume = val.reshape(res, res, res)
        print(volume.min(), volume.max())
        try:
            verts, faces, normals, values = measure.marching_cubes_lewiner(
                volume, 0.0)
            mesh = trimesh.Trimesh(
                vertices=verts, faces=faces, vertex_normals=normals)
            mesh.export(di+filename+'.stl')
        except Exception as err:
            print(err)


def load_data(filename):
    print(filename)
    mesh = trimesh.load(filename)
    pts = mesh.vertices.view(np.ndarray)
    size = pts.max(axis=0) - pts.min(axis=0)
    pts = 2 * pts / size.max()
    # pts = 2 * pts / 1.9
    # pts -= (pts.max(axis=0) + pts.min(axis=0)) / 2
    pts -= (pts.max(axis=0) + pts.min(axis=0)) / 2
    return pts


def predict(net, res, device, latent_vector=None):
    x = np.linspace(-1.2, 1.2, res)
    y = np.linspace(-1.2, 1.2, res)
    z = np.linspace(-1.2, 1.2, res)
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)
    pts = np.stack((X, Y, Z), axis=1)
    pts = pts.reshape(4096*4, -1, 3)
    val = []
    net.eval()
    for p in tqdm.tqdm(pts):
        if latent_vector == None:
            v = net(torch.Tensor(p).to(device))
        else:
            v = net(torch.Tensor(p).to(device), latent_vector)
        v = v.reshape(-1).detach().cpu().numpy()
        val.append(v)
    pts = pts.reshape((-1, 3))
    val = np.concatenate(val)
    return pts, val


class Modelnet(Dataset):
    def __init__(self, partition='train'):
        self.point_clouds, self.label = load_data_modelnet(partition)
        self.ptrees = []
        self.sigmas_list = []
        for i in range(self.point_clouds.shape[0]):
            self.ptrees.append(cKDTree(self.point_clouds[i]))
            self.sigmas_list.append(self.ptrees[i].query(
                self.point_clouds[i], np.int(30.0))[0][:, -1])
        self.count = self.point_clouds.shape[0]
        print(self.count)
    def __getitem__(self, item3):
        random_shape_no = np.random.randint(self.count)
        rand_idx = np.random.choice(self.point_clouds[random_shape_no].shape[0], 500)
        rand_pnts = self.point_clouds[random_shape_no][rand_idx]
        rand_sig = self.sigmas_list[random_shape_no][rand_idx]
        sample = np.concatenate([rand_pnts + np.expand_dims(rand_sig, -1) * np.random.normal(0.0, 1.0, size=rand_pnts.shape),
                                  rand_pnts + np.random.normal(0.0, 1.0, size=rand_pnts.shape)],
                                axis=0)
        distance, indx = self.ptrees[random_shape_no].query(sample)
        derv = 2*(self.ptrees[random_shape_no].data[indx] - sample)
        return torch.from_numpy(self.point_clouds[random_shape_no].T).float(), torch.from_numpy(sample).float(), torch.from_numpy(distance).float(), torch.from_numpy(derv).float()

    def __len__(self):
        return 100
# data_files = ['00158.obj', '00120.obj', '00136.obj', '00203.obj', '00030.obj', '00055.obj', '00000.obj', '00023.obj', '00229.obj', '00084.obj', '00163.obj', '00062.obj', '00092.obj']
# data_path = "data/scripts/50002_jiggle_on_toes/"

# for fil in data_files:
#     load_data(data_path+fil)


# def load_data_modelnet(partition):
#     DATA_DIR = "../data/"
#     all_data = []
#     all_label = []
#     for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' %partition)):
#         f = h5py.File(h5_name)
#         data = f['data'][:].astype('float32')
#         label = f['label'][:].astype('int64')
#         f.close()
#         all_data.append(data)
#         all_label.append(label)
#     all_data = np.concatenate(all_data, axis=0)
#     all_label = np.concatenate(all_label, axis=0)
#     return all_data[np.where(all_label ==8)[0]], all_label[np.where(all_label ==5)[0]]



class SHREC(Dataset):
    def __init__(self,file_list=glob.glob("/home/venkata.sathya/my_code/data/SHREC14/Real/Data/*1.*")[:1],path="/home/venkata.sathya/my_code/data/SHREC14/Real/Data/"):
        self.file_list = file_list
        print(file_list)
        self.path = ""
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,index):
        pts = trimesh.load(self.path+self.file_list[index]).vertices.view(np.ndarray)
        ptree = cKDTree(pts)
        sigmas = ptree.query(pts,np.int(10.0))[0][:, -1]
        rand_idx = np.random.choice(pts.shape[0],500)
        rand_pnts = pts[rand_idx]
        rand_sig = sigmas[rand_idx]
        sample = np.concatenate([ rand_pnts +np.expand_dims(rand_sig, -1) * np.random.normal(0.0, 1.0, size=rand_pnts.shape) ,
                              rand_pnts +  np.random.normal(0.0, 1.0, size=rand_pnts.shape)],
                            axis=0)
        distance,indx = ptree.query(sample)
        derv =2*( ptree.data[indx] - sample )
        return torch.from_numpy(pts).float() , torch.from_numpy(sample).float(),torch.from_numpy(distance).float(),torch.from_numpy(derv).float()