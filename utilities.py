import torch
import numpy as np
import scipy.io
import h5py
import sklearn.metrics
from torch_geometric.data import Data
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from torch_geometric.nn import GCNConv
import pdb


class MeshGenerator(object):
    def __init__(self, real_space, mesh_size, attr_features=1,grid_input=np.array([])):
        super(MeshGenerator, self).__init__()

        self.d = len(real_space)  #2
        # self.m = sample_size  #1000
        self.attr_features = attr_features

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            # self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                # grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                # grids.append(np.linspace(real_space[j][0]+(0.5/mesh_size[j]), real_space[j][1]-(0.5/mesh_size[j]), mesh_size[j]))
                self.n *= mesh_size[j]
        
        # self.n is the total number of grid points
        
        self.idx = np.array(range(self.n))
        self.grid=grid_input
        self.grid_sample = self.grid
        

    
    def sample(self, idx): 
        self.idx = torch.tensor(idx)
        self.grid_sample = self.grid[self.idx]
        return self.idx
    
    def get_grid(self):
        return torch.tensor(self.grid_sample, dtype=torch.float)      

    def deduplicate_rows(tensor):
        # This function is used to deduplicate the rows of a tensor
        unique_rows = []
        seen_rows = set()
        for row in tensor:
            row_tuple = tuple(row.tolist())
            if row_tuple not in seen_rows:
                unique_rows.append(row)
                seen_rows.add(row_tuple)
        deduplicated_tensor = torch.stack(unique_rows)
        return deduplicated_tensor

    def ball_connectivity(self, is_forward=False,ns=10,tri_edge=None):
        # 1. Compute pairwise distances between all points in the grid
            self.pwd = sklearn.metrics.pairwise_distances(self.grid_sample) 
            if tri_edge is not None:
                tri_edge = tri_edge.T
            
            
            # 2. Create nearest neighbor connections
            edge_index_1=np.array([])
            edge_index_2=np.array([])
            for i in range(self.grid_sample.shape[0]):
                # For each point i, connect it to its ns+1 nearest neighbors
                edge_index_1=np.append(edge_index_1,np.array([i]).repeat(ns+1))
                # argsort returns indices of ns+1 closest points to point i
                edge_index_2=np.append(edge_index_2,np.argsort(self.pwd[i])[:ns+1]) 
                
            # 3. Combine nearest neighbor connections with triangulation edges
            self.edge_index = np.vstack([edge_index_1,edge_index_2]) # Stack source and target indices
            
            if tri_edge is not None:
                self.edge_index = np.concatenate([self.edge_index,tri_edge],-1)  # Add triangulation edges

            self.edge_index=torch.tensor(self.edge_index)
            
            # Make the graph undirected by adding reverse edges
            self.edge_index = torch.cat([self.edge_index, self.edge_index.flip(0)], dim=1) 
            
            # Remove duplicate edges
            self.edge_index = MeshGenerator.deduplicate_rows(self.edge_index.T).T
            
            # Count the number of edges
            self.n_edges = self.edge_index.shape[1]
            
            # Optional: keep only forward edges if is_forward is True
            if is_forward:
                print(self.edge_index.shape)
                self.edge_index = self.edge_index[:, self.edge_index[0] >= self.edge_index[1]]
                print(self.edge_index.shape)
                self.n_edges = self.edge_index.shape[1]

            return torch.tensor(self.edge_index, dtype=torch.long)
        
    def attributes(self, theta=None):
        # pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        theta = theta[self.idx] 
        edge_attr = np.zeros((self.n_edges, 2 * self.d + 2*self.attr_features+1))
        self.edge_index=torch.tensor(self.edge_index).to(torch.int64)

        for p in range(self.n_edges):
            edge_attr[p,6:7]=self.pwd[self.edge_index[0][p]][self.edge_index[1][p]]
        edge_attr[:, 4:5] = theta[self.edge_index[0]].view(-1, self.attr_features)
        edge_attr[:, 5:6] = theta[self.edge_index[1]].view(-1, self.attr_features)
        edge_attr[:, 0:4] = self.grid_sample[self.edge_index.T].reshape((self.n_edges, -1))

        return torch.tensor(edge_attr, dtype=torch.float)  
    
# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    # def cuda(self):
    #     self.mean = self.mean.cuda()
    #     self.std = self.std.cuda()

    # def cpu(self):
    #     self.mean = self.mean.cpu()
    #     self.std = self.std.cpu()
    def cuda(self,device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        
#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=False, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]  #x.size()=[1,num_indomain]
        
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1) #pred-gd 
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        
        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class RandomMeshGenerator(object):
    def __init__(self, real_space, mesh_size, sample_size):
        super(RandomMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.m = sample_size

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

        if self.m > self.n:
                self.m = self.n

        self.idx = np.array(range(self.n))
        self.grid_sample = self.grid


    def sample(self):
        perm = torch.randperm(self.n)
        self.idx = perm[:self.m]
        self.grid_sample = self.grid[self.idx]
        return self.idx

    def poisson_disk_sample(self, idx): 
        self.idx = torch.tensor(idx)
        self.grid_sample = self.grid[self.idx]
        return self.idx
    
    def get_grid(self):
        return torch.tensor(self.grid_sample, dtype=torch.float)

    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        rbf = np.exp(-pwd**2/sigma**2)
        sample = np.random.binomial(1,rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            else:
                theta = theta[self.idx]
                edge_attr = np.zeros((self.n_edges, 3 * self.d))
                edge_attr[:, 0:2 * self.d] = self.grid_sample[self.edge_index.T].reshape((self.n_edges, -1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[1]]
        else:
            xy = self.grid_sample[self.edge_index.T].reshape((self.n_edges, -1))
            if theta is None:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                theta = theta[self.idx]
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)