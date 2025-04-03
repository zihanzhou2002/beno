import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pprint as pp
from timeit import default_timer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import scatter  # Not included in the latest version
from torchvision.transforms import GaussianBlur
import sys, os
# from utilities import *
from utilities import MeshGenerator,GaussianNormalizer,LpLoss
from util import record_data, to_cpu, to_np_array, make_dir
from BE_MPNN import HeteroGNS
from BE_MPNN_GPS import HeteroGNSGPS
import random
from loguru import logger
import matplotlib.tri as tri
from torch_geometric.data import HeteroData
from data_setup import make_graph
import warnings
warnings.filterwarnings('ignore')
import pdb
fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Args:
    epochs = 1000
    lr = 0.00005
    inspect_interval = 100
    id = "0"
    init_boudary_loc = "regular"
    boundary_encoding = "transformer"
    trans_layer = 3
    boundary_dim = 128
    batch_size = 1
    act = "silu"
    nmlp_layers = 3
    ns = 100
    n_heads = 4
    bd_shape = "4c"

args = Args()
# ===============================================================================   
gblur = GaussianBlur(kernel_size=5, sigma=5)

ntrain = 90
ntest = 10
batch_size = args.batch_size
batch_size2 = args.batch_size
width = 64
ker_width = 256
depth = 4
edge_features = 7
node_features = 10
ns=args.ns
epochs = args.epochs
learning_rate = args.lr
inspect_interval = args.inspect_interval

runtime = np.zeros(2, )
t1 = default_timer()

resolution = 32
s = resolution
n=s**2


trans_layer = args.trans_layer

path = 'Resolution_' + str(s) + '_poisson' + \
    '_ntrain'+str(ntrain)+'_kerwidth'+str(ker_width) + '_bd_enc_' + args.boundary_encoding + str(args.trans_layer) +\
    '_Rolling' + args.init_boudary_loc+  '_shape' + args.bd_shape +'_ns'+str(args.ns)+\
    '_nheads'+str(args.n_heads)+'_bddim'+str(args.boundary_dim)+"_act"+args.act+'lr'+str(args.lr)+'_nmlp_layers'+str(args.nmlp_layers)+'_epochs' + str(args.epochs)

cwd = os.getcwd()
result_path = os.path.join(cwd, "results")
path_model = os.path.join(result_path,path )
make_dir(path_model)

#logger.info('preprocessing finished, time used:{}', t2-t1)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

if args.act == 'leakyrelu':
    activation = nn.LeakyReLU
elif args.act == 'elu':
    activation = nn.ELU
elif args.act == 'relu':
    activation = nn.ReLU
else:
    activation = nn.SiLU

# ===========================================================

modelGPS = HeteroGNSGPS(nnode_in_features = node_features, nnode_out_features = 1, nedge_in_features = edge_features, nmlp_layers=args.nmlp_layers,
             activation = activation,boundary_dim = args.boundary_dim,trans_layer = trans_layer).to(device)
modelTransformer = HeteroGNS(nnode_in_features = node_features, nnode_out_features = 1, nedge_in_features = edge_features, nmlp_layers=args.nmlp_layers,
                activation = activation,boundary_dim = args.boundary_dim,trans_layer = trans_layer).to(device)

#filename_model='Non-zero_Resolution_32_poisson_dataset_4corners_ntrain900_kerwidth256_m01000_radius0.46875_Transformer_layer3_Rollingregular_nd11_nd215_nheads2_bddim128_actsilulr5e-05new_nmlp_layers3_TRANS_ns10_nstep10_bcnorm_0corner'
GPS_4c_filename_model = 'results\\Resolution_32_poisson_ntrain90_kerwidth256_bd_enc_GPS3_Rollingregular_shape4c_ns100_nheads4_bddim128_actsilulr5e-05_nmlp_layers3_epochs1000'
transformer_4c_filename_model = 'results\\Resolution_32_poisson_ntrain90_kerwidth256_bd_enc_transformer3_Rollingregular_shape4c_ns100_nheads4_bddim128_actsilulr5e-05_nmlp_layers3_epochs1000'





myloss = LpLoss(size_average=False)
#u_normalizer.cuda(device)   
GPS_4c_data_record = pickle.load(open(f"{cwd}\\{GPS_4c_filename_model}", "rb"))
transformer_4c_data_record = pickle.load(open(f"{cwd}\\{transformer_4c_filename_model}", "rb"))

modelGPS.load_state_dict(GPS_4c_data_record["state_dict"][-1])
modelTransformer.load_state_dict(transformer_4c_data_record["state_dict"][-1])
#pdb.set_trace()
analysis_record = {}

print(modelGPS)
print(modelTransformer)

modelGPS.eval()
modelTransformer.eval()

# ============================================================= 
shapes = ["0c","1c","2c","3c","4c", "mix"]
DATA_PATH = os.path.join(cwd, "data")

for shape in shapes:
    f_all = np.load(os.path.join(DATA_PATH, "Dirichlet\\RHS_N32_"+ str(shape) +"_100.npy")) # the source function
    sol_all = np.load(os.path.join(DATA_PATH, "Dirichlet\\SOL_N32_"+ str(shape) +"_100.npy")) # the solution
    bc_all=np.load(os.path.join(DATA_PATH, "Dirichlet\\BC_N32_"+ str(shape) +"_100.npy")) # the boundary condition
    
    data_train, data_test, coord_all, u_normalizer = make_graph(f_all, bc_all, sol_all, resolution, ntrain, ntest)
    
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=batch_size2, shuffle=False)

    test_l2_gps = 0.0
    test_mse_gps = 0.0
    test_l2_transformer = 0.0
    test_mse_transformer = 0.0
    print(f"Shape: {shape}")
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            out_gps = modelGPS(batch)
            loss = F.mse_loss(out_gps.view(-1, 1), batch['G1+2'].y.view(-1,1))
            test_mse_gps += loss.item()
            out_gps = u_normalizer.decode(out_gps.view(batch_size2,-1), sample_idx=batch['G1'].sample_idx.view(batch_size2,-1))
            test_l2_gps += myloss(out_gps, batch['G1+2'].y.view(batch_size2, -1)).item()
            
            out_transformer = modelTransformer(batch)
            loss = F.mse_loss(out_transformer.view(-1, 1), batch['G1+2'].y.view(-1,1))
            test_mse_transformer += loss.item()
            out_transformer = u_normalizer.decode(out_transformer.view(batch_size2,-1), sample_idx=batch['G1'].sample_idx.view(batch_size2,-1))
            test_l2_transformer += myloss(out_transformer, batch['G1+2'].y.view(batch_size2, -1)).item()

        for batch in test_loader:
            batch = batch.to(device)
            out_gps = modelGPS(batch)
            loss = F.mse_loss(out_gps.view(-1, 1), batch['G1+2'].y.view(-1,1))
            test_mse_gps += loss.item()
            out_gps = u_normalizer.decode(out_gps.view(batch_size2,-1), sample_idx=batch['G1'].sample_idx.view(batch_size2,-1))
            test_l2_gps += myloss(out_gps, batch['G1+2'].y.view(batch_size2, -1)).item()        
            
            out_transformer = modelTransformer(batch)
            loss = F.mse_loss(out_transformer.view(-1, 1), batch['G1+2'].y.view(-1,1))
            test_mse_transformer += loss.item()
            out_transformer = u_normalizer.decode(out_transformer.view(batch_size2,-1), sample_idx=batch['G1'].sample_idx.view(batch_size2,-1))
            test_l2_transformer += myloss(out_transformer, batch['G1+2'].y.view(batch_size2, -1)).item()
       
        print(f"Test MSE: {test_mse_gps/100:.6f} (GPS) {test_mse_transformer/100:.6f} (Transformer)")
        print(f"Test L2: {test_l2_gps/100:.6f} (GPS) {test_l2_transformer/100:.6f} (Transformer)")
        print("--------------------------------")

        i = 0
        batch = data_train[i]
        out_gps = modelGPS(batch.to(device))
        out_transformer = modelTransformer(batch.to(device))
        # Plot an example
        #out = u_normalizer.decode(out.view(batch_size2,-1), sample_idx=batch['G1'].sample_idx.view(batch_size2,-1))
        plot_file = 'Resolution_' + str(s) + '_poisson' + \
            '_ntrain'+str(ntrain)+'_kerwidth'+str(ker_width) + '_bd_enc_' + args.boundary_encoding + str(args.trans_layer) +\
            '_shape' + str(args.bd_shape) +'_Rolling' + args.init_boudary_loc+'_ns'+str(args.ns)+\
            '_nheads'+str(args.n_heads)+'_bddim'+str(args.boundary_dim)+"_act"+args.act+'lr'+str(args.lr)+'_nmlp_layers'+str(args.nmlp_layers)+'_epochs' + str(args.epochs) +"_test_" + shape + ".png"

        plots_path = os.path.join(cwd, "plots")
        plot_path = os.path.join(plots_path, plot_file)
        
        num_points = len(coord_all[i, :, :])
        length = int(np.sqrt(num_points))
        idx = batch['G1'].sample_idx.cpu().numpy()

        x_mesh = coord_all[i,:, 0].reshape(32, 32)
        y_mesh = coord_all[i,:, 1].reshape(32, 32)
        y_true = np.zeros(num_points)
        y_pred_gps = np.zeros(num_points)
        y_pred_transformer = np.zeros(num_points)
        
        y = batch['G1+2'].y.detach().to('cpu')
        y_true[np.array(idx)] = y.numpy()  # No need to detach or move to CPU again
        y_pred_gps[np.array(idx)] = out_gps.squeeze().cpu().numpy()
        y_pred_transformer[np.array(idx)] = out_transformer.squeeze().cpu().numpy()


        mask = np.zeros((len(coord_all[i, :, :])), dtype=bool)
        mask[idx] = True
        y_true_masked = np.ma.masked_where(~mask, y_true)
        y_pred_gps_masked = np.ma.masked_where(~mask, y_pred_gps)
        y_pred_transformer_masked = np.ma.masked_where(~mask, y_pred_transformer)
        
        fig, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)


        absmax = np.max(np.hstack((np.abs(y_true_masked), np.abs(y_pred_gps_masked), np.abs(y_pred_transformer_masked))))
        
        pcm = ax3.pcolormesh(x_mesh, y_mesh, y_true_masked.reshape(32, 32), shading='auto', cmap="viridis", vmin=-4, vmax=4)

        ax3.set_title("Groundtruth Solution")

        pcm = ax4.pcolormesh(x_mesh, y_mesh, y_pred_transformer_masked.reshape(32, 32), shading='auto', cmap="viridis", vmin=-4, vmax=4)

        ax4.set_title("Predicted Solution (Transformer)")

        pcm = ax5.pcolormesh(x_mesh, y_mesh, y_pred_gps_masked.reshape(32, 32), shading='auto', cmap="viridis", vmin=-4, vmax=4)
        fig.colorbar(pcm, ax=ax5)

        ax5.set_title("Predicted Solution (GPS)")

        #plt.savefig(plot_path)
        plt.show()
        plt.close()


