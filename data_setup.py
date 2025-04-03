import numpy as np
import torch
from torchvision.transforms import GaussianBlur
from utilities import MeshGenerator,GaussianNormalizer,LpLoss
from util import record_data, to_cpu, to_np_array, make_dir
import matplotlib.tri as tri
from torch_geometric.data import HeteroData


def set_dataset(f_all,bc_all,sol_all,resolution,ntrain):
    n = resolution*resolution
    gblur = GaussianBlur(kernel_size=5, sigma=5)
    cells_state=f_all[:,:,3] # node type \in {0,1,2,3}
    coord_all=f_all[:,:,0:2] # all node corrdinate

    bc_euco=bc_all[:,:,0:2]  # boundary corrdinate
    bc_value=bc_all[:,:,2].reshape(-1,128,1)   # boundary value
    bc_value=torch.tensor(bc_value)
    bc_value_1=bc_value[0:900,:,:]
    bc_euco=torch.tensor(bc_euco)
    bcv_normalizer = GaussianNormalizer(bc_value_1)
    bc_value = bcv_normalizer.encode(bc_value)
    bc_euco= to_np_array(torch.cat([bc_euco,bc_value],dim=-1))

    all_a = f_all[:,:,2] # actual source function values
    all_a_smooth = to_np_array(gblur(torch.tensor(all_a.reshape(all_a.shape[0], resolution, resolution))).flatten(start_dim=1))
    all_a_reshape = all_a_smooth.reshape(-1, resolution, resolution)

    # Calculate gradient using finite difference
    all_a_gradx = np.concatenate([
        all_a_reshape[:,1:2] - all_a_reshape[:,0:1],
        (all_a_reshape[:,2:] - all_a_reshape[:,:-2]) / 2,
        all_a_reshape[:,-1:] - all_a_reshape[:,-2:-1],
    ], 1)
    all_a_gradx = all_a_gradx.reshape(-1, n)
    all_a_grady = np.concatenate([
        all_a_reshape[:,:,1:2] - all_a_reshape[:,:,0:1],
        (all_a_reshape[:,:,2:] - all_a_reshape[:,:,:-2]) / 2,
        all_a_reshape[:,:,-1:] - all_a_reshape[:,:,-2:-1],
    ], 2)
    all_a_grady = all_a_grady.reshape(-1, n)
    all_u = sol_all[:,:,0]

    # Setting up train and test dataset
    train_a = torch.FloatTensor(all_a[:ntrain])  # [num_train, 4096]
    train_a_smooth = torch.FloatTensor(all_a_smooth[:ntrain]) # [num_train, 4096]
    train_a_gradx = torch.FloatTensor(all_a_gradx[:ntrain])   # [num_train, 4096]
    train_a_grady = torch.FloatTensor(all_a_grady[:ntrain])   # [num_train, 4096]
    train_u = torch.FloatTensor(all_u[:ntrain])  # [num_train, 4096]
    test_a = torch.FloatTensor(all_a[ntrain:])
    test_a_smooth = torch.FloatTensor(all_a_smooth[ntrain:])
    test_a_gradx = torch.FloatTensor(all_a_gradx[ntrain:])
    test_a_grady = torch.FloatTensor(all_a_grady[ntrain:])
    test_u = torch.FloatTensor(all_u[ntrain:])

    bc_euco_train=bc_euco[:ntrain,:,:]
    bc_euco_test=bc_euco[ntrain:,:,:]
    
# Process in-domain and out-of-domain data separately
    indomain_a = np.array([])
    indomain_u = np.array([])

    for j in range(ntrain):
        outdomain_idx=np.array([],dtype=int)
        indomain_idx=np.array([],dtype=int)
        for p in range(f_all.shape[1]):
            # If the cell is not in-domain, add it to the out-of-domain index
            if (cells_state[j][p]!=0):
                outdomain_idx=np.append(outdomain_idx,int(p))

        # If the cell is in-domain, add it to the in-domain index
        indomain_idx = list(set([i for i in range(resolution*resolution)]) - set(list(outdomain_idx)))
        indomain_u = np.append(indomain_u,sol_all[j][indomain_idx])
        indomain_a = np.append(indomain_a,f_all[j][indomain_idx][:,2])

    # Convert to tensors
    indomain_u=torch.tensor(indomain_u)
    indomain_a=torch.tensor(indomain_a)

    # Initialize normalizers for in-domain data
    a_normalizer = GaussianNormalizer(indomain_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)


    # Normalize the smoothed source function
    as_normalizer = GaussianNormalizer(train_a_smooth)
    train_a_smooth = as_normalizer.encode(train_a_smooth)
    test_a_smooth = as_normalizer.encode(test_a_smooth)

    # Normalize the gradient of the source function
    agx_normalizer = GaussianNormalizer(train_a_gradx)
    train_a_gradx = agx_normalizer.encode(train_a_gradx)
    test_a_gradx = agx_normalizer.encode(test_a_gradx)
    agy_normalizer = GaussianNormalizer(train_a_grady)
    train_a_grady = agy_normalizer.encode(train_a_grady)
    test_a_grady = agy_normalizer.encode(test_a_grady)

    # Normalize the solution function
    u_normalizer = GaussianNormalizer(x=indomain_u)
    train_u = u_normalizer.encode(train_u)

        
    train_initial_data = (train_a, train_a_smooth, train_a_gradx, train_a_grady, train_u, bc_euco_train)
    test_initial_data = (test_a, test_a_smooth, test_a_gradx, test_a_grady, test_u, bc_euco_test)
    
    return train_initial_data, test_initial_data, coord_all, cells_state, u_normalizer

def make_graph(f_all, bc_all, sol_all, resolution, ntrain, ntest):
    s = resolution
    train_initial_data, test_initial_data, coord_all, cells_state, u_normalizer = set_dataset(f_all, bc_all, sol_all, resolution, ntrain)
    train_a, train_a_smooth, train_a_gradx, train_a_grady, train_u, bc_euco_train = train_initial_data
    test_a, test_a_smooth, test_a_gradx, test_a_grady, test_u, bc_euco_test = test_initial_data
    
    grid_input=f_all[-1,:,0:2]  # grid_inpus is of shape 1024*2 which is the coordinates of all nodes
    meshgenerator = MeshGenerator([[0,1],[0,1]],[s,s], grid_input = grid_input)
# Construct graph data for training
    data_train = []
    for j in range(ntrain):
        mesh_idx_temp=[p for p in range(resolution**2)]
        outdomain_idx=np.array([])
        for p in range(f_all.shape[1]):
            if (cells_state[j][p]!=0):  # = 0 means the cell is in domain
                outdomain_idx=np.append(outdomain_idx,p)
        for p in range(len(outdomain_idx)):
                mesh_idx_temp.remove(outdomain_idx[p])


        dist2bd_x=np.array([0,0])[np.newaxis,:]
        dist2bd_y=np.array([0,0])[np.newaxis,:]

        # Compute distance to boundary for each point in the mesh
        for p in range(len(mesh_idx_temp)):
            indomain_x = coord_all[j][mesh_idx_temp[p]][0] # e.g. 0.5
            indomain_y = coord_all[j][mesh_idx_temp[p]][1] # e.g. 0.3

            # Find boundary points that have the same x-coordinate (0.5)
            horizon_bd_y = np.where(bc_euco_train[j,:,0].round(4) == indomain_x.round(4))[0]

            # Calculate distances to these boundary points
            dist2bd_y_temp = np.array(
                [np.abs(bc_euco_train[j,horizon_bd_y[0],1] - indomain_y),
                np.abs(bc_euco_train[j,horizon_bd_y[1],1] - indomain_y)
                ]
            )
            dist2bd_y = np.vstack([dist2bd_y,dist2bd_y_temp[np.newaxis,:]])

            # Find boundary points that have the same y-coordinate (0.3)
            horizon_bd_x = np.where(bc_euco_train[j,:,1].round(4) == indomain_y.round(4))[0]

            # Calculate distances to these boundary points
            dist2bd_x_temp = np.array(
                [np.abs(bc_euco_train[j,horizon_bd_x[0],0] - indomain_x),
                np.abs(bc_euco_train[j,horizon_bd_x[1],0] - indomain_x)
                ]
            )
            dist2bd_x = np.vstack([dist2bd_x,dist2bd_x_temp[np.newaxis,:]])


        dist2bd_y = torch.tensor(dist2bd_y[1:]).float()
        dist2bd_x = torch.tensor(dist2bd_x[1:]).float() # [num, 2]


        idx = meshgenerator.sample(mesh_idx_temp)  #这一步只是将indomain的idx输入，并赋给get_grid
        grid = meshgenerator.get_grid()

        xx=to_np_array(grid[:,0])
        yy=to_np_array(grid[:,1])
        triang = tri.Triangulation(xx, yy)
        tri_edge = triang.edges

        edge_index = meshgenerator.ball_connectivity(ns=10,tri_edge=tri_edge)
        edge_attr = meshgenerator.attributes(theta=train_a[j,:]) # theta is the source function, used for attributes
        train_x = torch.cat([grid, train_a[j, idx].reshape(-1, 1),
                                train_a_smooth[j, idx].reshape(-1, 1), train_a_gradx[j, idx].reshape(-1, 1),
                                train_a_grady[j, idx].reshape(-1, 1), dist2bd_x,dist2bd_y
                                ], dim=1)
        train_x_2 = torch.cat([grid, torch.zeros([grid.shape[0],4]), dist2bd_x,dist2bd_y
                                ], dim=1)

        bd_coord_input = torch.tensor(bc_euco_train[j])

        bd_coord_input_1=bd_coord_input.clone()
        bd_coord_input_1[:,2]=0

        # Add edge features for boundary conditions
        bd_idx = [p for p in range(len(bd_coord_input))]
        bd_grid_input = bd_coord_input[:, 0:2]

        # Create a generator
        bd_meshgenerator = MeshGenerator([[0,1],[0,1]],[s,s], grid_input = bd_grid_input)
        bd_idx_mesh = bd_meshgenerator.sample(bd_idx)
        bd_grid = bd_meshgenerator.get_grid()

        # Boudnary edge features for non zero
        bd_edge_index = bd_meshgenerator.ball_connectivity(ns=3)
        bd_edge_attr = bd_meshgenerator.attributes(theta=bd_coord_input[:,2]) # theta is the source function, used for attributes

        # Boudnary edge features for non zero
        bd_edge_index_1 = bd_meshgenerator.ball_connectivity(ns=3)
        bd_edge_attr_1 = bd_meshgenerator.attributes(theta=bd_coord_input_1[:,2]) # theta is the source function, used for attributes

        data=HeteroData()
        data['G1'].x=train_x #node features ▲u=f
        data['G1'].boundary=bd_coord_input_1 #boundary value=0
        data['G1'].edge_features=edge_attr
        data['G1'].sample_idx=idx
        data['G1'].edge_index=edge_index
        data['G1'].bd_edge_index=bd_edge_index
        data['G1'].bd_edge_features=bd_edge_attr

        data['G2'].x=train_x_2  ##node features ▲u=0
        data['G2'].boundary=bd_coord_input #boundary value=g(x)
        data['G2'].edge_features=edge_attr
        data['G2'].sample_idx=idx
        data['G2'].edge_index=edge_index
        data['G2'].bd_edge_index=bd_edge_index_1
        data['G2'].bd_edge_features=bd_edge_attr_1

        data['G1+2'].y=train_u[j, idx]

        data_train.append(data)


    data_test = []
    for j in range(ntest):
        mesh_idx_temp=[p for p in range(resolution**2)]
        outdomain_idx=np.array([])
        for p in range(f_all.shape[1]):
            if (cells_state[j+ntrain][p]!=0):
                outdomain_idx=np.append(outdomain_idx,p)

        for p in range(len(outdomain_idx)):
                mesh_idx_temp.remove(outdomain_idx[p])

        dist2bd_x=np.array([0,0])[np.newaxis,:]
        dist2bd_y=np.array([0,0])[np.newaxis,:]
        for p in range(len(mesh_idx_temp)):
            indomain_x = coord_all[j+ntrain][mesh_idx_temp[p]][0]
            indomain_y = coord_all[j+ntrain][mesh_idx_temp[p]][1]

            horizon_bd_y = np.where(bc_euco_test[j,:,0].round(4) == indomain_x.round(4))[0]

            dist2bd_y_temp = np.array(
                [np.abs(bc_euco_test[j,horizon_bd_y[0],1] - indomain_y),
                np.abs(bc_euco_test[j,horizon_bd_y[1],1] - indomain_y)
                ]
            )
            dist2bd_y = np.vstack([dist2bd_y,dist2bd_y_temp[np.newaxis,:]])
            horizon_bd_x = np.where(bc_euco_test[j,:,1].round(4) == indomain_y.round(4))[0]

            dist2bd_x_temp = np.array(
                [np.abs(bc_euco_test[j,horizon_bd_x[0],0] - indomain_x),
                np.abs(bc_euco_test[j,horizon_bd_x[1],0] - indomain_x)
                ]
            )
            dist2bd_x = np.vstack([dist2bd_x,dist2bd_x_temp[np.newaxis,:]])
        dist2bd_y = torch.tensor(dist2bd_y[1:]).float()
        dist2bd_x = torch.tensor(dist2bd_x[1:]).float() # [num, 2]



        idx = meshgenerator.sample(mesh_idx_temp)
        grid = meshgenerator.get_grid()

        xx=to_np_array(grid[:,0])
        yy=to_np_array(grid[:,1])
        triang = tri.Triangulation(xx, yy)
        tri_edge = triang.edges

        edge_index = meshgenerator.ball_connectivity(ns=10,tri_edge=tri_edge)
        edge_attr = meshgenerator.attributes(theta=test_a[j,:])

        test_x = torch.cat([grid, test_a[j, idx].reshape(-1, 1),
                            test_a_smooth[j, idx].reshape(-1, 1), test_a_gradx[j, idx].reshape(-1, 1),
                            test_a_grady[j, idx].reshape(-1, 1),dist2bd_x,dist2bd_y
                        ], dim=1)
        test_x_2 = torch.cat([grid, torch.zeros([grid.shape[0],4]), dist2bd_x,dist2bd_y
                                ], dim=1)

        # Add edge features for boundary conditions
        bd_idx = [p for p in range(len(bd_coord_input))]

        bd_grid_input = bd_coord_input[:, 0:2]

        # Create a generator
        bd_meshgenerator = MeshGenerator([[0,1],[0,1]],[s,s], grid_input = bd_grid_input)
        bd_idx_mesh = bd_meshgenerator.sample(bd_idx)
        bd_grid = bd_meshgenerator.get_grid()

        # Boudnary edge features for non zero
        bd_edge_index = bd_meshgenerator.ball_connectivity(ns=3)
        bd_edge_attr = bd_meshgenerator.attributes(theta=bd_coord_input[:,2]) # theta is the source function, used for attributes

        # Boudnary edge features for non zero
        bd_edge_index_1 = bd_meshgenerator.ball_connectivity(ns=3)
        bd_edge_attr_1 = bd_meshgenerator.attributes(theta=bd_coord_input_1[:,2]) # theta is the source function, used for attributes

        data=HeteroData()
        data['G1'].x=test_x #node features ▲u=f
        data['G1'].boundary=bd_coord_input_1 #boundary value=0
        data['G1'].edge_features=edge_attr
        data['G1'].sample_idx=idx
        data['G1'].edge_index=edge_index
        data['G1'].bd_edge_index=bd_edge_index
        data['G1'].bd_edge_features=bd_edge_attr

        data['G2'].x=test_x_2  ##node features ▲u=0
        data['G2'].boundary=bd_coord_input #boundary value=g(x)
        data['G2'].edge_features=edge_attr
        data['G2'].sample_idx=idx
        data['G2'].edge_index=edge_index
        data['G2'].bd_edge_index=bd_edge_index_1
        data['G2'].bd_edge_features=bd_edge_attr_1

        data['G1+2'].y=test_u[j, idx]

        data_test.append(data)
    
    return data_train, data_test, coord_all, u_normalizer









