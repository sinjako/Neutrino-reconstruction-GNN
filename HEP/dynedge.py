import torch
from torch_scatter import scatter_mean
from torch_scatter import scatter_sum
from torch_scatter import scatter_min
from torch_scatter import scatter_max
# torch.autograd.set_detect_anomaly(True)
from torch_cluster import knn_graph
from torch_geometric.nn import EdgeConv

class dynedge(torch.nn.Module):                                                     
    def __init__(self, input_size, output_size = 1,x_col = 0,y_col = 1,z_col = 2, mode = 'custom', k = 4, c=3, device = 'cuda' ):
        ####
        # INPUTS:
        # input_size    : INTEGER - dimension of input tensor
        # output_size   : INTEGER - dimension of output tensor
        # x             : INTEGER - column index in input tensor for x-coordinate of DOM position. DEFAULT - 0
        # y             : INTEGER - column index in input tensor for y-coordinate of DOM position. DEFAULT - 1
        # z             : INTEGER - column index in input tensor for z-coordinate of DOM position. DEFAULT - 2
        # k             : INTEGER - number of neighbours. DEFAULT - 8
        # device        : STRING  - the device ID on which the model is run. DEFAULT - 'cuda'
        # c             : INTEGER - the dimension factor. DEFAULT - 3
        # target        : STRING  - specifies which version of dynedge to run. ['energy', 'angle', 'classifcation']
        #                  target = energy         : Regresses energy_log10. Use in conjuction with 'logcosh' loss function.
        #                  target = angle          : Regresses either zenith or azimuth. Use in conjunction with 'vonMisesSineCosineLoss
        #                  target = pid            : Use in conjuction with torch.loss.CrossEntropyLoss
        #                 
                                                                                   
        super(dynedge, self).__init__()
        self.k = k
        self.mode = mode
        self.device = device
        self.pos_idx = [x_col,y_col,z_col]
        
        
        l1, l2, l3, l4, l5,l6,l7 = input_size,c*16*2,c*32*2,c*42*2,c*32*2,c*16*2,output_size
        
        if mode == 'angle':
            output_size = 3            # VonMisesSineCosineLoss requires three dimensionsional output
        if mode == 'energy':
            output_size = 1            # logcosh requires one-dimensional output
        if mode == 'pid':
            output_size = 2            # CrossEntropyLoss requires two-dimensional output
        
        self.nn_conv1 = torch.nn.Sequential(torch.nn.Linear(l1*2,l2),torch.nn.LeakyReLU(),torch.nn.Linear(l2,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add = EdgeConv(self.nn_conv1,aggr = 'add')

        self.nn_conv2 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add2 = EdgeConv(self.nn_conv2,aggr = 'add')

        self.nn_conv3 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add3 = EdgeConv(self.nn_conv3,aggr = 'add')

        self.nn_conv4 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add4 = EdgeConv(self.nn_conv4,aggr = 'add')

        self.nn1 = torch.nn.Linear(l3*4 + l1,l4)                                               
        self.nn2   = torch.nn.Linear(l4,l5)
        self.nn3 =  torch.nn.Linear(4*l5,l6)
        self.nn4 = torch.nn.Linear(l6,l7)
        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
                                          
                                                                                
    def forward(self, data):
        k = self.k        
        device = self.device
        mode   = self.mode
        pos_idx = self.pos_idx
        #changing xtype to float, change back after saving graphs properly
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        edge_index = knn_graph(x=x[:,pos_idx],k=k,batch=batch).to(device)

        a = self.conv_add(x,edge_index)
        
        edge_index = knn_graph(x=a[:,pos_idx],k=k,batch=batch).to(device)
        "check if this recalculation of edge indices is correct, maybe you can do it over all of x"
        b = self.conv_add2(a,edge_index)

        edge_index = knn_graph(x=b[:,pos_idx],k=k,batch=batch).to(device)
        
        c = self.conv_add3(b,edge_index)

        edge_index = knn_graph(x=c[:,pos_idx],k=k,batch=batch).to(device)
        
        d = self.conv_add4(c,edge_index)

        x = torch.cat((x,a,b,c,d),dim = 1) 
        del a,b,c,d
        x = self.nn1(x)
        x = self.relu(x)
        x = self.nn2(x)
        
        a,_ = scatter_max(x, batch, dim = 0)
        b,_ = scatter_min(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        d = scatter_mean(x,batch,dim= 0)
        x = torch.cat((a,b,c,d),dim = 1)
        
        x = self.relu(x)
        x = self.nn3(x)
        
        x = self.relu(x)
        x = self.nn4(x)
        
        if mode == 'angle':
            x[:,0] = self.tanh(x[:,0])
            x[:,1] = self.tanh(x[:,1])
        

        return x