import torch
from torch_scatter import scatter_mean
from torch_scatter import scatter_sum
from torch_scatter import scatter_min
from torch_scatter import scatter_max
# torch.autograd.set_detect_anomaly(True)
from torch_cluster import knn_graph
from torch_geometric.nn import EdgeConv,GatedGraphConv,BatchNorm,dense_diff_pool,TopKPooling,GlobalAttention,EdgePooling,SAGPooling

class dynedge(torch.nn.Module):                                                     
    def __init__(self, input_size=4, output_size = 1,x_col = 0,y_col = 1,z_col = 2, mode = 'custom', k = [4,4,4,4], c=3, device = 'cuda' ):
        ####
        # INPUTS:
        # input_size    : INTEGER - dimension of input tensor.DEFAUlT -4
        # output_size   : INTEGER - dimension of output tensor. DEFAUlT -1
        # x             : INTEGER - column index in input tensor for x-coordinate of DOM position. DEFAULT - 0
        # y             : INTEGER - column index in input tensor for y-coordinate of DOM position. DEFAULT - 1
        # z             : INTEGER - column index in input tensor for z-coordinate of DOM position. DEFAULT - 2
        # k             : INTEGER - number of neighbours. DEFAULT - 4
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
        
        if mode == 'angle':
            output_size = 3            # VonMisesSineCosineLoss requires three dimensionsional output
        if mode == 'energy':
            output_size = 1            # logcosh requires one-dimensional output
        if mode == 'pid':
            output_size = 2            # CrossEntropyLoss requires two-dimensional output
        l1, l2, l3, l4, l5,l6,l7 = input_size,c*16*2,c*32*2,c*42*2,c*32*2,c*16*2,output_size
        

        
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
        self.Softmax=torch.nn.Softmax(dim=0)
                                                                                
    def forward(self, data):
        k = self.k        
        device = self.device
        mode   = self.mode
        pos_idx = self.pos_idx
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        edge_index = knn_graph(x=x[:,pos_idx],k=k[0],batch=batch).to(device)
        
        
        a = self.conv_add(x,edge_index)


        edge_index = knn_graph(x=a[:,pos_idx],k=k[1],batch=batch).to(device)
        
        b = self.conv_add2(a,edge_index)

        edge_index = knn_graph(x=b[:,pos_idx],k=k[2],batch=batch).to(device)
        
        c = self.conv_add3(b,edge_index)

        edge_index = knn_graph(x=c[:,pos_idx],k=k[3],batch=batch).to(device)
        
        d = self.conv_add4(c,edge_index)
  
        x = torch.cat((x,a,b,c,d),dim = 1) 
        del a,b,c,d
        x = self.nn1(x)
        x = self.relu(x)
        x = self.nn2(x)
        

        x=self.pool4cat(x,batch)
        x = self.relu(x)
        x = self.nn3(x)
        
        x = self.relu(x)
        x = self.nn4(x)
        
        if mode == 'angle':
            x[:,0] = self.tanh(x[:,0])
            x[:,1] = self.tanh(x[:,1])

        return x
    def pool4cat(self,x,batch):
        a,_ = scatter_max(x, batch, dim = 0)
        b,_ = scatter_min(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        d = scatter_mean(x,batch,dim= 0)
        x = torch.cat((a,b,c,d),dim = 1)
        return x

class dynedgegauss(torch.nn.Module):                                                     
    def __init__(self, input_size=4, output_size = 1,x_col = 0,y_col = 1,z_col = 2, mode = 'custom', k = [4,4,4,4], c=3, device = 'cuda',eps=1e-15 ):
        ####
        # INPUTS:
        # input_size    : INTEGER - dimension of input tensor.DEFAUlT -4
        # output_size   : INTEGER - dimension of output tensor. DEFAUlT -1
        # x             : INTEGER - column index in input tensor for x-coordinate of DOM position. DEFAULT - 0
        # y             : INTEGER - column index in input tensor for y-coordinate of DOM position. DEFAULT - 1
        # z             : INTEGER - column index in input tensor for z-coordinate of DOM position. DEFAULT - 2
        # k             : INTEGER - number of neighbours. DEFAULT - 4
        # device        : STRING  - the device ID on which the model is run. DEFAULT - 'cuda'
        # c             : INTEGER - the dimension factor. DEFAULT - 3
        # target        : STRING  - specifies which version of dynedge to run. ['energy', 'angle', 'classifcation']
        #                  target = energy         : Regresses energy_log10. Use in conjuction with 'logcosh' loss function.
        #                  target = angle          : Regresses either zenith or azimuth. Use in conjunction with 'vonMisesSineCosineLoss
        #                  target = pid            : Use in conjuction with torch.loss.CrossEntropyLoss
        #                 
                                                                                   
        super(dynedgegauss, self).__init__()
        self.k = k
        self.mode = mode
        self.device = device
        self.pos_idx = [x_col,y_col,z_col]
        self.eps=eps
        
        if mode == 'angle':
            output_size = 3            # VonMisesSineCosineLoss requires three dimensionsional output
        if mode == 'energy':
            output_size = 1            # logcosh requires one-dimensional output
        if mode == 'pid':
            output_size = 2            # CrossEntropyLoss requires two-dimensional output
        if mode=='gauss':
            output_size=2
        l1, l2, l3, l4, l5,l6,l7 = input_size,c*16*2,c*32*2,c*42*2,c*32*2,c*16*2,output_size
        

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
        self.Softmax=torch.nn.Softmax(dim=0)
        self.reluhard=torch.nn.ReLU()
    def forward(self, data):
        k = self.k        
        device = self.device
        mode   = self.mode
        pos_idx = self.pos_idx
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        edge_index = knn_graph(x=x[:,pos_idx],k=k[0],batch=batch).to(device)
        
        
        a = self.conv_add(x,edge_index)


        edge_index = knn_graph(x=a[:,pos_idx],k=k[1],batch=batch).to(device)
        
        b = self.conv_add2(a,edge_index)

        edge_index = knn_graph(x=b[:,pos_idx],k=k[2],batch=batch).to(device)
        
        c = self.conv_add3(b,edge_index)

        edge_index = knn_graph(x=c[:,pos_idx],k=k[3],batch=batch).to(device)
        
        d = self.conv_add4(c,edge_index)
  
        x = torch.cat((x,a,b,c,d),dim = 1) 
        del a,b,c,d
        x = self.nn1(x)
        x = self.relu(x)
        x = self.nn2(x)
        

        x=self.pool4cat(x,batch)
        x = self.relu(x)
        x = self.nn3(x)
        
        x = self.relu(x)
        x = self.nn4(x)
        # print (x.size())
        y=torch.unsqueeze(x[:,0],1)
        z=self.reluhard(torch.unsqueeze(x[:,1],1))+self.eps
        # print (x.size(),y.size(),z.size())
        x= torch.hstack((y,z))
        # print (x.size())
        del y,z
        if mode == 'angle':
            x[:,0] = self.tanh(x[:,0])
            x[:,1] = self.tanh(x[:,1])
        

        return x
    def pool4cat(self,x,batch):
        a,_ = scatter_max(x, batch, dim = 0)
        b,_ = scatter_min(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        d = scatter_mean(x,batch,dim= 0)
        x = torch.cat((a,b,c,d),dim = 1)
        return x

class dynedgeSAG(torch.nn.Module):                                                     
    def __init__(self, input_size=4, output_size = 1,x_col = 0,y_col = 1,z_col = 2, mode = 'custom', k = [4,4,4,4], c=3, device = 'cuda' ):
        ####
        # INPUTS:
        # input_size    : INTEGER - dimension of input tensor.DEFAUlT -4
        # output_size   : INTEGER - dimension of output tensor. DEFAUlT -1
        # x             : INTEGER - column index in input tensor for x-coordinate of DOM position. DEFAULT - 0
        # y             : INTEGER - column index in input tensor for y-coordinate of DOM position. DEFAULT - 1
        # z             : INTEGER - column index in input tensor for z-coordinate of DOM position. DEFAULT - 2
        # k             : INTEGER - number of neighbours. DEFAULT - 4
        # device        : STRING  - the device ID on which the model is run. DEFAULT - 'cuda'
        # c             : INTEGER - the dimension factor. DEFAULT - 3
        # target        : STRING  - specifies which version of dynedge to run. ['energy', 'angle', 'classifcation']
        #                  target = energy         : Regresses energy_log10. Use in conjuction with 'logcosh' loss function.
        #                  target = angle          : Regresses either zenith or azimuth. Use in conjunction with 'vonMisesSineCosineLoss
        #                  target = pid            : Use in conjuction with torch.loss.CrossEntropyLoss
        #                 
                                                                                   
        super(dynedgeSAG, self).__init__()
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
        
        self.sagpool=SAGPooling(l1,nonlinearity=torch.nn.LeakyReLU())
        
        self.nn1 = torch.nn.Linear(l3*4 + l1,l4)                                                  
        self.nn2   = torch.nn.Linear(l4,l5)
        self.nn3 =  torch.nn.Linear(4*l5,l6)
        self.nn4 = torch.nn.Linear(l6,l7)
        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
        self.Softmax=torch.nn.Softmax(dim=0)
                                                                                
    def forward(self, data):
        k = self.k        
        device = self.device
        mode   = self.mode
        pos_idx = self.pos_idx
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x,edge_index,edge_attr,batch,perm,score=self.sagpool(x=x,edge_index=edge_index,batch=batch)
        edge_index = knn_graph(x=x[:,pos_idx],k=k[0],batch=batch).to(device)
        
        
        a = self.conv_add(x,edge_index)


        edge_index = knn_graph(x=a[:,pos_idx],k=k[1],batch=batch).to(device)
        
        b = self.conv_add2(a,edge_index)

        edge_index = knn_graph(x=b[:,pos_idx],k=k[2],batch=batch).to(device)
        
        c = self.conv_add3(b,edge_index)

        edge_index = knn_graph(x=c[:,pos_idx],k=k[3],batch=batch).to(device)
        
        d = self.conv_add4(c,edge_index)
  
        x = torch.cat((x,a,b,c,d),dim = 1) 
        del a,b,c,d
        x = self.nn1(x)
        x = self.relu(x)
        x = self.nn2(x)
        

        x=self.pool4cat(x,batch)
        x = self.relu(x)
        x = self.nn3(x)
        
        x = self.relu(x)
        x = self.nn4(x)
        
        if mode == 'angle':
            x[:,0] = self.tanh(x[:,0])
            x[:,1] = self.tanh(x[:,1])
        

        return x
    def pool4cat(self,x,batch):
        a,_ = scatter_max(x, batch, dim = 0)
        b,_ = scatter_min(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        d = scatter_mean(x,batch,dim= 0)
        x = torch.cat((a,b,c,d),dim = 1)
        return x
    
class dynedgeTopk(torch.nn.Module):                                                     
    def __init__(self, input_size=4, output_size = 1,x_col = 0,y_col = 1,z_col = 2, mode = 'custom', k = [4,4,4,4], c=3, device = 'cuda' ):
        ####
        # INPUTS:
        # input_size    : INTEGER - dimension of input tensor.DEFAUlT -4
        # output_size   : INTEGER - dimension of output tensor. DEFAUlT -1
        # x             : INTEGER - column index in input tensor for x-coordinate of DOM position. DEFAULT - 0
        # y             : INTEGER - column index in input tensor for y-coordinate of DOM position. DEFAULT - 1
        # z             : INTEGER - column index in input tensor for z-coordinate of DOM position. DEFAULT - 2
        # k             : INTEGER - number of neighbours. DEFAULT - 4
        # device        : STRING  - the device ID on which the model is run. DEFAULT - 'cuda'
        # c             : INTEGER - the dimension factor. DEFAULT - 3
        # target        : STRING  - specifies which version of dynedge to run. ['energy', 'angle', 'classifcation']
        #                  target = energy         : Regresses energy_log10. Use in conjuction with 'logcosh' loss function.
        #                  target = angle          : Regresses either zenith or azimuth. Use in conjunction with 'vonMisesSineCosineLoss
        #                  target = pid            : Use in conjuction with torch.loss.CrossEntropyLoss
        #                 
                                                                                   
        super(dynedgeTopk, self).__init__()
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
        
        self.topkpool=TopKPooling(l1,nonlinearity=torch.nn.LeakyReLU())
        
        self.nn1 = torch.nn.Linear(l3*4 + l1,l4)                                                  
        self.nn2   = torch.nn.Linear(l4,l5)
        self.nn3 =  torch.nn.Linear(4*l5,l6)
        self.nn4 = torch.nn.Linear(l6,l7)
        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
        self.Softmax=torch.nn.Softmax(dim=0)
                                                                                
    def forward(self, data):
        k = self.k        
        device = self.device
        mode   = self.mode
        pos_idx = self.pos_idx
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x,edge_index,edge_attr,batch,perm,score=self.topkpool(x=x,edge_index=edge_index,batch=batch)
        edge_index = knn_graph(x=x[:,pos_idx],k=k[0],batch=batch).to(device)
        
        
        a = self.conv_add(x,edge_index)


        edge_index = knn_graph(x=a[:,pos_idx],k=k[1],batch=batch).to(device)
        
        b = self.conv_add2(a,edge_index)

        edge_index = knn_graph(x=b[:,pos_idx],k=k[2],batch=batch).to(device)
        
        c = self.conv_add3(b,edge_index)

        edge_index = knn_graph(x=c[:,pos_idx],k=k[3],batch=batch).to(device)
        
        d = self.conv_add4(c,edge_index)
  
        x = torch.cat((x,a,b,c,d),dim = 1) 
        del a,b,c,d
        x = self.nn1(x)
        x = self.relu(x)
        x = self.nn2(x)
        

        x=self.pool4cat(x,batch)
        x = self.relu(x)
        x = self.nn3(x)
        
        x = self.relu(x)
        x = self.nn4(x)
        
        if mode == 'angle':
            x[:,0] = self.tanh(x[:,0])
            x[:,1] = self.tanh(x[:,1])
        

        return x
    def pool4cat(self,x,batch):
        a,_ = scatter_max(x, batch, dim = 0)
        b,_ = scatter_min(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        d = scatter_mean(x,batch,dim= 0)
        x = torch.cat((a,b,c,d),dim = 1)
        return x
class dynedgeglobvar(torch.nn.Module):                                                     
    def __init__(self, input_size=5, output_size = 1,x_col = 0,y_col = 1,z_col = 2, mode = 'custom', k = [4,4,4,4], c=3, device = 'cuda' ):
        ####
        # INPUTS:
        # input_size    : INTEGER - dimension of input tensor.DEFAUlT -4
        # output_size   : INTEGER - dimension of output tensor. DEFAUlT -1
        # x             : INTEGER - column index in input tensor for x-coordinate of DOM position. DEFAULT - 0
        # y             : INTEGER - column index in input tensor for y-coordinate of DOM position. DEFAULT - 1
        # z             : INTEGER - column index in input tensor for z-coordinate of DOM position. DEFAULT - 2
        # k             : INTEGER - number of neighbours. DEFAULT - 4
        # device        : STRING  - the device ID on which the model is run. DEFAULT - 'cuda'
        # c             : INTEGER - the dimension factor. DEFAULT - 3
        # target        : STRING  - specifies which version of dynedge to run. ['energy', 'angle', 'classifcation']
        #                  target = energy         : Regresses energy_log10. Use in conjuction with 'logcosh' loss function.
        #                  target = angle          : Regresses either zenith or azimuth. Use in conjunction with 'vonMisesSineCosineLoss
        #                  target = pid            : Use in conjuction with torch.loss.CrossEntropyLoss
        #                 
                                                                                   
        super(dynedgeglobvar, self).__init__()
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
        self.Softmax=torch.nn.Softmax(dim=0)
                                                                                
    def forward(self, data):
        k = self.k        
        device = self.device
        mode   = self.mode
        pos_idx = self.pos_idx
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x[:,-1]=torch.log10(x[:,-1])
        edge_index = knn_graph(x=x[:,pos_idx],k=k[0],batch=batch).to(device)
        # if(torch.sum(torch.isnan(edge_index)) != 0):
        #     raise TypeError("NAN at edgeindex 1")
        
        a = self.conv_add(x,edge_index)
        # if(torch.sum(torch.isnan(a)) != 0):
        #     print (a)
        #     raise TypeError("NAN at a1")
            
        edge_index = knn_graph(x=a[:,pos_idx],k=k[1],batch=batch).to(device)
        
        b = self.conv_add2(a,edge_index)
        # if(torch.sum(torch.isnan(b)) != 0):
        #     raise TypeError("NAN at b1")
        edge_index = knn_graph(x=b[:,pos_idx],k=k[2],batch=batch).to(device)
        
        c = self.conv_add3(b,edge_index)
        # if(torch.sum(torch.isnan(c)) != 0):
        #     raise TypeError("NAN at c1")
        edge_index = knn_graph(x=c[:,pos_idx],k=k[3],batch=batch).to(device)
        
        d = self.conv_add4(c,edge_index)
        # if(torch.sum(torch.isnan(d)) != 0):
        #     raise TypeError("NAN at d1")
        x = torch.cat((x,a,b,c,d),dim = 1) 
        del a,b,c,d
        x = self.nn1(x)
        # if(torch.sum(torch.isnan(x)) != 0):
        #     raise TypeError("NAN at nn1")
        x = self.relu(x)
        x = self.nn2(x)
        # if(torch.sum(torch.isnan(x)) != 0):
        #     raise TypeError("NAN at nn2")

        x=self.pool4cat(x,batch)
        x = self.relu(x)
        x = self.nn3(x)
        # if(torch.sum(torch.isnan(x)) != 0):
        #     raise TypeError("NAN at nn3")
        x = self.relu(x)
        x = self.nn4(x)
        # if(torch.sum(torch.isnan(x)) != 0):
        #     raise TypeError("NAN at nn4")
        if mode == 'angle':
            x[:,0] = self.tanh(x[:,0])
            x[:,1] = self.tanh(x[:,1])
        

        return x
    def pool4cat(self,x,batch):
        a,_ = scatter_max(x, batch, dim = 0)
        b,_ = scatter_min(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        d = scatter_mean(x,batch,dim= 0)
        x = torch.cat((a,b,c,d),dim = 1)
        return x
class dynedgefeatspace(torch.nn.Module):                                                     
    def __init__(self, input_size=4, output_size = 1,x_col = 0,y_col = 1,z_col = 2, mode = 'custom', k = [4,4,4,4], c=3, device = 'cuda' ):
        ####
        # INPUTS:
        # input_size    : INTEGER - dimension of input tensor.DEFAUlT -4
        # output_size   : INTEGER - dimension of output tensor. DEFAUlT -1
        # x             : INTEGER - column index in input tensor for x-coordinate of DOM position. DEFAULT - 0
        # y             : INTEGER - column index in input tensor for y-coordinate of DOM position. DEFAULT - 1
        # z             : INTEGER - column index in input tensor for z-coordinate of DOM position. DEFAULT - 2
        # k             : INTEGER - number of neighbours. DEFAULT - 4
        # device        : STRING  - the device ID on which the model is run. DEFAULT - 'cuda'
        # c             : INTEGER - the dimension factor. DEFAULT - 3
        # target        : STRING  - specifies which version of dynedge to run. ['energy', 'angle', 'classifcation']
        #                  target = energy         : Regresses energy_log10. Use in conjuction with 'logcosh' loss function.
        #                  target = angle          : Regresses either zenith or azimuth. Use in conjunction with 'vonMisesSineCosineLoss
        #                  target = pid            : Use in conjuction with torch.loss.CrossEntropyLoss
        #                 
                                                                                   
        super(dynedgefeatspace, self).__init__()
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
        self.Softmax=torch.nn.Softmax(dim=0)
                                                                                
    def forward(self, data):
        k = self.k        
        device = self.device
        mode   = self.mode
        pos_idx = self.pos_idx
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        edge_index = knn_graph(x=x[:,pos_idx],k=k[0],batch=batch).to(device)
        
        
        a = self.conv_add(x,edge_index)


        edge_index = knn_graph(x=a[:,pos_idx],k=k[1],batch=batch).to(device)
        
        b = self.conv_add2(a,edge_index)

        edge_index = knn_graph(x=b[:,pos_idx],k=k[2],batch=batch).to(device)
        
        c = self.conv_add3(b,edge_index)

        edge_index = knn_graph(x=c[:,pos_idx],k=k[3],batch=batch).to(device)
        
        d = self.conv_add4(c,edge_index)
  
        x = torch.cat((x,a,b,c,d),dim = 1) 
        del a,b,c,d
        x = self.nn1(x)
        x = self.relu(x)
        x = self.nn2(x)
        

        x=self.pool4cat(x,batch)
        x = self.relu(x)
        x = self.nn3(x)
        
        x = self.relu(x)
        x = self.nn4(x)
        
        if mode == 'angle':
            x[:,0] = self.tanh(x[:,0])
            x[:,1] = self.tanh(x[:,1])
        

        return x
    def pool4cat(self,x,batch):
        a,_ = scatter_max(x, batch, dim = 0)
        b,_ = scatter_min(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        d = scatter_mean(x,batch,dim= 0)
        x = torch.cat((a,b,c,d),dim = 1)
        return x
class dynedgecontext(torch.nn.Module):                                                     
    def __init__(self, input_size=4, output_size = 1,x_col = 0,y_col = 1,z_col = 2, mode = 'custom', k = [4,4,4,4], c=3, device = 'cuda' ):
        ####
        # INPUTS:
        # input_size    : INTEGER - dimension of input tensor.DEFAUlT -4
        # output_size   : INTEGER - dimension of output tensor. DEFAUlT -1
        # x             : INTEGER - column index in input tensor for x-coordinate of DOM position. DEFAULT - 0
        # y             : INTEGER - column index in input tensor for y-coordinate of DOM position. DEFAULT - 1
        # z             : INTEGER - column index in input tensor for z-coordinate of DOM position. DEFAULT - 2
        # k             : INTEGER - number of neighbours. DEFAULT - 4
        # device        : STRING  - the device ID on which the model is run. DEFAULT - 'cuda'
        # c             : INTEGER - the dimension factor. DEFAULT - 3
        # target        : STRING  - specifies which version of dynedge to run. ['energy', 'angle', 'classifcation']
        #                  target = energy         : Regresses energy_log10. Use in conjuction with 'logcosh' loss function.
        #                  target = angle          : Regresses either zenith or azimuth. Use in conjunction with 'vonMisesSineCosineLoss
        #                  target = pid            : Use in conjuction with torch.loss.CrossEntropyLoss
        #                 
                                                                                   
        super(dynedgecontext, self).__init__()
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
        self.context=torch.nn.Parameter(torch.Tensor(5))

        # self.nn1 = torch.nn.Linear(l3 + l1,l4)      # for context nn
        self.nntemp=torch.nn.Linear(l3+l1,l3*4+l1)
        
        self.nn1 = torch.nn.Linear(l3*4 + l1,l4)                                                  
        self.nn2   = torch.nn.Linear(l4,l5)
        self.nn3 =  torch.nn.Linear(4*l5,l6)
        self.nn4 = torch.nn.Linear(l6,l7)
        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
        self.Softmax=torch.nn.Softmax(dim=0)
                                                                                
    def forward(self, data):
        k = self.k        
        device = self.device
        mode   = self.mode
        pos_idx = self.pos_idx
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        edge_index = knn_graph(x=x[:,pos_idx],k=k[0],batch=batch).to(device)
        
        
        a = self.conv_add(x,edge_index)


        edge_index = knn_graph(x=a[:,pos_idx],k=k[1],batch=batch).to(device)
        
        b = self.conv_add2(a,edge_index)

        edge_index = knn_graph(x=b[:,pos_idx],k=k[2],batch=batch).to(device)
        
        c = self.conv_add3(b,edge_index)

        edge_index = knn_graph(x=c[:,pos_idx],k=k[3],batch=batch).to(device)
        
        d = self.conv_add4(c,edge_index)
  
        context=self.tanh(self.context)
        # x=x*context[0]
        # convs=a*context[1]+b*context[2]+c*context[3]+d*context[4]
        # x=torch.cat((x,convs),dim=1)
        x = torch.cat((x*context[0],a*context[1],b*context[2],c*context[3],d*context[4]),dim = 1) 
        # x = torch.cat((x,a,b,c,d),dim = 1) 
        del a,b,c,d
        # x=self.nntemp(x)
        x = self.nn1(x)
        x = self.relu(x)
        x = self.nn2(x)
        

        x=self.pool4cat(x,batch)
        x = self.relu(x)
        x = self.nn3(x)
        
        x = self.relu(x)
        x = self.nn4(x)
        
        if mode == 'angle':
            x[:,0] = self.tanh(x[:,0])
            x[:,1] = self.tanh(x[:,1])
        

        return x
    def pool4cat(self,x,batch):
        a,_ = scatter_max(x, batch, dim = 0)
        b,_ = scatter_min(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        d = scatter_mean(x,batch,dim= 0)
        x = torch.cat((a,b,c,d),dim = 1)
        return x
class dynedgeGlobatt(torch.nn.Module):                                                     
    def __init__(self, input_size=4, output_size = 1,x_col = 0,y_col = 1,z_col = 2, mode = 'custom', k = [4,4,4,4], c=3, device = 'cuda' ):
        ####
        # INPUTS:
        # input_size    : INTEGER - dimension of input tensor.DEFAUlT -4
        # output_size   : INTEGER - dimension of output tensor. DEFAUlT -1
        # x             : INTEGER - column index in input tensor for x-coordinate of DOM position. DEFAULT - 0
        # y             : INTEGER - column index in input tensor for y-coordinate of DOM position. DEFAULT - 1
        # z             : INTEGER - column index in input tensor for z-coordinate of DOM position. DEFAULT - 2
        # k             : INTEGER - number of neighbours. DEFAULT - 4
        # device        : STRING  - the device ID on which the model is run. DEFAULT - 'cuda'
        # c             : INTEGER - the dimension factor. DEFAULT - 3
        # target        : STRING  - specifies which version of dynedge to run. ['energy', 'angle', 'classifcation']
        #                  target = energy         : Regresses energy_log10. Use in conjuction with 'logcosh' loss function.
        #                  target = angle          : Regresses either zenith or azimuth. Use in conjunction with 'vonMisesSineCosineLoss
        #                  target = pid            : Use in conjuction with torch.loss.CrossEntropyLoss
        #                 
                                                                                   
        super(dynedgeGlobatt, self).__init__()
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
        
        self.gatenn=torch.nn.Linear(l5,1)
        self.nnGat=torch.nn.Linear(l5,l7)
        self.globatt=GlobalAttention(self.gatenn,self.nnGat)
        
        
        self.nn1 = torch.nn.Linear(l3*4 + l1,l4)                                                  
        self.nn2   = torch.nn.Linear(l4,l5)
        self.nn3 =  torch.nn.Linear(4*l5,l6)
        self.nn4 = torch.nn.Linear(l6,l7)
        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
        self.Softmax=torch.nn.Softmax(dim=0)
                                                                                
    def forward(self, data):
        k = self.k        
        device = self.device
        mode   = self.mode
        pos_idx = self.pos_idx
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        edge_index = knn_graph(x=x[:,pos_idx],k=k[0],batch=batch).to(device)
        
        
        a = self.conv_add(x,edge_index)


        edge_index = knn_graph(x=a[:,pos_idx],k=k[1],batch=batch).to(device)
        
        b = self.conv_add2(a,edge_index)

        edge_index = knn_graph(x=b[:,pos_idx],k=k[2],batch=batch).to(device)
        
        c = self.conv_add3(b,edge_index)

        edge_index = knn_graph(x=c[:,pos_idx],k=k[3],batch=batch).to(device)
        
        d = self.conv_add4(c,edge_index)
  
        x = torch.cat((x,a,b,c,d),dim = 1) 
        del a,b,c,d
        x = self.nn1(x)
        x = self.relu(x)
        x = self.nn2(x)
        x=self.globatt(x,batch)
        x = self.relu(x)
        # x = self.nn4(x)
        
        if mode == 'angle':
            x[:,0] = self.tanh(x[:,0])
            x[:,1] = self.tanh(x[:,1])
        

        return x
    def pool4cat(self,x,batch):
        a,_ = scatter_max(x, batch, dim = 0)
        b,_ = scatter_min(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        d = scatter_mean(x,batch,dim= 0)
        x = torch.cat((a,b,c,d),dim = 1)
        return x
class dynedgeEdgepool(torch.nn.Module):                                                     
    def __init__(self, input_size=4, output_size = 1,x_col = 0,y_col = 1,z_col = 2, mode = 'custom', k = [4,4,4,4], c=3, device = 'cuda' ):
        ####
        # INPUTS:
        # input_size    : INTEGER - dimension of input tensor.DEFAUlT -4
        # output_size   : INTEGER - dimension of output tensor. DEFAUlT -1
        # x             : INTEGER - column index in input tensor for x-coordinate of DOM position. DEFAULT - 0
        # y             : INTEGER - column index in input tensor for y-coordinate of DOM position. DEFAULT - 1
        # z             : INTEGER - column index in input tensor for z-coordinate of DOM position. DEFAULT - 2
        # k             : INTEGER - number of neighbours. DEFAULT - 4
        # device        : STRING  - the device ID on which the model is run. DEFAULT - 'cuda'
        # c             : INTEGER - the dimension factor. DEFAULT - 3
        # target        : STRING  - specifies which version of dynedge to run. ['energy', 'angle', 'classifcation']
        #                  target = energy         : Regresses energy_log10. Use in conjuction with 'logcosh' loss function.
        #                  target = angle          : Regresses either zenith or azimuth. Use in conjunction with 'vonMisesSineCosineLoss
        #                  target = pid            : Use in conjuction with torch.loss.CrossEntropyLoss
        #                 
                                                                                   
        super(dynedgeEdgepool, self).__init__()
        self.k = k
        self.mode = mode
        self.device = device
        self.pos_idx = [x_col,y_col,z_col]
        
        if mode == 'angle':
            output_size = 3            # VonMisesSineCosineLoss requires three dimensionsional output
        if mode == 'energy':
            output_size = 1            # logcosh requires one-dimensional output
        if mode == 'pid':
            output_size = 2            # CrossEntropyLoss requires two-dimensional output
        l1, l2, l3, l4, l5,l6,l7 = input_size,c*16*2,c*32*2,c*42*2,c*32*2,c*16*2,output_size
        

        
        self.nn_conv1 = torch.nn.Sequential(torch.nn.Linear(l1*2,l2),torch.nn.LeakyReLU(),torch.nn.Linear(l2,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add = EdgeConv(self.nn_conv1,aggr = 'add')

        self.nn_conv2 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add2 = EdgeConv(self.nn_conv2,aggr = 'add')

        self.nn_conv3 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add3 = EdgeConv(self.nn_conv3,aggr = 'add')

        self.nn_conv4 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add4 = EdgeConv(self.nn_conv4,aggr = 'add')
        
        self.edgepool=EdgePooling(l1,add_to_edge_score=0.5,)
        
        self.nn1 = torch.nn.Linear(l3*4+ l1,l4)                                                  
        self.nn2   = torch.nn.Linear(l4,l5)
        self.nn3 =  torch.nn.Linear(4*l5,l6)
        self.nn4 = torch.nn.Linear(l6,l7)
        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
        self.Softmax=torch.nn.Softmax(dim=0)
                                                                                
    def forward(self, data):
        k = self.k        
        device = self.device
        mode   = self.mode
        pos_idx = self.pos_idx
        x, edge_index, batch = data.x, data.edge_index, data.batch
        randpos=torch.rand((x.size()[0]),2).to(device) # testing random neighbors
        
        edge_index = knn_graph(x=randpos,k=k[0],batch=batch).to(device)
        x,edge_index,batch,_=self.edgepool(x,edge_index,batch)
        edge_index = knn_graph(x=x[:,pos_idx],k=k[0],batch=batch).to(device)
        
        
        a = self.conv_add(x,edge_index)


        edge_index = knn_graph(x=a[:,pos_idx],k=k[1],batch=batch).to(device)
        
        b = self.conv_add2(a,edge_index)

        edge_index = knn_graph(x=b[:,pos_idx],k=k[2],batch=batch).to(device)
        
        c = self.conv_add3(b,edge_index)

        edge_index = knn_graph(x=c[:,pos_idx],k=k[3],batch=batch).to(device)
        
        d = self.conv_add4(c,edge_index)
  
        x = torch.cat((x,a,b,c,d),dim = 1) 
        del a,b,c,d
        x = self.nn1(x)
        x = self.relu(x)
        x = self.nn2(x)
        

        x=self.pool4cat(x,batch)
        x = self.relu(x)
        x = self.nn3(x)
        
        x = self.relu(x)
        x = self.nn4(x)
        
        if mode == 'angle':
            x[:,0] = self.tanh(x[:,0])
            x[:,1] = self.tanh(x[:,1])
        

        return x
    def pool4cat(self,x,batch):
        a,_ = scatter_max(x, batch, dim = 0)
        b,_ = scatter_min(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        d = scatter_mean(x,batch,dim= 0)
        x = torch.cat((a,b,c,d),dim = 1)
        return x
class dynedgegauss_edgepool(torch.nn.Module):                                                     
    def __init__(self, input_size=4, output_size = 1,x_col = 0,y_col = 1,z_col = 2, mode = 'custom', k = [4,4,4,4], c=3, device = 'cuda',eps=1e-15 ):
        ####
        # INPUTS:
        # input_size    : INTEGER - dimension of input tensor.DEFAUlT -4
        # output_size   : INTEGER - dimension of output tensor. DEFAUlT -1
        # x             : INTEGER - column index in input tensor for x-coordinate of DOM position. DEFAULT - 0
        # y             : INTEGER - column index in input tensor for y-coordinate of DOM position. DEFAULT - 1
        # z             : INTEGER - column index in input tensor for z-coordinate of DOM position. DEFAULT - 2
        # k             : INTEGER - number of neighbours. DEFAULT - 4
        # device        : STRING  - the device ID on which the model is run. DEFAULT - 'cuda'
        # c             : INTEGER - the dimension factor. DEFAULT - 3
        # target        : STRING  - specifies which version of dynedge to run. ['energy', 'angle', 'classifcation']
        #                  target = energy         : Regresses energy_log10. Use in conjuction with 'logcosh' loss function.
        #                  target = angle          : Regresses either zenith or azimuth. Use in conjunction with 'vonMisesSineCosineLoss
        #                  target = pid            : Use in conjuction with torch.loss.CrossEntropyLoss
        #                 
                                                                                   
        super(dynedgegauss_edgepool, self).__init__()
        self.k = k
        self.mode = mode
        self.device = device
        self.pos_idx = [x_col,y_col,z_col]
        self.eps=eps
        
        if mode == 'angle':
            output_size = 3            # VonMisesSineCosineLoss requires three dimensionsional output
        if mode == 'energy':
            output_size = 1            # logcosh requires one-dimensional output
        if mode == 'pid':
            output_size = 2            # CrossEntropyLoss requires two-dimensional output
        if mode=='gauss':
            output_size=2
        l1, l2, l3, l4, l5,l6,l7 = input_size,c*16*2,c*32*2,c*42*2,c*32*2,c*16*2,output_size
        
        self.edgepool=EdgePooling(l1,add_to_edge_score=0.5)
        
        self.nn_conv1 = torch.nn.Sequential(torch.nn.Linear(l1*2,l2),torch.nn.LeakyReLU(),torch.nn.Linear(l2,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add = EdgeConv(self.nn_conv1,aggr = 'add')

        self.nn_conv2 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add2 = EdgeConv(self.nn_conv2,aggr = 'add')

        self.nn_conv3 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add3 = EdgeConv(self.nn_conv3,aggr = 'add')

        self.nn_conv4 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add4 = EdgeConv(self.nn_conv4,aggr = 'add')
        

        self.nn1 = torch.nn.Linear(l3*3 + l1,l4)                                                  
        self.nn2   = torch.nn.Linear(l4,l5)
        self.nn3 =  torch.nn.Linear(4*l5,l6)
        self.nn4 = torch.nn.Linear(l6,l7)
        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
        self.Softmax=torch.nn.Softmax(dim=0)
        self.reluhard=torch.nn.ReLU()
    def forward(self, data):
        k = self.k        
        device = self.device
        mode   = self.mode
        pos_idx = self.pos_idx
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x,edge_index,batch,_=self.edgepool(x,edge_index,batch)
        edge_index = knn_graph(x=x[:,pos_idx],k=k[0],batch=batch).to(device)
        
        
        a = self.conv_add(x,edge_index)


        edge_index = knn_graph(x=a[:,pos_idx],k=k[1],batch=batch).to(device)
        
        b = self.conv_add2(a,edge_index)

        edge_index = knn_graph(x=b[:,pos_idx],k=k[2],batch=batch).to(device)
        
        c = self.conv_add3(b,edge_index)

        edge_index = knn_graph(x=c[:,pos_idx],k=k[3],batch=batch).to(device)
        
        d = self.conv_add4(c,edge_index)
  
        x = torch.cat((x,a,b,c),dim = 1) 
        del a,b,c
        x = self.nn1(x)
        x = self.relu(x)
        x = self.nn2(x)
        

        x=self.pool4cat(x,batch)
        x = self.relu(x)
        x = self.nn3(x)
        
        x = self.relu(x)
        x = self.nn4(x)
        # print (x.size())
        y=torch.unsqueeze(x[:,0],1)
        z=self.reluhard(torch.unsqueeze(x[:,1],1))+self.eps
        # print (x.size(),y.size(),z.size())
        x= torch.hstack((y,z))
        # print (x.size())
        del y,z
        if mode == 'angle':
            x[:,0] = self.tanh(x[:,0])
            x[:,1] = self.tanh(x[:,1])
        

        return x
    def pool4cat(self,x,batch):
        a,_ = scatter_max(x, batch, dim = 0)
        b,_ = scatter_min(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        d = scatter_mean(x,batch,dim= 0)
        x = torch.cat((a,b,c,d),dim = 1)
        return x    

class GRUconv(torch.nn.Module):  
    def __init__(self, input_size=4, output_size = 1,x_col = 0,y_col = 1,z_col = 2, mode = 'custom', k = 4, c=3,seqlength=1, device = 'cuda', ):
                ####
        # INPUTS:
        # input_size    : INTEGER - dimension of input tensor. DEFAULT - 4
        # output_size   : INTEGER - dimension of output tensor. DEFALT - 1
        # x             : INTEGER - column index in input tensor for x-coordinate of DOM position. DEFAULT - 0
        # y             : INTEGER - column index in input tensor for y-coordinate of DOM position. DEFAULT - 1
        # z             : INTEGER - column index in input tensor for z-coordinate of DOM position. DEFAULT - 2
        # k             : INTEGER - number of neighbours. DEFAULT - 4
        # device        : STRING  - the device ID on which the model is run. DEFAULT - 'cuda'
        # c             : INTEGER - the dimension factor. DEFAULT - 3
        #seqlength      : INTEGER - Length of sequence fed to decoder. DEFAULT -'5' , must be larger than number of features
        # target        : STRING  - specifies which version of dynedge to run. ['energy', 'angle', 'classifcation']
        #                  target = energy         : Regresses energy_log10. Use in conjuction with 'logcosh' loss function.
        #                  target = angle          : Regresses either zenith or azimuth. Use in conjunction with 'vonMisesSineCosineLoss
        #                  target = pid            : Use in conjuction with torch.loss.CrossEntropyLoss
        super(GRUconv, self).__init__()
        self.k = k
        self.mode = mode
        self.device = device
        self.pos_idx = [x_col,y_col,z_col]
        if mode == 'angle':
            output_size = 3            # VonMisesSineCosineLoss requires three dimensionsional output
        if mode == 'energy':
            output_size = 1            # logcosh requires one-dimensional output
        if mode == 'pid':
            output_size = 2 
        l1, l2, l3, l4, l5,l6,l7,l8 = input_size,c*5,c*5,c*5,c*5,c*5,c*5,output_size
        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
        
        self.GGconv1=GatedGraphConv(input_size, seqlength)
        
        self.nn1 = torch.nn.Linear(l1,l2)
        
        self.resblock1=torch.nn.Sequential(BatchNorm(l2),self.relu,torch.nn.Linear(l2,l3),BatchNorm(l3),self.relu,torch.nn.Linear(l3,l2),self.relu).to(device)
                                                                                                                                                       
        self.resblock2=torch.nn.Sequential(BatchNorm(l2),self.relu,torch.nn.Linear(l2,l3),BatchNorm(l3),self.relu,torch.nn.Linear(l3,l2),self.relu).to(device)
        
        
        self.nn2 =torch.nn.Linear(l2,l4)
        

        self.GGconv2=GatedGraphConv(l4, seqlength)
        
        self.resblock3=torch.nn.Sequential(BatchNorm(l4),self.relu,torch.nn.Linear(l4,l5),BatchNorm(l5),self.relu,torch.nn.Linear(l5,l4),self.relu).to(device)
        
        self.resblock4=torch.nn.Sequential(BatchNorm(l4),self.relu,torch.nn.Linear(l6,l7),BatchNorm(l7),self.relu,torch.nn.Linear(l7,l4),self.relu).to(device)
        
        self.nn3=torch.nn.Linear(l4,l8)
        self.nncat = torch.nn.Linear(4,l8)


           # CrossEntropyLoss requires two-dimensional output
    def forward(self,data):
        # device = self.device
        # mode   = self.mode
        k = self.k        
        device = self.device
        pos_idx = self.pos_idx
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index = knn_graph(x=x[:,pos_idx],k=k,batch=batch).to(device)
        x=self.GGconv1(x,edge_index)
        x=self.relu(x)

        x=self.nn1(x)
        x=self.relu(x)
        
        y=self.resblock1(x)
        x=x+y
        
        z=self.resblock2(x)
        x=x+z
        
        del y,z
        
        x=self.nn2(x)
        x=self.relu(x)
        
        x=self.GGconv2(x,edge_index)
        x=self.relu(x)
        
        p=self.resblock3(x)
        x=x+p
        
        o=self.resblock4(x)
        x=x+o
        del p,o
        
        x=self.nn3(x)
        x=self.relu(x)
        
        a,_ = scatter_max(x, batch, dim = 0)
        b,_ = scatter_min(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        d = scatter_mean(x,batch,dim= 0)
        x = torch.cat((a,b,c,d),dim = 1)
        # print ("cat size",x.size())
        del a,b,c,d

        x=self.nncat(x)
        x=self.relu(x)
        # if(torch.sum(torch.isnan(x)) != 0):
                # print('NAN ENCOUNTERED AT NN2')

        # print ("xsize %s batchsize %s a size %s b size %s y size %s end forward" %(x.size(),batch.size(),a.size(),b.size(),data.y[:,0].size()))
        return x
    
class simpedge(torch.nn.Module):                                                     
    def __init__(self, input_size=4, output_size = 1,x_col = 0,y_col = 1,z_col = 2, mode = 'custom', k = 4, c=3, device = 'cuda' ):
        super(simpedge, self).__init__()
        self.k = k
        self.mode = mode
        self.device = device
        self.pos_idx = [x_col,y_col,z_col]
        
        
        l1, l2, l3, l4, l5,l6,l7 = input_size,c*5,c*5,c*42*2,c*32*2,c*16*2,output_size
        
        if mode == 'angle':
            output_size = 3            # VonMisesSineCosineLoss requires three dimensionsional output
        if mode == 'energy':
            output_size = 1            # logcosh requires one-dimensional output
        if mode == 'pid':
            output_size = 2            # CrossEntropyLoss requires two-dimensional output
        
        self.nn_conv1 = torch.nn.Sequential(torch.nn.Linear(l1*2,l2),torch.nn.LeakyReLU(),torch.nn.Linear(l2,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add = EdgeConv(self.nn_conv1,aggr = 'add')

        self.nn1 = torch.nn.Linear(l3 ,l7)                                               
        self.nn2 = torch.nn.Linear(l7*4 ,l7)    
        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
                                          
                                                                                
    def forward(self, data):
        k = self.k        
        device = self.device
        mode   = self.mode
        pos_idx = self.pos_idx
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        edge_index = knn_graph(x=x[:,pos_idx],k=k,batch=batch).to(device)

        x = self.conv_add(x,edge_index)
        
        edge_index = knn_graph(x=x[:,pos_idx],k=k,batch=batch).to(device)
        "check if this recalculation of edge indices is correct, maybe you can do it over all of x"

        x = self.nn1(x)
        x = self.relu(x)

        
        a,_ = scatter_max(x, batch, dim = 0)
        b,_ = scatter_min(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        d = scatter_mean(x,batch,dim= 0)
        x = torch.cat((a,b,c,d),dim = 1)
        x = self.nn2(x)

        if mode == 'angle':
            x[:,0] = self.tanh(x[:,0])
            x[:,1] = self.tanh(x[:,1])
        

        return x