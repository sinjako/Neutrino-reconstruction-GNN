import torch





def VonMisesSineCosineLoss(output,truth):
        #####
        # OUTPUT: torch.tensor - output of model. Must have three columns. 
        # TRUTH : torch.tensor - truth values for angle
        # 
        # ABOUT THE LOSS FUNCTION:
        # This loss function optimizes the relative angle between vectors u and x where
        # u = [sin(truth), cos(truth), 1]
        # x = [output[:,0], output[:,1],1]
        # 
        # These vectors are treated as embedded vectors in a subspace of R^3 and treated with 
        # a vonMisesFisher distribution (directional statistics) by utilizing the negative log likelihood trick.
        # This way the model not only learns to predict sin(angle) and cos(angle) but also to produce a
        # reasonable error estimate k.
        # 
        # The actual prediction of the angle can be obtained by
        # 
        #  angle_prediction = torch.atan2(output[:,0],output[:,1])
        #  and the associated error is given as
        #  sigma = 1/(2*sqrt(abs(output[:,2])))              
 
        k            = torch.abs(output[:,2])       ### analogous to 1/sigma**2
            
        u_1 = (1/torch.sqrt(torch.tensor(2,dtype = torch.float)))*torch.sin(truth)
        u_2 = (1/torch.sqrt(torch.tensor(2,dtype = torch.float)))*torch.cos(truth)
        u_3 = (1/torch.sqrt(torch.tensor(2,dtype = torch.float)))*(1)
        
        norm_x  = torch.sqrt(1 + output[:,0]**2 + output[:,1]**2)
        
        x_1 = (1/norm_x)*output[:,0]                ## output[:,0] is sine(angle)
        x_2 = (1/norm_x)*output[:,1]                ## output[:,1] is cosine(angle)
        x_3 = (1/norm_x)*(1)                        ## adds numerical stability
        
        dotprod = u_1*x_1 + u_2*x_2 + u_3*x_3
        logc_3 = - torch.log(k) + k + torch.log(1 - torch.exp(-2*k))
        loss = torch.mean(-k*dotprod + logc_3)
        return loss
def gaussloss(probfunc,truth):
    log_likelihood  = -torch.square((probfunc[:,0] - truth))* probfunc[:,1] /2 + torch.log(probfunc[:,1]) / 2
    return torch.sum(-log_likelihood)
def logcosh(output, truth):
    #####
    # OUTPUT: torch.tensor - output of model. Must be one-dimensional. 
    # TRUTH : torch.tensor - truth. Must be one-dimensional.
    # 
    # ABOUT THE LOSS FUNCTION:
    # The usual log cosh of the numerical difference. 
    loss = torch.sum(torch.log(torch.cosh(((output-truth)))))
    return loss

