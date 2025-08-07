

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available() 


# Utility functions for tensor conversion and CUDA handling
def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v
def toTensor(v,dtype = torch.float,requires_grad = False):       
    return cuda(Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad))
def toNumpy(v):
    if USE_CUDA:
        return v.detach().cpu().numpy()
    return v.detach().numpy()


# Calculates the transductive loss using test sample prediction
def TransductiveLoss(z):

    closs = torch.exp(-3*(z**2)) 
    return closs

class TransductiveSurvivalRankerRejection:
    def __init__(self,model=None,lambda_w=0.1,lambda_u = 0.0,p=2,lr=1e-2,Tmax = 1000,dropout=0.5):

        self.lambda_w = lambda_w
        self.lambda_u = lambda_u
        self.p = p
        self.Tmax = Tmax
        self.lr = lr
        self.model = model
        self.dropout = dropout


    def fit(self,X_train,T_train,E_train,X_test = None,plot_loss=False):        
       
        #Handle data as tensors
        x = toTensor(X_train)
        if X_test is not None:
            X_test = toTensor(X_test)
        y = toTensor(T_train)
        e = toTensor(E_train)
        
        
        #NN input and output size
        N,D_in = x.shape        
        H, D_out = D_in, 1
        
        #Create a new model if none exist using a linear nn with tanh activation
        if self.model is None:                    
            self.model = torch.nn.Sequential( 
                torch.nn.Linear(H, D_out,bias=True),
                torch.nn.Tanh(),
                nn.Dropout(p=self.dropout)
            )
        model = self.model
        model=cuda(model)
        learning_rate = self.lr
        
        #initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.0)
        epochs = self.Tmax  
        lambda_w = self.lambda_w 
        p = self.p
        L = [] #collects losses for all epochs
        
        #Create pairs of samples where E_j=1 AND T_i > T_j
        dT = T_train[:, None] - T_train[None, :] 
        dP = (dT>0)*E_train
        dP = toTensor(dP,requires_grad=False)>0

        self.bias = 0.0
        loss_uv = 0.0
        transductiveLosses=[]
        if plot_loss:
            fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))

        for t in (range(epochs)):
            
            #Get predictions on the training set and calculate Ranking Loss and add it to the overall loss
            y_pred = model(x).flatten()
            dZ = (y_pred.unsqueeze(1) - y_pred)[dP]  
            loss = torch.mean(torch.max(toTensor([0],requires_grad=False),1.0-dZ))
            
            #Get predictions on the test set and calculate Transductive Loss and add it to the overall loss
            if X_test is not None and self.lambda_u > 0:
                test_predictions = model(X_test).flatten()
                loss_u = torch.mean(TransductiveLoss(test_predictions))
                transductive_loss=self.lambda_u*loss_u
                loss+=transductive_loss
                loss_uv = loss_u.item()
                transductiveLosses.append(loss_uv)

            w = model[0].weight.view(-1) #only input layer weights (exclude bias from regularization)
            
            #Calculate the regularization term and add it to the overall loss
            regularization_term=lambda_w*torch.norm(w, p)**p 
            loss+=regularization_term
            
            L.append(loss.item())

            # Calculate the gradient during the backward pass and perform a single optimization step (update weights w)
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 
            
            if plot_loss:
                ax1.clear()
                ax1.plot(L, label="Train Loss")
                ax1.set_xlabel('Epochs')
                ax1.set_ylabel('Loss')
                ax1.set_title(f'Loss (Epoch {t}/{epochs})')
                ax1.legend()          
                plt.pause(0.01)
            
        w = model[0].weight.view(-1)
        self.w = w
        self.L = L
        self.model = model

        return self
    
    
    def decision_function(self,x):
        """
        Returns the risk prediction for the given input x.
        """
        x = toTensor(x)
        self.model.eval()
        with torch.no_grad():
            preds=toNumpy(self.model(x)-self.bias).flatten()
        return preds
    
    
    def getW(self):
        """
        Returns the normalized weights of the model.
        Normalization is done using L1 norm.
        """
        
        return toNumpy(self.w/torch.linalg.norm(self.w,ord=1))

