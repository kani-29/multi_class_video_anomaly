import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from learner import LNN, dense
from loss import *
import matplotlib.pyplot as plt
from dataset import *
from sklearn import metrics
import os 
import logging
import logging

# Configure logging
os.makedirs("logs/",exist_ok=True)
logging.basicConfig(filename='logs/train_loss_lnn.log', level=logging.DEBUG,
                    format='%(message)s')

random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)



train_dataset = data_loader(is_train=1)

train_loader = DataLoader(
     train_dataset, batch_size=16, shuffle=True)



device = 'cpu'
model_lnn = LNN(input_dim=2048, drop_p=0.6).to(device)
optimizer_1 = torch.optim.Adagrad(model_lnn.parameters(), lr= 0.01, weight_decay=0.0012)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_1, milestones=[20,40,60])
criterion=nn.CrossEntropyLoss()



def train_model(model, optimizer, epoch, lnn=True):
    print('\nEpoch: %d' % epoch)
    model_lnn.train()
    train_loss = 0
    loss = []
    for batch_idx, data in enumerate(train_loader):
        # inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        name,inputs,n_class=data
        batch_size = inputs.shape[0]
        # inputs = inputs.view(-1, inputs.size(-1)).to(device)
        output=model_lnn(inputs)
        
        # output4=model_lnn(roadaccident_inputs)
        print(output.shape)
        print(name,n_class)
        # print(output1.view(-1,output1.size(-2)).shape)
        labels= torch.zeros(batch_size, 3)
        for i in range(len(n_class)):
            labels[i,n_class[i]]=1


        labels.requires_grad=True
       

        
        loss = criterion(output.float(), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    loss = train_loss/len(train_loader)
    logging.info(
        f"epoch {epoch} with loss = {train_loss/len(train_loader)}")
    scheduler.step()
    os.makedirs("checkpoint",exist_ok=True)
    if lnn != True:
        file = "checkpoint/dense_model"+str(epoch)+".pth"
    else:
        file = "checkpoint/lnn_model"+str(epoch)+".pth"
    torch.save(model.state_dict(), file)
    

epoch=100
for i in range(epoch):
    train_model(model_lnn,optimizer_1,epoch=i+1)
    
