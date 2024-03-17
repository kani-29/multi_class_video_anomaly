import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch
import numpy
from torch.utils.data import DataLoader
from learner import LNN, dense
from loss import *
import matplotlib.pyplot as plt
from dataset import *
import os
from sklearn import metrics

path="/home/kani/Documents/project_phase2/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch/UCF-Crime/"
rgb_npy = np.load(os.path.join(path+'all_rgbs','Shoplifting/Shoplifting007_x264.mp4.npy'))
flow_npy = np.load(os.path.join(path+'all_flows', 'Shoplifting/Shoplifting007_x264.mp4.npy'))
concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
device = 'cpu'
model_lnn = LNN(input_dim=2048, drop_p=0.6).to(device)

score = model_lnn(torch.tensor(concat_npy))
print(score.shape)
# model_dense = dense(input_dim=2048, drop_p=0.6).to(device)
# optimizer = torch.optim.Adagrad(model.parameters(), lr= 0.0015, weight_decay=0.0012)
# criterion = MIL
# auc_values=[]
# roc_auc_values= []
# tpr_all=[]
# fpr_all=[]
# gt=[]
# st=[]
# with torch.no_grad():
#     model_lnn.load_state_dict(torch.load("/home/kani/Documents/laptop-files/Project - liquid neural network - Copy/try/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch/checkpoint/lnn_model75.pth"))
#     model_lnn.eval()
    
#     # name_a = f_name1[0].split("/")[1].split(".")[0]
#     # inputs = inputs.view(-1, inputs.size(-1)).to(device)
        
#     score = model_lnn(torch.tensor(concat_npy))
#     scores = score.cpu().detach().numpy()
#     # score_list = np.zeros(frames[0])
#     # step = np.round(np.linspace(0, frames[0]//16, 33))
#     # gt.append(1)

#     # for j in range(32):
#     #     score_list[int(step[j])*16:(int(step[j+1]))*16] = scores[j]
            
#     # gt_list = np.zeros(frames[0])
#     # for k in range(len(gts)//2):
#     #     s = gts[k*2]
#     #     e = min(gts[k*2+1], frames)
#     #     gt_list[s-1:e] = 1

#     # st.append(round(max(score_list)))


# num_segments=np.arange(1,33)
# # Plotting the graph
# plt.plot(num_segments, scores, marker='o', linestyle='-')
# plt.title('Anomaly Scores for Different Segments')
# plt.xlabel('Number of Segments')
# plt.ylabel('Anomaly Score')
# plt.grid()

# # Save the plot as a figure (in PNG format in this example)
# plt.savefig('anomaly_shoplifting.png')
