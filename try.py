import os 
path="/home/kani/Documents/project_phase2/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch/UCF-Crime/all_rgbs/Normal_Videos_event"


# for i in os.listdir(path)[80:100]:
#     print(i+"|0")


import torch 
import torch.nn.functional as F
# a=torch.randn(10,32,3)
# a=torch.max(a,dim=1)
# print(a.values)
# # # # Example predictions (logits) - shape: [32, 3]
# predictions = torch.randn(32, 3)
n_class=[0,1,2]
labels= torch.zeros(3, 3)
for i in range(len(n_class)):
    labels[i,n_class[i]]=1
print(labels)
# print(torch.mode(torch.argmax(predictions,dim=1)).values)
# # # Example target labels (class indices) - shape: [32]
# # target_labels = torch.randint(0, 3, (32,))

# # print(target_labels)
# # # Compute the cross-entropy loss
# # loss = F.cross_entropy(predictions, target_labels)

# # print("Cross-Entropy Loss:", loss.item())
# print(torch.ones(10)*2)