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
from sklearn.metrics import confusion_matrix, classification_report
import logging
import seaborn as sns

random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)

os.makedirs("logs/", exist_ok=True)

logger1 = logging.getLogger('logger1')
logger1.setLevel(logging.INFO)

file_handler1 = logging.FileHandler('logs/test_lnn_new.log')
file_handler1.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(message)s')
file_handler1.setFormatter(formatter)

# Add the handler to the logger
logger1.addHandler(file_handler1)


logger2 = logging.getLogger('logger2')
logger2.setLevel(logging.INFO)

# Create a file handler and set the level to INFO
file_handler2 = logging.FileHandler('logs/test_dense_anomaly.log')
file_handler2.setLevel(logging.DEBUG)

# Create a formatter and add it to the handler
file_handler2.setFormatter(formatter)

# Add the handler to the logger
logger2.addHandler(file_handler2)
# sample video

test_dataset = data_loader(is_train=0)
test_loader = DataLoader(test_dataset, batch_size=1,
                                shuffle=True)

# Robbery_test_dataset = data_loader(is_train=0,Aclass="Robbery")


# Stealing_test_dataset = data_loader(is_train=0,Aclass="Stealing")


# normal_test_loader = DataLoader(normal_test_dataset, batch_size=1,
#                                 shuffle=True)

# Robbery_test_loader = DataLoader(
#     Robbery_test_dataset, batch_size=1, shuffle=True)

# Stealing_test_loader = DataLoader(
#     Stealing_test_dataset, batch_size=1, shuffle=True)

device = 'cpu'
model_lnn = LNN(input_dim=2048, drop_p=0.6).to(device)
# model_dense = dense(input_dim=2048, drop_p=0.6).to(device)
# optimizer = torch.optim.Adagrad(model.parameters(), lr= 0.0015, weight_decay=0.0012)
criterion = nn.CrossEntropyLoss()
auc_values=[]
roc_auc_values= []
tpr_all=[]
fpr_all=[]
st=[]
pred=[]
actual=[]
with torch.no_grad():
    model_lnn.load_state_dict(torch.load("/home/kani/Documents/project_phase2/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch/code/checkpoint/lnn_model100.pth"))
    model_lnn.eval()
    for i, data in enumerate(test_loader):
        name,inputs,actual_label=data
        batch_size = inputs.shape[0]
        # inputs = inputs.view(-1, inputs.size(-1)).to(device)
        output=model_lnn(inputs)
        label_pred = torch.argmax(output).cpu().detach().numpy()
        pred.append(label_pred)
        actual.append(actual_label)
        
        # fpr, tpr, thresholds = metrics.roc_curve(
        #     actual,pred, pos_label=1)
        # roc_auc_value = metrics.auc(fpr, tpr)
        auc_value=metrics.accuracy_score(actual,pred)
        # Plot ROC curve
        # print(type(fpr))
        # fpr_all.append(fpr.tolist())
        # tpr_all.append(tpr.tolist())
        print(actual)
        print(pred)
        # print(auc_value)


        # logger1.info(
            # f"{name_n , name_a } | accuracy value is {auc_value} ")
        auc_values.append(auc_value)  
        # roc_auc_values.append(roc_auc_value)  
        
logger1.info(
    f"The normal videos  | accuracy value is {sum(auc_values)/len(auc_values)}")
# logger2.info(f"The anomaly videos  | accuracy value is {sum(auc_values_a)/len(auc_values_a)} | roc_auc value is {sum(roc_auc_a)/len(roc_auc_a)}")


# print(
#     # f"the accuracy value is {(sum(auc_values_n)/len(auc_values_n)+sum(auc_values_a)/len(auc_values_a))/2} ")
# # print(tpr_all)
# print(len(fpr_all),len(fpr_all[0]),len(fpr_all[2]))
# fpr_m=np.mean(fpr_all,axis=0)
# tpr_m=np.mean(tpr_all,axis=0)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr_m, tpr_m, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(sum(roc_auc_values)/len(roc_auc_values)))
# plt.plot([0, 1], [0, 1], color='navy', lw=2, label='Random')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.savefig(f"plots/final_roc_curve_dense.png")
# confusion=confusion_matrix(actual,st)
# acc=metrics.accuracy_score(actual,st)

# print(acc)
# print(st)
# cm_display = metrics.ConfusionMatrixDisplay(
#             confusion_matrix=confusion, display_labels=["Positive", "Negative"])

# cm_display.plot()
# plt.savefig("/home/kani/Documents/laptop-files/Project - liquid neural network - Copy/try/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch/plots/confusion_matrix_lnn_video_wise")

acc=metrics.accuracy_score(actual,pred)
precision=metrics.precision_score(actual,pred,average="macro")
recall=metrics.recall_score(actual,pred,average="macro")
f1_score=metrics.recall_score(actual,pred,average="macro")

print(f"precision is {precision}")
print(f"accuracy is {acc}")
print(f"recall is {recall}")
print(f"f1 score is {f1_score}")

# Compute confusion matrix
cm = confusion_matrix(actual ,pred)

# Define class labels
classes = ['Normal', 'Robbery', 'Stealing']

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig("/home/kani/Documents/project_phase2/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch/plots/multi_Class_confusion_matrix")
