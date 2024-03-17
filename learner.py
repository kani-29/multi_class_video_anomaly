import torch
import torch.nn as nn
from torch.nn import functional as F
from ncps.torch import LTC,CfC
from torchsummary import summary



class LNN(nn.Module):
    def __init__(self, input_dim=1024, drop_p=0.4):
        super(LNN, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            CfC(512, 32),
            # CfC(512,32),
            # nn.Linear(512,32),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(32, 3),
            #CfC(256,1),
            nn.Sigmoid()
        )
        self.ltc=CfC(512,32,return_sequences=True)
        self.drop_p = drop_p
        # self.weight_init()
        self.vars = nn.ParameterList()
        

        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param)

    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.ones_(layer.weight)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        #
        # print(vars)
        x = F.linear(x, vars[0], vars[1])
        x = F.gelu(x)
        x = F.dropout(x, self.drop_p, training=self.training)
        x = self.ltc(x)[0]
        # print(x.shape)
        # x = F.linear(x, vars[2], vars[3])
        x=F.gelu(x)
        # x=self.ltc(x)[0]
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[12],vars[13])
        x=torch.max(x,dim=1).values
        x=F.softmax(x,dim=1)
        
        # x=torch.argmax(x,dim=1)
        #x=self.ltc(x)[0]

        # x=self.classifier(x)
        
        # x = [torch.tensor(0) if i < 0.5 else torch.tensor(1)
        #            for i in x]
        # torch.tensor(x, dtype=float, requires_grad=True)
        return x
        # return x

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
    
    
    
    # def reset_model_parameters(model):
    #     # for param in model.parameters():
    #     # #     # if len(param.shape) > 1:  # Exclude biases
    #     # #     #     nn.init.xavier_normal_(param)
    #     # #     # else:
    #     # #     nn.init.zeros_(param)
    #     #     param.reset_parameters()
    #     # for layer in model.children():
    #         # if hasattr(layer, 'reset_parameters'):
    #     model.reset_parameters()


class dense(nn.Module):
    def __init__(self, input_dim=1000, drop_p=0.0):
        super(dense, self).__init__()
        
        self.classifier = nn.Sequential(
            # nn.Linear(input_dim, 512),
            # nn.GELU(),
            # nn.Dropout(0.4),
            # # LTC(512, wiring),
            # nn.Linear(512, 32),
            # # nn.Linear(512,32),
            # nn.GELU(),
            # nn.Dropout(0.4),
            # nn.Linear(32, 1),
            # #CfC(256,1),
            # nn.Sigmoid()
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.drop_p = 0.6
        self.weight_init()
        self.vars = nn.ParameterList()
        
        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param)

    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        x = F.linear(x, vars[0], vars[1])
        x = F.relu(x)
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[2], vars[3])
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[4], vars[5])
        return torch.sigmoid(x)

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars

# dense=dense(input_dim=2048, drop_p=0.6).cuda()
# # print(lnn.parameters())
# # print("---------------")
# # print(dense.parameters())
# # print(lnn)
# print(dense.parameters())

