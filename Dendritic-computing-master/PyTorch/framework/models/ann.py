# 开摩托的小猫
import torch
import torch.nn as nn

__all__ = ['ann']

class Dendrite(nn.Module):

    def __init__(self,
                 nonlinearity=0.1,
                 # nonlinearity=0.25,
                 ):
        super(Dendrite, self).__init__()
        self.a = torch.tensor(50 * nonlinearity / (1 - nonlinearity))
        # self.a = torch.tensor(6.18)

    def forward(self, x):
        xmax = x.abs().max(dim=1)[0]
        x = x / xmax.unsqueeze(1)
        if self.a <= 0:
            out= x
        else:
            out = (torch.exp(self.a * x) - 1) / (torch.exp(self.a) - 1)
        xmax = out.abs().max(dim=1)[0]
        out = out / xmax.unsqueeze(1)
        return out

class ddNet(nn.Module):

    def __init__(self,input_channels, num_iterations):
        super(ddNet, self).__init__()
        # xw+b
        self.dd = nn.Linear(input_channels,input_channels,bias=False)
        self.num_interations=num_iterations

    def forward(self, x):
        # x: [b, 1, 28, 28]
        c = x
        # h1 = x@w1*x
        for i in range(self.num_interations):
             x=self.dd(x)*c
        return x

class Soma(nn.Module):

    def __init__(self,
                 # threshold=0.1,
                 threshold=0.11,
                 ):
        super(Soma, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        # print(x.max())
        y = torch.clamp(2 * torch.sigmoid(x - self.threshold) - 1, min=0)
        return y

class Layer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channel,
                 num_iterations
                 ):
        super(Layer, self).__init__()
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.liners = nn.ModuleList([nn.Linear(in_channel,self.out_channel, bias=False) for in_channel in self.in_channels])
        self.soma = Soma()
        self.dendrite = Dendrite()
        self.dd_list = nn.ModuleList([ddNet(in_channel,num_iterations) for in_channel in self.in_channels])

    def forward(self, x_split):

        x0 = self.dendrite(self.liners[0]((self.dd_list[0](x_split[0]))))
        for i in range(1,len(x_split)):
            x0 += self.dendrite(self.liners[i]((self.dd_list[i](x_split[i]))))
        x0 = self.soma(x0)
        return x0

# 2fc   debug
# class ANN(nn.Module):
#
#     def __init__(self,
#                  args,
#                  **kwargs
#                  ):
#         super(ANN, self).__init__()
#         self.layer1 = nn.Linear(32*32*3,500)
#         self.layer2 = nn.Linear(500,10)
#
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, std=1)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         x_split = x.flatten(1)
#         x_split = self.layer1(x_split)
#         x = self.layer2(x_split)
#
#         return x

# 论文中结构
class ANN(nn.Module):

    def __init__(self,
                 args,
                 num_iterations,
                 **kwargs
                 ):
        super(ANN, self).__init__()
        self.layer1 = Layer([32*10]*3+[32*12]*3+[32*10]*3,500,num_iterations)
        self.layer2 = Layer([125]*4,10,num_iterations)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1.)

    def forward(self, x):
        # x = x * 255.
        # 将输入分为9份 分别是 x1(rgb) x2(rgb) x3(rgb)
        split_1_x = 10
        split_2_x = 22
        x_split = []

        x_split.append(x[:,0,:,:split_1_x].flatten(1))
        x_split.append(x[:,1,:,:split_1_x].flatten(1))
        x_split.append(x[:,2,:,:split_1_x].flatten(1))
        x_split.append(x[:,0,:,split_1_x:split_2_x].flatten(1))
        x_split.append(x[:,1,:,split_1_x:split_2_x].flatten(1))
        x_split.append(x[:,2,:,split_1_x:split_2_x].flatten(1))
        x_split.append(x[:,0,:,split_2_x:].flatten(1))
        x_split.append(x[:,1,:,split_2_x:].flatten(1))
        x_split.append(x[:,2,:,split_2_x:].flatten(1))

        x = self.layer1(x_split)

        # 将输入分为4份 分别是 x1 x2 x3 x4
        split = 500//4
        x_split = []
        for i in range(4):
            x_split.append(x[:,i*split:(i+1)*split])
        x = self.layer2(x_split)

        return x

def ann(args, **kwargs):
    return ANN(args, **kwargs)
