# full assembly of the sub-parts to form the complete net
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from .unet_parts import *
from .ResNet import *
import time
def weights_init(modules, type='xavier'):
    m = modules
    if isinstance(m, nn.Conv2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            m.weight.data.fill_(1.0)

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Module):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    m.weight.data.fill_(1.0)

                if m.bias is not None:
                    m.bias.data.zero_()


def convbn(in_planes, out_planes, kernel_size, stride):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(out_planes))

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1)

        self.ds = convbn(inplanes, planes, 3, stride)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        x = self.ds(x)
        out += x
        out = self.relu(out)
        return out

def PCA(data):
    # preprocess the data
    result = torch.empty(128)
    X_all = torch.squeeze(data,1)
    for i in range(data.size(0)):
        X = X_all[i,:,:] 
        X_mean = torch.mean(X,0)
        X = X - X_mean.expand_as(X)
        # svd
        U,S,V = torch.svd(torch.t(X))
        tempV = V[0].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        if i==0:
            result=tempV
        else:
            result = torch.cat((result,tempV),1)
    return result

# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, input_shape=None):
#         super(UNet, self).__init__()
#         self.inc = inconv(n_channels, 64)

#         self.down1 = down(64, 128)
#         self.down2 = down(128, 256)
#         self.down3 = down(256, 512)
#         self.down4 = down(512, 512)
#         self.up1 = up(1024, 256)
#         self.up2 = up(512, 128)
#         self.up3 = up(256, 64)
#         self.up4 = up(128, 32)
        
#         self.outc1 = outconv(32, n_classes)
#         self.outc2 = outconv(32, n_classes)
#         self.outc3 = outconv(32, n_classes)
#         #Res-Net18
#         self.clsnet = ResNet([2, 2, 2, 2], num_classes=3, model_path=None)
        
#         weights_init(self.modules(), 'xavier')

#     def forward(self, x):
#         x3_c = self.clsnet(x)
        
#         x1_0 = self.inc(x)
#         x2_0 = self.down1(x1_0)
#         x3_0 = self.down2(x2_0)
#         x4 = self.down3(x3_0)
#         x5 = self.down4(x4)

#         x = self.up1(x5, x4)
#         x = self.up2(x, x3_0)
#         x = self.up3(x, x2_0)
#         x = self.up4(x, x1_0)

#         x1 = self.outc1(x)
#         x1 = F.sigmoid(x1)       

#         x2 = self.outc2(x)
#         x2 = F.sigmoid(x2)    
             
#         x3 = self.outc3(x)
#         x3 = F.sigmoid(x3)
        
#         x3_c = F.sigmoid(x3_c)
#         x1_c = torch.unsqueeze(x3_c[:,0],1)
#         x2_c = torch.unsqueeze(x3_c[:,1],1)
#         x3_c2 = torch.unsqueeze(x3_c[:,2],1)
        
#         out = torch.cat((x1,x2,x3), 1)
  
#         return out,x1_c,x2_c,x3_c2

# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, input_shape=None):
#         super(UNet, self).__init__()
#         self.inc = inconv(n_channels, 64)

#         self.down1 = down(64, 128)
#         self.down2 = down(128, 256)
#         self.down3 = down(256, 512)
#         self.down4 = down(512, 512)
#         self.up1 = up(1024, 256)
#         self.up2 = up(512, 128)
#         self.up3 = up(256, 64)
#         self.up4 = up(128, 32)
        
#         self.outc1 = outconv(32, n_classes)
#         self.outc2 = outconv(32, n_classes)
#         self.outc3 = outconv(32, n_classes)

#         self.mp1 = nn.Sequential(
#                                  nn.Conv2d(64, 16, 3, stride=2, padding=1),
#                                  nn.BatchNorm2d(16),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(2),
#                                  nn.Conv2d(16, 1, 3, stride=2, padding=1),
#                                  nn.BatchNorm2d(1),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(2)
#                                 )
#         self.mp2 = nn.Sequential(
#                                  nn.Conv2d(64, 16, 3, stride=2, padding=1),
#                                  nn.BatchNorm2d(16),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(2),
#                                  nn.Conv2d(16, 1, 3, stride=2, padding=1),
#                                  nn.BatchNorm2d(1),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(2)
#                                 )
#         self.mp3 = nn.Sequential(
#                                  nn.Conv2d(64, 16, 3, stride=2, padding=1),
#                                  nn.BatchNorm2d(16),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(2),
#                                  nn.Conv2d(16, 1, 3, stride=2, padding=1),
#                                  nn.BatchNorm2d(1),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(2)
#                                 )

#         # compute conv feature size
#         # with torch.no_grad():
#         #     self.feature_size = self._forward_conv(
#         #         torch.zeros(*input_shape)).view(-1).shape[0]

#         self.fc1 = nn.Linear(512, n_classes)
#         self.fc2 = nn.Linear(512, n_classes)
#         self.fc3 = nn.Linear(512, n_classes)
        
#         weights_init(self.modules(), 'xavier')

#     def forward(self, x):
#         #end = time.time()
#         x1_0 = self.inc(x)
#         x2_0 = self.down1(x1_0)
#         x3_0 = self.down2(x2_0)
#         x4 = self.down3(x3_0)
#         x5 = self.down4(x4)

#         x = self.up1(x5, x4)
#         x = self.up2(x, x3_0)
#         x = self.up3(x, x2_0)
#         x = self.up4(x, x1_0)

#         x1 = self.outc1(x)
#         x1 = F.sigmoid(x1)       

#         x2 = self.outc2(x)
#         x2 = F.sigmoid(x2)    
             
#         x3 = self.outc3(x)
#         x3 = F.sigmoid(x3)
#         #temp_gpu = time.time() - end
#         x1_2 = x1_0
#         x1_ap = self.mp1(x1_2)
#         b,c,h,w = x1_ap.size()
#         x1_ap = x1_ap.view(b,-1)
#         x2_2 = x1_0
#         x2_ap = self.mp2(x2_2)
#         x2_ap = x2_ap.view(b,-1)
#         x3_2 = x1_0
#         x3_ap = self.mp3(x3_2)
#         x3_ap = x3_ap.view(b,-1)

#         x1_c = self.fc1(x1_ap)
#         x1_c = F.sigmoid(x1_c)
#         x2_c = self.fc2(x2_ap)
#         x2_c = F.sigmoid(x2_c)
#         x3_c = self.fc3(x3_ap)
#         x3_c = F.sigmoid(x3_c)

#         out = torch.cat((x1,x2,x3), 1)
        
#         return out,x1_c,x2_c,x3_c#,temp_gpu

# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, input_shape=None):
#         super(UNet, self).__init__()
#         self.inc = inconv(n_channels, 64)
#         self.mask_1 = ResBlock(64,64,1)
#         self.mask_2 = ResBlock(64,64,1)
#         self.mask_3 = ResBlock(64,64,1)
#         self.mask_4 = ResBlock(64,64,1)
#         self.predict_mask = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)

#         self.down1 = down(64, 128)
#         self.down2 = down(128, 256)
#         self.down3 = down(256, 512)
#         self.down4 = down(512, 512)
#         self.up1 = up(1024, 256)
#         self.up2 = up(512, 128)
#         self.up3 = up(256, 64)
#         self.up4 = up(128, 32)
        
#         self.outc1 = outconv(32, n_classes)
#         self.outc2 = outconv(32, n_classes)
#         self.outc3 = outconv(32, n_classes)

#         self.mp1 = nn.Sequential(
#                                  nn.Conv2d(64, 16, 3, stride=2, padding=1),
#                                  nn.BatchNorm2d(16),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(2),
#                                  nn.Conv2d(16, 1, 3, stride=2, padding=1),
#                                  nn.BatchNorm2d(1),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(2)
#                                 )
#         self.mp2 = nn.Sequential(
#                                  nn.Conv2d(64, 16, 3, stride=2, padding=1),
#                                  nn.BatchNorm2d(16),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(2),
#                                  nn.Conv2d(16, 1, 3, stride=2, padding=1),
#                                  nn.BatchNorm2d(1),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(2)
#                                 )
#         self.mp3 = nn.Sequential(
#                                  nn.Conv2d(64, 16, 3, stride=2, padding=1),
#                                  nn.BatchNorm2d(16),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(2),
#                                  nn.Conv2d(16, 1, 3, stride=2, padding=1),
#                                  nn.BatchNorm2d(1),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(2)
#                                 )

#         # compute conv feature size
#         # with torch.no_grad():
#         #     self.feature_size = self._forward_conv(
#         #         torch.zeros(*input_shape)).view(-1).shape[0]

#         self.fc1 = nn.Linear(512, n_classes)
#         self.fc2 = nn.Linear(512, n_classes)
#         self.fc3 = nn.Linear(512, n_classes)
        
#         weights_init(self.modules(), 'xavier')

#     def forward(self, x):
#         x1_0 = self.inc(x)
#         x2_0 = self.down1(x1_0)
#         x3_0 = self.down2(x2_0)
#         x4 = self.down3(x3_0)
#         x5 = self.down4(x4)

#         x = self.up1(x5, x4)
#         x = self.up2(x, x3_0)
#         x = self.up3(x, x2_0)
#         x = self.up4(x, x1_0)

#         x1 = self.outc1(x)
#         x1 = F.sigmoid(x1)       

#         x2 = self.outc2(x)
#         x2 = F.sigmoid(x2)    
             
#         x3 = self.outc3(x)
#         x3 = F.sigmoid(x3)
        
#         mask1 = self.mask_1(x1_0)
#         mask2 = self.mask_2(mask1)
#         mask3 = self.mask_3(mask2)
#         mask4 = self.mask_4(mask3)
#         mask_flow = self.predict_mask(mask4)

#         x1_2 = x1_0
#         x1_ap = self.mp1(x1_2)
#         b,c,h,w = x1_ap.size()
#         x1_ap = x1_ap.view(b,-1)
#         x2_2 = x1_0
#         x2_ap = self.mp2(x2_2)
#         x2_ap = x2_ap.view(b,-1)
#         x3_2 = x1_0
#         x3_ap = self.mp3(x3_2)
#         x3_ap = x3_ap.view(b,-1)

#         x1_c = self.fc1(x1_ap)
#         x1_c = F.sigmoid(x1_c)
#         x2_c = self.fc2(x2_ap)
#         x2_c = F.sigmoid(x2_c)
#         x3_c = self.fc3(x3_ap)
#         x3_c = F.sigmoid(x3_c)
#         out = torch.cat((x1,x2,x3), 1)

#         return out,mask_flow,x1_c,x2_c,x3_c

# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, input_shape=None):
#         super(UNet, self).__init__()
#         self.inc = inconv(n_channels, 64)

#         self.down1 = down(64, 128)
#         self.down2 = down(128, 256)
#         self.down3 = down(256, 512)
#         self.down4 = down(512, 512)
#         self.up1 = up(1024, 256)
#         self.up2 = up(512, 128)
#         self.up3 = up(256, 64)
#         self.up4 = up(128, 32)
        
#         self.outc1 = outconv(32, n_classes)
#         self.outc2 = outconv(32, n_classes)
#         self.outc3 = outconv(32, n_classes)

#         self.mp1 = nn.Sequential(
#                                  nn.Conv2d(65, 16, 3, stride=2, padding=1),
#                                  nn.BatchNorm2d(16),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(2),
#                                  nn.Conv2d(16, 1, 3, stride=2, padding=1),
#                                  nn.BatchNorm2d(1),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(2)
#                                 )
#         self.mp2 = nn.Sequential(
#                                  nn.Conv2d(65, 16, 3, stride=2, padding=1),
#                                  nn.BatchNorm2d(16),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(2),
#                                  nn.Conv2d(16, 1, 3, stride=2, padding=1),
#                                  nn.BatchNorm2d(1),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(2)
#                                 )
#         self.mp3 = nn.Sequential(
#                                  nn.Conv2d(65, 16, 3, stride=2, padding=1),
#                                  nn.BatchNorm2d(16),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(2),
#                                  nn.Conv2d(16, 1, 3, stride=2, padding=1),
#                                  nn.BatchNorm2d(1),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(2)
#                                 )

#         # compute conv feature size
#         # with torch.no_grad():
#         #     self.feature_size = self._forward_conv(
#         #         torch.zeros(*input_shape)).view(-1).shape[0]

#         self.fc1 = nn.Linear(512, n_classes)
#         self.fc2 = nn.Linear(512, n_classes)
#         self.fc3 = nn.Linear(512, n_classes)
        
#         weights_init(self.modules(), 'xavier')

#     def forward(self, x):
#         #end = time.time()
#         x1_0 = self.inc(x)
#         x2_0 = self.down1(x1_0)
#         x3_0 = self.down2(x2_0)
#         x4 = self.down3(x3_0)
#         x5 = self.down4(x4)

#         x = self.up1(x5, x4)
#         x = self.up2(x, x3_0)
#         x = self.up3(x, x2_0)
#         x = self.up4(x, x1_0)

#         x1 = self.outc1(x)
#         x1 = F.sigmoid(x1)       

#         x2 = self.outc2(x)
#         x2 = F.sigmoid(x2)    
             
#         x3 = self.outc3(x)
#         x3 = F.sigmoid(x3)
#         #temp_gpu = time.time() - end
        
#         x1_2 = torch.cat((x1_0,x1),1)
#         x1_ap = self.mp1(x1_2)
#         b,c,h,w = x1_ap.size()
#         x1_ap = x1_ap.view(b,-1)
#         x2_2 = torch.cat((x1_0,x2),1)
#         x2_ap = self.mp2(x2_2)
#         x2_ap = x2_ap.view(b,-1)
#         x3_2 = torch.cat((x1_0,x3),1)
#         x3_ap = self.mp3(x3_2)
#         x3_ap = x3_ap.view(b,-1)

#         x1_c = self.fc1(x1_ap)
#         x1_c = F.sigmoid(x1_c)
#         x2_c = self.fc2(x2_ap)
#         x2_c = F.sigmoid(x2_c)
#         x3_c = self.fc3(x3_ap)
#         x3_c = F.sigmoid(x3_c)

#         out = torch.cat((x1,x2,x3), 1)
  
#         return out,x1_c,x2_c,x3_c#,temp_gpu


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, input_shape=None):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.mask_1 = ResBlock(64,64,1)
        self.mask_2 = ResBlock(64,64,1)
        self.mask_3 = ResBlock(64,64,1)
        self.mask_4 = ResBlock(64,64,1)
        self.predict_mask = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 32)
        
        self.outc1 = outconv(32, n_classes)
        self.outc2 = outconv(32, n_classes)
        self.outc3 = outconv(32, n_classes)

        self.mp1 = nn.Sequential(
                                 nn.Conv2d(65, 16, 3, stride=2, padding=1),
                                 nn.BatchNorm2d(16),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2),
                                 nn.Conv2d(16, 1, 3, stride=2, padding=1),
                                 nn.BatchNorm2d(1),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2)
                                )
        self.mp2 = nn.Sequential(
                                 nn.Conv2d(65, 16, 3, stride=2, padding=1),
                                 nn.BatchNorm2d(16),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2),
                                 nn.Conv2d(16, 1, 3, stride=2, padding=1),
                                 nn.BatchNorm2d(1),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2)
                                )
        self.mp3 = nn.Sequential(
                                 nn.Conv2d(65, 16, 3, stride=2, padding=1),
                                 nn.BatchNorm2d(16),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2),
                                 nn.Conv2d(16, 1, 3, stride=2, padding=1),
                                 nn.BatchNorm2d(1),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2)
                                )

        # compute conv feature size
        # with torch.no_grad():
        #     self.feature_size = self._forward_conv(
        #         torch.zeros(*input_shape)).view(-1).shape[0]

        self.fc1 = nn.Linear(512, n_classes)
        self.fc2 = nn.Linear(512, n_classes)
        self.fc3 = nn.Linear(512, n_classes)
        
        weights_init(self.modules(), 'xavier')

    def forward(self, x):
        end = time.time()
        x1_0 = self.inc(x)
        x2_0 = self.down1(x1_0)
        x3_0 = self.down2(x2_0)
        x4 = self.down3(x3_0)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3_0)
        x = self.up3(x, x2_0)
        x = self.up4(x, x1_0)

        x1 = self.outc1(x)
        x1 = F.sigmoid(x1)       

        x2 = self.outc2(x)
        x2 = F.sigmoid(x2)    
             
        x3 = self.outc3(x)
        x3 = F.sigmoid(x3)
        
        mask1 = self.mask_1(x1_0)
        mask2 = self.mask_2(mask1)
        mask3 = self.mask_3(mask2)
        mask4 = self.mask_4(mask3)
        mask_flow = self.predict_mask(mask4)
        temp_gpu = time.time() - end

        x1_2 = torch.cat((x1_0,x1),1)
        x1_ap = self.mp1(x1_2)
        b,c,h,w = x1_ap.size()
        x1_ap = x1_ap.view(b,-1)
        x2_2 = torch.cat((x1_0,x2),1)
        x2_ap = self.mp2(x2_2)
        x2_ap = x2_ap.view(b,-1)
        x3_2 = torch.cat((x1_0,x3),1)
        x3_ap = self.mp3(x3_2)
        x3_ap = x3_ap.view(b,-1)

        x1_c = self.fc1(x1_ap)
        x1_c = F.sigmoid(x1_c)
        x2_c = self.fc2(x2_ap)
        x2_c = F.sigmoid(x2_c)
        x3_c = self.fc3(x3_ap)
        x3_c = F.sigmoid(x3_c)

        out = torch.cat((x1,x2,x3), 1)
        return out,mask_flow,x1_c,x2_c,x3_c,temp_gpu
      
# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes):
#         super(UNet, self).__init__()
#         self.inc = inconv(n_channels, 64)
#         self.down1 = down(64, 128)
#         self.down2 = down(128, 256)
#         self.down3 = down(256, 512)
#         self.down4 = down(512, 512)
#         self.up1 = up(1024, 256)
#         self.up2 = up(512, 128)
#         self.up3 = up(256, 64)
#         self.up4 = up(128, 64)
#         self.outc = outconv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         x = self.outc(x)
#         return F.sigmoid(x)