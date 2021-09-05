import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from Code.lib.res2net_v1b_base import Res2Net_model

def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



#Global Contextual module
class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



###############################################################################

class CIM0(nn.Module):    
    def __init__(self,in_dim, out_dim):
        super(CIM0, self).__init__()
        
        act_fn = nn.ReLU(inplace=True)
        

        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)   
        
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_21 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        

        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        

    def forward(self, rgb, depth):
        
        ################################
        
        x_rgb = self.layer_10(rgb)
        x_dep = self.layer_20(depth)
        
        rgb_w = nn.Sigmoid()(x_rgb)
        dep_w = nn.Sigmoid()(x_dep)
        
        ##
        x_rgb_w = rgb.mul(dep_w)
        x_dep_w = depth.mul(rgb_w)
        
        x_rgb_r = x_rgb_w + rgb
        x_dep_r = x_dep_w + depth
        
        ## fusion 
        x_rgb_r = self.layer_11(x_rgb_r)
        x_dep_r = self.layer_21(x_dep_r)
        
        
        ful_mul = torch.mul(x_rgb_r, x_dep_r)         
        x_in1   = torch.reshape(x_rgb_r,[x_rgb_r.shape[0],1,x_rgb_r.shape[1],x_rgb_r.shape[2],x_rgb_r.shape[3]])
        x_in2   = torch.reshape(x_dep_r,[x_dep_r.shape[0],1,x_dep_r.shape[1],x_dep_r.shape[2],x_dep_r.shape[3]])
        x_cat   = torch.cat((x_in1, x_in2),dim=1)
        ful_max = x_cat.max(dim=1)[0]
        ful_out = torch.cat((ful_mul,ful_max),dim=1)
        
        out1 = self.layer_ful1(ful_out)
         
        return out1


class CIM(nn.Module):    
    def __init__(self,in_dim, out_dim):
        super(CIM, self).__init__()
        
        act_fn = nn.ReLU(inplace=True)
        
        self.reduc_1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)
        self.reduc_2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)
        
        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)   
        
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_21 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        

        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        self.layer_ful2 = nn.Sequential(nn.Conv2d(out_dim+out_dim//2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)

    def forward(self, rgb, depth, xx):
        
        ################################
        x_rgb = self.reduc_1(rgb)
        x_dep = self.reduc_2(depth)
        
        x_rgb1 = self.layer_10(x_rgb)
        x_dep1 = self.layer_20(x_dep)
        
        rgb_w = nn.Sigmoid()(x_rgb1)
        dep_w = nn.Sigmoid()(x_dep1)
        
        ##
        x_rgb_w = x_rgb.mul(dep_w)
        x_dep_w = x_dep.mul(rgb_w)
        
        x_rgb_r = x_rgb_w + x_rgb
        x_dep_r = x_dep_w + x_dep
        
        ## fusion 
        x_rgb_r = self.layer_11(x_rgb_r)
        x_dep_r = self.layer_21(x_dep_r)
        
        
        ful_mul = torch.mul(x_rgb_r, x_dep_r)         
        x_in1   = torch.reshape(x_rgb_r,[x_rgb_r.shape[0],1,x_rgb_r.shape[1],x_rgb_r.shape[2],x_rgb_r.shape[3]])
        x_in2   = torch.reshape(x_dep_r,[x_dep_r.shape[0],1,x_dep_r.shape[1],x_dep_r.shape[2],x_dep_r.shape[3]])
        x_cat   = torch.cat((x_in1, x_in2),dim=1)
        ful_max = x_cat.max(dim=1)[0]
        ful_out = torch.cat((ful_mul,ful_max),dim=1)
        
        out1 = self.layer_ful1(ful_out)
        out2 = self.layer_ful2(torch.cat([out1,xx],dim=1))
         
        return out2



class MFA(nn.Module):    
    def __init__(self,in_dim):
        super(MFA, self).__init__()
         
        self.relu = nn.ReLU(inplace=True)
        
        self.layer_10 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)   
        self.layer_cat1 = nn.Sequential(nn.Conv2d(in_dim*2, in_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim),)        
        
    def forward(self, x_ful, x1, x2):
        
        ################################
    
        x_ful_1 = x_ful.mul(x1)
        x_ful_2 = x_ful.mul(x2)
        
     
        x_ful_w = self.layer_cat1(torch.cat([x_ful_1, x_ful_2],dim=1))
        out     = self.relu(x_ful + x_ful_w)
        
        return out
    
    

  
   
###############################################################################

class SPNet(nn.Module):
    def __init__(self, channel=32,ind=50):
        super(SPNet, self).__init__()
        
       
        self.relu = nn.ReLU(inplace=True)
        
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        
        #Backbone model
        #Backbone model
        self.layer_rgb  = Res2Net_model(ind)
        self.layer_dep  = Res2Net_model(ind)
        
        self.layer_dep0 = nn.Conv2d(1, 3, kernel_size=1)
        
        ###############################################
        # funsion encoders #
        ###############################################
        self.fu_0 = CIM0(64, 64)#
        
        self.fu_1 = CIM(256, 128) #MixedFusion_Block_IMfusion
        self.pool_fu_1 = maxpool()
        
        self.fu_2 = CIM(512, 256)
        self.pool_fu_2 = maxpool()
        
        self.fu_3 = CIM(1024, 512)
        self.pool_fu_3 = maxpool()

        self.fu_4 = CIM(2048, 1024)
        self.pool_fu_4 = maxpool()
        
        
        ###############################################
        # decoders #
        ###############################################
        
        ## rgb
        self.rgb_conv_4   = nn.Sequential(BasicConv2d(2048,    256, 3, padding=1),self.relu)
        self.rgb_gcm_4    = GCM(2048,  channel)
        
        self.rgb_conv_3   = nn.Sequential(BasicConv2d(1024+32, 256, 3, padding=1),self.relu)
        self.rgb_gcm_3    = GCM(1024+32,  channel)

        self.rgb_conv_2   = nn.Sequential(BasicConv2d(512+32, 128, 3, padding=1),self.relu)
        self.rgb_gcm_2    = GCM(512+32,  channel)

        self.rgb_conv_1   = nn.Sequential(BasicConv2d(256+32, 128, 3, padding=1),self.relu)
        self.rgb_gcm_1    = GCM(256+32,  channel)

        self.rgb_conv_0   = nn.Sequential(BasicConv2d(64+32, 64, 3, padding=1),self.relu)
        self.rgb_gcm_0    = GCM(64+32,  channel)        
        self.rgb_conv_out = nn.Conv2d(channel, 1, 1)
        
        ## depth
        self.dep_conv_4   = nn.Sequential(BasicConv2d(2048, 256, 3, padding=1),self.relu)
        self.dep_gcm_4    = GCM(2048,  channel)
        
        self.dep_conv_3   = nn.Sequential(BasicConv2d(1024+32, 256, 3, padding=1),self.relu)
        self.dep_gcm_3    = GCM(1024+32,  channel)

        self.dep_conv_2   = nn.Sequential(BasicConv2d(512+32, 128, 3, padding=1),self.relu)
        self.dep_gcm_2    = GCM(512+32,  channel)

        self.dep_conv_1   = nn.Sequential(BasicConv2d(256+32, 128, 3, padding=1),self.relu)
        self.dep_gcm_1    = GCM(256+32,  channel)

        self.dep_conv_0   = nn.Sequential(BasicConv2d(64+32, 64, 3, padding=1),self.relu)
        self.dep_gcm_0    = GCM(64+32,  channel)        
        self.dep_conv_out = nn.Conv2d(channel, 1, 1)
        
        ## fusion
        self.ful_conv_4   = nn.Sequential(BasicConv2d(2048, 256, 3, padding=1),self.relu)
        self.ful_gcm_4    = GCM(1024,  channel)
        
        self.ful_conv_3   = nn.Sequential(BasicConv2d(1024+32*3, 256, 3, padding=1),self.relu)
        self.ful_gcm_3    = GCM(512+32,  channel)

        self.ful_conv_2   = nn.Sequential(BasicConv2d(512+32*3, 128, 3, padding=1),self.relu)
        self.ful_gcm_2    = GCM(256+32,  channel)

        self.ful_conv_1   = nn.Sequential(BasicConv2d(256+32*3, 128, 3, padding=1),self.relu)
        self.ful_gcm_1    = GCM(128+32,  channel)

        self.ful_conv_0   = nn.Sequential(BasicConv2d(128+32*3, 64, 3, padding=1),self.relu)
        self.ful_gcm_0    = GCM(64+32,  channel)        
        self.ful_conv_out = nn.Conv2d(channel, 1, 1)
        
        self.ful_layer4   = MFA(channel)
        self.ful_layer3   = MFA(channel)
        self.ful_layer2   = MFA(channel)
        self.ful_layer1   = MFA(channel)
        self.ful_layer0   = MFA(channel)
        
                

    def forward(self, imgs, depths):
        
        img_0, img_1, img_2, img_3, img_4 = self.layer_rgb(imgs)
        dep_0, dep_1, dep_2, dep_3, dep_4 = self.layer_dep(self.layer_dep0(depths))
        
    
      
        ####################################################
        ## fusion
        ####################################################
        ful_0    = self.fu_0(img_0, dep_0)
        ful_1    = self.fu_1(img_1, dep_1, ful_0)
        ful_2    = self.fu_2(img_2, dep_2, self.pool_fu_1(ful_1))
        ful_3    = self.fu_3(img_3, dep_3, self.pool_fu_2(ful_2))
        ful_4    = self.fu_4(img_4, dep_4, self.pool_fu_3(ful_3))
        
        ####################################################
        ## decoder rgb
        ####################################################        
        #
        x_rgb_42    = self.rgb_gcm_4(img_4)
        
        x_rgb_3_cat = torch.cat([img_3, self.upsample_2(x_rgb_42)], dim=1)
        x_rgb_32    = self.rgb_gcm_3(x_rgb_3_cat)
        
        x_rgb_2_cat = torch.cat([img_2, self.upsample_2(x_rgb_32)], dim=1)
        x_rgb_22    = self.rgb_gcm_2(x_rgb_2_cat)        

        x_rgb_1_cat = torch.cat([img_1, self.upsample_2(x_rgb_22)], dim=1)
        x_rgb_12    = self.rgb_gcm_1(x_rgb_1_cat)     

        x_rgb_0_cat = torch.cat([img_0, x_rgb_12], dim=1)
        x_rgb_02    = self.rgb_gcm_0(x_rgb_0_cat)     
        rgb_out     = self.upsample_4(self.rgb_conv_out(x_rgb_02))
        
        
        ####################################################
        ## decoder depth
        ####################################################        
        #
        x_dep_42    = self.dep_gcm_4(dep_4)
        
        x_dep_3_cat = torch.cat([dep_3, self.upsample_2(x_dep_42)], dim=1)
        x_dep_32    = self.dep_gcm_3(x_dep_3_cat)
        
        x_dep_2_cat = torch.cat([dep_2, self.upsample_2(x_dep_32)], dim=1)
        x_dep_22    = self.dep_gcm_2(x_dep_2_cat)        

        x_dep_1_cat = torch.cat([dep_1, self.upsample_2(x_dep_22)], dim=1)
        x_dep_12    = self.dep_gcm_1(x_dep_1_cat)     

        x_dep_0_cat = torch.cat([dep_0, x_dep_12], dim=1)
        x_dep_02    = self.dep_gcm_0(x_dep_0_cat)     
        dep_out     = self.upsample_4(self.dep_conv_out(x_dep_02))
        

        ####################################################
        ## decoder fusion
        ####################################################        
        #
        x_ful_42    = self.ful_gcm_4(ful_4)
        
        x_ful_3_cat = torch.cat([ful_3, self.ful_layer3(self.upsample_2(x_ful_42),self.upsample_2(x_rgb_42),self.upsample_2(x_dep_42))], dim=1)
        x_ful_32    = self.ful_gcm_3(x_ful_3_cat)
        
        x_ful_2_cat = torch.cat([ful_2, self.ful_layer2(self.upsample_2(x_ful_32),self.upsample_2(x_rgb_32),self.upsample_2(x_dep_32))], dim=1)
        x_ful_22    = self.ful_gcm_2(x_ful_2_cat)        

        x_ful_1_cat = torch.cat([ful_1, self.ful_layer1(self.upsample_2(x_ful_22),self.upsample_2(x_rgb_22),self.upsample_2(x_dep_22))], dim=1)
        x_ful_12    = self.ful_gcm_1(x_ful_1_cat)     

        x_ful_0_cat = torch.cat([ful_0, self.ful_layer0(x_ful_12, x_rgb_12, x_dep_12)], dim=1)
        x_ful_02    = self.ful_gcm_0(x_ful_0_cat)     
        ful_out     = self.upsample_4(self.ful_conv_out(x_ful_02))


        return rgb_out, dep_out, ful_out
    
    

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
    
   

 