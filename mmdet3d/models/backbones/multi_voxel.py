import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=3):
        super(ChannelAttention, self).__init__()
        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 两层全连接网络 (共享的MLP)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.init_weights()  # 调用初始化权重的函数
    def init_weights(self):
        # 对 `fc` 中的所有卷积层进行 Xavier 初始化
        for m in self.fc:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)  # Xavier 正态分布初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 平均池化和最大池化的MLP输出
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        # 将两者相加并通过sigmoid激活
        out = avg_out + max_out
        return self.sigmoid(out)*x

# class SpatialAttention(nn.Module):
#     def __init__(self,kernel_size=7):
#         super().__init__()
#         self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=3)
#         self.sigmoid=nn.Sigmoid()

#     def forward(self, x) :
#         max_result,_=torch.max(x,dim=1,keepdim=True)
#         avg_result=torch.mean(x,dim=1,keepdim=True)
#         result=torch.cat([max_result,avg_result],1)
#         output=self.conv(result)
#         output=self.sigmoid(output)
#         return x*output


class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))
        
    def forward(self, x):
        return x * self.scale

class fusion(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # self.CA = ChannelAttention(in_features)

        # self.block_conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(in_features),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(in_features),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=7, stride=1, padding=3),
        #     nn.BatchNorm2d(in_features),
        #     nn.ReLU(),
        # )
        self.block_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(),
        )
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
            )
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
        self._sigmoid = nn.Sigmoid()
        self.scale_g1 = Scale(scale=1.0)
        self.init_weights() 
    def init_weights(self):
        # 遍历block_conv3的所有层
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)  # Xavier 正态分布初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x1, x2):
        
        x_add = x1 + x2
        # out = self.CA(x_add)
        g1 = x_add

        g1_cat = torch.cat(
            [self.conv_1_1(g1), 
            torch.max(g1, 1)[0].unsqueeze(1), 
            torch.mean(g1, 1).unsqueeze(1)], dim=1)
        g1_attentionmap = self.conv3(g1_cat)
        g1_out = (1 + self.scale_g1(self._sigmoid(g1_attentionmap))) * g1

        out = g1_out

        return out


class multi_voxel(nn.Module):
    def __init__(self,):
        super(multi_voxel, self).__init__()

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.conv_share_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv_share_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.convT1_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # self.fusion = fusion(256, 256)

        self.init_weights()  # 调用权重初始化函数
    def init_weights(self):
        # 对模型中的所有卷积层和反卷积层进行Xavier初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)  # 使用Xavier正态分布初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x1, x2):
        xb1_1 = self.conv1_1(x1)
        xb1_2 = self.conv1_2(xb1_1)

        height, width = x2.shape[2], x2.shape[3]
        x_expanded = F.interpolate(x2, size=(height + 72, width + 64), mode='bilinear', align_corners=False)
        xb2_1 = self.conv2_1(x_expanded)
        xb2_2 = self.conv2_2(xb2_1)


        # x_fusion1 = torch.cat([xb1_2, xb2_2, xb3_2], dim=1)
        # x_fusion1 = self.fusion(xb1_2, xb2_2)
        # x_fusion1 = torch.cat([xb1_2, xb2_2], dim=1)
        x_fusion1 = xb1_2 + xb2_2

        xshare_1 = self.conv_share_1(x_fusion1)
        xshare_2 = self.conv_share_2(xshare_1)

        xn1_1 = self.convT1_1(xshare_2)

        x_out = torch.cat([x_fusion1, xn1_1], dim=1)
        return x_out


if __name__ == '__main__':
    x = torch.randn(4, 128, 200, 176)
    x2 = torch.randn(4, 128, 144, 128)
    fusion_x = fusion(128, 128)
    out = fusion_x(x,x2)
    print('out.shape: ', out.shape)
    


