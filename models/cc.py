import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
import math


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B*W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim, out_dim, h=8, num_patch=1, scaled=True):
        super().__init__()
        self.d = in_dim // h
        self.num_patch = num_patch
        self.scaled = scaled
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=self.d, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=self.d, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

        # self.value_conv.weight.data.normal_(
        #     0, 0.05)  # to delete, default is better
        # self.value_conv.bias.data.zero_()

    def forward(self, x):
        x = torch.cat(x.chunk(self.num_patch, dim=2), 0)
        x = torch.cat(x.chunk(self.num_patch, dim=3), 0)

        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(
            m_batchsize*width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(
            m_batchsize*height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(
            m_batchsize*width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(
            m_batchsize*height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(
            0, 3, 1, 2).contiguous().view(m_batchsize*width, -1, height)
        proj_value_W = proj_value.permute(
            0, 2, 1, 3).contiguous().view(m_batchsize*height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height,
                    width).to(x.device)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(
            m_batchsize, height, width, width)

        if self.scaled:
            concate = self.softmax(
                torch.cat([energy_H, energy_W], 3)/math.sqrt(self.d))
        else:
            concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(
            0, 2, 1, 3).contiguous().view(m_batchsize*width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height +
                        width].contiguous().view(m_batchsize*height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(
            m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(
            m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())

        # try:
        #     output = self.gamma*(out_H + out_W) + x
        # except:
        #     output = self.gamma*(out_H + out_W) + proj_value

        # output = self.gamma*(out_H + out_W) + proj_value
        # output = self.gamma*(out_H + out_W)
        output = out_H + out_W

        output = torch.cat(output.chunk(self.num_patch, dim=0), 3)
        output = torch.cat(output.chunk(self.num_patch, dim=0), 2)

        return output


class CrissCrossMean(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim, out_dim):
        super(CrissCrossMean, self).__init__()
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.value_conv.weight.data.normal_(0, 0.05)
        self.value_conv.bias.data.zero_()

    def forward(self, x):
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.mean(dim=2, keepdim=True)
        proj_value_W = proj_value.mean(dim=3, keepdim=True)

        return self.gamma*(proj_value_H + proj_value_W) + proj_value

if __name__ == '__main__':
    model = CrissCrossAttention(64).cuda()
    x = torch.randn(2, 64, 5, 6).cuda()
    out = model(x)
    print(out.shape)
