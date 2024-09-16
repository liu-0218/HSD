import torch
import torch.nn as nn
import torchvision.models as models

class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)

class Encoder(nn.Module):
    def __init__(self, half_channel=False, in_channel=1,filters=[64, 128, 256, 512]):
        super(Encoder, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.bridge(x3)
        return x4,x1,x2,x3


class Decoder(nn.Module):
    def __init__(self,out_channel, half_channel=False,filters=[64, 128, 256, 512],
                 bilinear=True, decoder_freeze=False):
        super(Decoder, self).__init__()
        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

    def forward(self, x,x1,x2,x3):
        x4 = self.upsample_1(x)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        return x10

class HE_Encoder(nn.Module):
    def __init__(self, in_channel,
                 half_channel=False):
        super(HE_Encoder, self).__init__()
        self.H2E_encoder = Encoder(half_channel=half_channel,
                                   in_channel=in_channel)
        # self.E2H_encoder = Encoder(half_channel=half_channel,
        #                            in_channel=in_channel)

    def forward(self, h):
        h_out,x1,x2,x3 = self.H2E_encoder(h)
        # e_out= self.E2H_encoder(e)
        return h_out,x1,x2,x3


class HE_Decoder(nn.Module):
    def __init__(self, in_channel, half_channel=False,
                 bilinear=True, decoder_freeze=False):
        super(HE_Decoder, self).__init__()
        self.H2E_decoder = Decoder(out_channel=in_channel,
                                   half_channel=half_channel,
                                   bilinear=bilinear, decoder_freeze=decoder_freeze)
        # self.E2H_decoder = Decoder(out_channel=in_channel,
        #                            half_channel=half_channel,
        #                            bilinear=bilinear, decoder_freeze=decoder_freeze)

    def forward(self, h_out,x1,x2,x3):
        e_pred = self.H2E_decoder(h_out,x1,x2,x3)
        return e_pred

class UResnet_H(nn.Module):
    def __init__(self,half_channel = True,input_channel=1,
                 zero_init_residual=False, groups=1,filters=[64, 128, 256, 512],
                 width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(UResnet_H, self).__init__()
        self.online_he_encoder = HE_Encoder(in_channel=1)
        self.he_decoder = HE_Decoder(in_channel=1)
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

    def _forward_impl(self, x):
        x,x1,x2,x3 = self.online_he_encoder(x)
        x4 = self.he_decoder(x, x1, x2, x3)
        output = self.output_layer(x4)

        return output

    def forward(self, x):
        return self._forward_impl(x)


if __name__ == '__main__':
    UResnet50 = UResnet_H()
    print(UResnet50)
