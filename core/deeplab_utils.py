import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_fn=None):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = norm_fn(planes)
        self.relu = nn.ReLU(inplace=True)

        self.initialize([self.atrous_conv, self.bn])

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def initialize(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, output_stride, norm_fn, inplanes=2048):
        super().__init__()

        inplanes = inplanes

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        
        self.aspp1 = ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], norm_fn=norm_fn)
        self.aspp2 = ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], norm_fn=norm_fn)
        self.aspp3 = ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], norm_fn=norm_fn)
        self.aspp4 = ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], norm_fn=norm_fn)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
            norm_fn(256),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = norm_fn(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        
        self.initialize([self.conv1, self.bn1] + list(self.global_avg_pool.modules()))
    
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x

    def initialize(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder(nn.Module):
    def __init__(self, num_classes, low_level_inplanes, norm_fn, kernel_size=3, padding=1):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = norm_fn(48)
        self.relu = nn.ReLU(inplace=True)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=False),
            norm_fn(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=False),
            norm_fn(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

        self.initialize([self.conv1, self.bn1] + list(self.classifier.modules()))

    def forward(self, x, x_low_level):
        x_low_level = self.conv1(x_low_level)
        x_low_level = self.bn1(x_low_level)
        x_low_level = self.relu(x_low_level)

        x = F.interpolate(x, size=x_low_level.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, x_low_level), dim=1)
        x = self.classifier(x)

        return x

    def initialize(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder_Attention(nn.Module):
    def __init__(self, num_classes, low_level_inplanes, norm_fn, kernel_size=3, padding=1, attention_mode='CBAM'):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = norm_fn(48)
        self.relu = nn.ReLU(inplace=True)
        
        self.attention_mode = attention_mode

        if attention_mode == 'CBAM':
            self.attention = CBAM(304)
            self.attention1 = CBAM(256)
        elif attention_mode == 'SA':
            self.attention = SpatialAttention()
            self.attention1 = SpatialAttention()
        else:
            self.attention = ChannelAttention(304)
            self.attention1 = ChannelAttention(256)

        self.conv2 = nn.Conv2d(304, 256, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=False)
        self.bn2 = norm_fn(256)
        self.dropout2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=False)
        self.bn3 = norm_fn(256)
        self.dropout3 = nn.Dropout(0.1)
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

        self.initialize([self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3, self.conv4, self.attention, self.attention1])

    def forward(self, x, x_low_level):
        x_low_level = self.conv1(x_low_level)
        x_low_level = self.bn1(x_low_level)
        x_low_level = self.relu(x_low_level)

        x = F.interpolate(x, size=x_low_level.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, x_low_level), dim=1)
        x = self.attention(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.attention1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        return x

    def initialize(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x*self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return x*self.sigmoid(y)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        out = self.ca(x)
        result = self.sa(out)
        return result


