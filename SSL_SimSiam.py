import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")


        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class CustomResNet(nn.Module):

    def __init__(self, block, layers, groups=1,width_per_group=64,
                 replace_stride_with_dilation=None,norm_layer=None):
        super(CustomResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        self.layer0 = nn.Sequential(
            nn.Conv2d(1, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(self.inplanes),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

class projection_MLP(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=512, out_dim=512):
        super(projection_MLP, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128, out_dim=512):
        super(prediction_MLP,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim,out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class SSL(nn.Module):
    def __init__(self):
        super(SSL, self).__init__()

        self.distance=[]
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.backbone = CustomResNet(BasicBlock, [2, 2, 2, 2])
        self.projector = projection_MLP()

        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()

    def set_input(self, input):
        self.batch_view_1 = input[0]
        self.batch_view_2 = input[1]
        self.batch_view_1 = self.batch_view_1.to(self.device)
        self.batch_view_2 = self.batch_view_2.to(self.device)

        return self.batch_view_1, self.batch_view_2


    def loss_fn(self, p, z):
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()


    def forward(self,x1,x2):
        f, h = self.encoder, self.predictor

        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)

        L = self.loss_fn(p1, z2) / 2 + self.loss_fn(p2, z1) / 2

        return {'loss':L}


class CDNet(nn.Module):
    def __init__(self):
        super(CDNet, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.backbone = CustomResNet(BasicBlock, [2, 2, 2, 2])
        self.projector = projection_MLP()

        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Linear(128, 2)
        )

    def set_input(self, input):
        self.batch_view_1 = input[0]
        self.batch_view_2 = input[1]
        self.batch_label = input[2]

        self.batch_view_1 = self.batch_view_1.to(self.device)
        self.batch_view_2 = self.batch_view_2.to(self.device)
        self.batch_label = self.batch_label.to(self.device)

        return self.batch_view_1, self.batch_view_2, self.batch_label

    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

