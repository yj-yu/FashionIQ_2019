from model.dynamic_fc import DynamicFC
import torchvision.models as models
import torch.nn as nn
import torch
from torch.autograd import Variable


class FilmLayer(nn.Module):

    def __init__(self):
        super(FilmLayer, self).__init__()

        self.batch_size = None
        self.channels = None
        self.height = None
        self.width = None
        self.feature_size = None

        self.fc = DynamicFC().cuda()

    def forward(self, feature_maps, context):
        self.batch_size, self.channels, self.height, self.width = feature_maps.data.shape
        self.feature_size = feature_maps.data.shape[1]

        film_params = self.fc(context, out_planes=2 * self.feature_size, activation=None)
        film_params = torch.stack([film_params] * self.height, dim=2)
        film_params = torch.stack([film_params] * self.width, dim=3)

        gammas = film_params[:, :self.feature_size, :, :]
        betas = film_params[:, self.feature_size:, :, :]

        output = (1 + gammas) * feature_maps + betas

        return output


class FilmResBlock(nn.Module):

    def __init__(self, in_channels, feature_size, spatial_location=True):
        super(FilmResBlock, self).__init__()

        self.spatial_location = spatial_location
        self.feature_size = feature_size
        self.in_channels = in_channels
        # add 2 channels for spatial location
        if spatial_location:
            self.in_channels += 2

        # modulated resnet block with FiLM layer
        self.conv1 = nn.Conv2d(self.in_channels, self.feature_size, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.feature_size, self.feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.feature_size)
        self.film_layer = FilmLayer().cuda()
        self.relu2 = nn.ReLU()

    def forward(self, feature_maps, context):
        if self.spatial_location:
            feature_maps = append_spatial_location(feature_maps)

        conv1 = self.conv1(feature_maps)
        out1 = self.relu1(conv1)

        conv2 = self.conv2(out1)
        bn = self.bn2(conv2)

        film_out = self.film_layer(bn, context)
        out2 = self.relu2(film_out)

        out = out1 + out2

        return out


def append_spatial_location(feature_maps, min_val=-1, max_val=1):
    batch_size, channels, height, width = feature_maps.data.shape
    h_array = Variable(torch.stack([torch.linspace(min_val, max_val, height)]*width, dim=1).cuda())
    w_array = Variable(torch.stack([torch.linspace(min_val, max_val, width)]*height, dim=0).cuda())
    spatial_array = torch.stack([h_array, w_array], dim=0)
    spatial_array = torch.stack([spatial_array]*batch_size, dim=0)
    feature_maps = torch.cat([feature_maps, spatial_array], dim=1)

    return feature_maps