import torch
import torch.nn as nn
import torchvision.models as models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VGG19 layers we'll use for content and style
CONTENT_LAYERS = ['conv4_2']
STYLE_LAYERS   = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

class VGG19Features(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        vgg = vgg.to(DEVICE).eval()

        # Map layer names to indices in VGG19
        self.layer_map = {
            'conv1_1':  '0',  'conv1_2':  '2',
            'conv2_1':  '5',  'conv2_2':  '7',
            'conv3_1': '10',  'conv3_2': '12',
            'conv3_3': '14',  'conv3_4': '16',
            'conv4_1': '19',  'conv4_2': '21',
            'conv4_3': '23',  'conv4_4': '25',
            'conv5_1': '28',
        }

        # Only keep layers up to conv5_1
        self.model = nn.Sequential(*list(vgg.children())[:29])
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            for layer_name, idx in self.layer_map.items():
                if name == idx:
                    features[layer_name] = x
        return features


def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    gram = torch.mm(features, features.t())
    return gram / (c * h * w)


def content_loss(gen_features, content_features):
    loss = 0
    for layer in CONTENT_LAYERS:
        loss += torch.mean((gen_features[layer] - content_features[layer]) ** 2)
    return loss


def style_loss(gen_features, style_features):
    loss = 0
    for layer in STYLE_LAYERS:
        G = gram_matrix(gen_features[layer])
        S = gram_matrix(style_features[layer])
        loss += torch.mean((G - S) ** 2)
    return loss