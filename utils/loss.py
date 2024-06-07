import torch
import torch.nn as nn
from torchvision import models

import numpy as np

# Define a function to extract feature maps from VGG
class FeatureExtractor(nn.Module):
    def __init__(self, vgg, layer_indices):
        super(FeatureExtractor, self).__init__()
        self.vgg = vgg
        self.layer_indices = layer_indices

    def forward(self, x):
        features = []
        vgg = self.vgg.module if isinstance(self.vgg, nn.DataParallel) else self.vgg  # Access the underlying VGG module
        for i, layer in enumerate(vgg):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
        return features
    
class PerCeptualLoss():
    def __init__(self, device, loss, content_layers=[], style_layers=[]):
        super().__init__()
        self.vgg = nn.DataParallel(models.vgg16(pretrained=True).features).to(device)
        self.vgg.to(device).eval()
        #self.vgg = nn.DataParallel(self.vgg).to(device)

        self.content_layers = content_layers
        self.style_layers = style_layers
        
        self.loss = loss
        self.content_extractor = FeatureExtractor(self.vgg, content_layers)
        self.style_extractor = FeatureExtractor(self.vgg, style_layers)

    def content_loss(self, generated, target):
        gen_features = self.content_extractor(generated)
        target_features = self.content_extractor(target)

        content_loss=0
        for gen, target in zip(gen_features, target_features):
            content_loss += self.loss(gen, target)

        return content_loss
    
    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a,b, c * d)
        gram = torch.bmm(features, features.transpose(1,2))
        gram /= (b*c*d)

        return gram
     
    def style_loss(self, generated, target):
        gen_features = self.style_extractor(generated)
        target_features = self.style_extractor(target)

        style_loss = 0.0

        for gen, target in zip(gen_features, target_features):
            G_gen = self.gram_matrix(gen)
            G_target = self.gram_matrix(target)
            style_matrix = G_gen - G_target

            style_loss += torch.norm(style_matrix, p='fro')

        return style_loss

    def cal_perceptual_loss(self, generated, target):
        # grayscale to RGB
        generated = generated.repeat(1,3,1,1)
        target = target.repeat(1,3,1,1)

        if self.content_layers and self.style_layers:
            content_loss = self.content_loss(generated, target)
            style_loss = self.style_loss(generated, target)

            return content_loss, style_loss
        else:
            if self.content_layers:
                content_loss = self.content_loss(generated, target)
                return content_loss, 0
            
            if self.style_layers:
                style_loss = self.style_loss(generated, target)
                return 0, style_loss