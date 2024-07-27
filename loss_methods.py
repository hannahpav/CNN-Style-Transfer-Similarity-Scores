import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import torchvision.transforms as transforms


class StyleTransferLoss(nn.Module):
    def __init__(self, content_weight=1.0, style_weight=1.0, perceptual_weight=0.0, tv_weight=0.0,
                 wasserstein_weight=0.0):
        super(StyleTransferLoss, self).__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.perceptual_weight = perceptual_weight
        self.tv_weight = tv_weight
        self.wasserstein_weight = wasserstein_weight

        # Load pre-trained VGG16 model for content and style loss, and perceptual loss
        self.vgg = vgg16(pretrained=True).features
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # Define layers to use for content and style loss
        self.content_layers = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.layers = {
            'conv1_1': 0, 'conv1_2': 1, 'conv2_1': 2, 'conv2_2': 3, 'conv3_1': 4, 'conv3_2': 5, 'conv3_3': 6,
            'conv4_1': 7, 'conv4_2': 8, 'conv4_3': 9, 'conv5_1': 10, 'conv5_2': 11, 'conv5_3': 12
        }

    def forward(self, generated, content, style):
        loss = 0.0

        # Extract features
        generated_features = self.extract_features(generated)
        content_features = self.extract_features(content)
        style_features = self.extract_features(style)

        # Content Loss
        if self.content_weight > 0:
            content_loss = self.compute_content_loss(generated_features, content_features)
            loss += self.content_weight * content_loss

        # Style Loss
        if self.style_weight > 0:
            style_loss = self.compute_style_loss(generated_features, style_features)
            loss += self.style_weight * style_loss

        # Perceptual Loss
        if self.perceptual_weight > 0:
            perceptual_loss = self.compute_perceptual_loss(generated, content)
            loss += self.perceptual_weight * perceptual_loss

        # Total Variation Loss
        if self.tv_weight > 0:
            tv_loss = self.compute_tv_loss(generated)
            loss += self.tv_weight * tv_loss

        # Wasserstein Loss
        if self.wasserstein_weight > 0:
            wasserstein_loss = self.compute_wasserstein_loss(generated, style)
            loss += self.wasserstein_weight * wasserstein_loss

        return loss

    def preprocess(self, img):
        img = (img - self.mean.to(img.device)) / self.std.to(img.device)
        return img

    def extract_features(self, img):
        features = []
        x = img
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features.append(x)
        return features

    def compute_content_loss(self, generated_features, content_features):
        content_loss = 0.0
        for i, layer in enumerate(self.content_layers):
            generated_feature = generated_features[self.layers[layer]]
            content_feature = content_features[self.layers[layer]]
            content_loss += F.mse_loss(generated_feature, content_feature)
        return content_loss

    def compute_style_loss(self, generated_features, style_features):
        style_loss = 0.0
        for i, layer in enumerate(self.style_layers):
            generated_feature = generated_features[self.layers[layer]]
            style_feature = style_features[self.layers[layer]]
            generated_gram = self.gram_matrix(generated_feature)
            style_gram = self.gram_matrix(style_feature)
            style_loss += F.mse_loss(generated_gram, style_gram)
        return style_loss

    def compute_perceptual_loss(self, generated, target):
        generated_features = self.vgg(self.preprocess(generated))
        target_features = self.vgg(self.preprocess(target))
        perceptual_loss = F.mse_loss(generated_features, target_features)
        return perceptual_loss

    def compute_tv_loss(self, img):
        tv_loss = torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])) + torch.sum(
            torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
        return tv_loss

    def compute_wasserstein_loss(self, generated, target):
        generated_flat = generated.view(generated.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        wasserstein_loss = torch.mean(generated_flat) - torch.mean(target_flat)
        return wasserstein_loss

    def gram_matrix(self, x):
        (b, c, h, w) = x.size()
        features = x.view(b, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G.div(c * h * w)

