import torch
import torch.nn as nn
    

# Baseline Models
# cGANs

class BaseGenerator(nn.Module):
    """
        Baseline Generator
        Input: Noise vector and labels
        Output: Generated image
    """
    def __init__(self, layer_sizes, noise_size, image_size, class_count, channels=1):
        """
        Constructor for the class
        :param layer_sizes: List of layer sizes
        :param noise_size: Size of noise vector
        :param image_size: Size of image
        :param class_count: Number of classes
        :param channels: Number of channels
        """
        super().__init__()
        self.noise_size = noise_size
        self.image_size = image_size
        self.channels = channels
        self.label_embedding = nn.Embedding(class_count, class_count)
        self.model = nn.Sequential(
            nn.Linear(self.noise_size + class_count, layer_sizes[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.LeakyReLU(0.2),
            nn.Linear(layer_sizes[2], self.image_size * self.image_size * self.channels),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        """
        Forward pass
        :param noise: Noise vector
        :param labels: Labels
        :return: Generated image
        """
        noise = noise.view(-1, self.noise_size)
        label_embedded = self.label_embedding(labels)
        x = torch.cat([noise, label_embedded], dim=1)
        output = self.model(x)
        output = output.view(-1, self.channels, self.image_size, self.image_size)
        return output


class BaseDiscriminator(nn.Module):
    """
        Baseline Discriminator
        Input: Image and labels
        Output: Probability of image being real
    """
    def __init__(self, layer_sizes, image_size, class_count, channels=1):
        """
        Constructor for the class
        :param layer_sizes: List of layer sizes
        :param image_size: Size of image
        :param class_count: Number of classes
        :param channels: Number of channels
        """
        super().__init__()
        self.label_embedding = nn.Embedding(class_count, class_count)
        self.image_size = image_size
        self.channels = channels
        self.model = nn.Sequential(
            nn.Linear(self.image_size * self.image_size * self.channels + class_count, layer_sizes[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(layer_sizes[2], 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        """
        Forward pass
        :param x: Image
        :param labels: Labels
        :return: Probability of image being real
        """
        x = x.view(-1, self.image_size * self.image_size * self.channels)
        label_embedded = self.label_embedding(labels)
        x = torch.cat([x, label_embedded], dim=1)
        output = self.model(x)
        output = output.squeeze()
        return output




# cDCGANs


def weight_init(m):
    """
    Weight initialization for convolutional and batch normalization layers
    :param m: Module
    :return: None
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    cDCGANs Generator
    Input: Noise vector and labels
    Output: Generated image
    """
    def __init__(self, noise_dim=10, num_classes=10, label_embed_dim=5, channels=3, conv_dim=64):
        """
        Constructor for the class
        :param noise_dim: Dimension of noise vector
        :param num_classes: Number of classes
        :param label_embed_dim: Dimension of label embedding
        :param channels: Number of channels
        :param conv_dim: Convolutional layer's dimension
        """
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, label_embed_dim)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(noise_dim + label_embed_dim, conv_dim * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(conv_dim * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(conv_dim * 2, conv_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(conv_dim, channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh()
        )
        for layer in self.modules():
            weight_init(layer)

    def forward(self, noise, label):
        """
        Forward pass
        :param noise: Noise vector
        :param label: Label
        :return: Generated image
        """
        noise = noise.view([noise.shape[0], -1, 1, 1])
        label_embed = self.label_embedding(label)
        label_embed = label_embed.view([label_embed.shape[0], -1, 1, 1])
        # print('noise', noise.shape)
        # print('label embed', label_embed.shape)
        x = torch.cat((noise, label_embed), dim=1)
        # print('x', x.shape)
        output = self.model(x)
        # print('output', output.shape)
        return output


class Discriminator(nn.Module):
    """
    Final Model's Discriminator
    Input: Image and labels
    Output: Probability of image being real
    """
    def __init__(self, num_classes=10, channels=3, conv_dim=64):
        """
        Constructor for the class
        :param num_classes: Number of classes
        :param channels: Number of channels
        :param conv_dim: Convolutional layer's dimension
        """
        super(Discriminator, self).__init__()
        self.image_size = 32
        self.label_embedding = nn.Embedding(num_classes, self.image_size * self.image_size)
        self.model = nn.Sequential(
            nn.Conv2d(channels + 1, conv_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(conv_dim * 4, 1, kernel_size=4, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        for layer in self.modules():
            weight_init(layer)

    def forward(self, x, label):
        """
        Forward pass
        :param x: Image
        :param label: Label
        :return: Probability of image being real
        """
        label_embed = self.label_embedding(label)
        label_embed = label_embed.view([label_embed.shape[0], 1, self.image_size, self.image_size])
        x = torch.cat((x, label_embed), dim=1)
        output = self.model(x)
        output = output.squeeze()
        return output




# cDCGANs2

class Generator2(nn.Module):
    """
    cDCGANs2 Generator
    Input: Noise vector and labels
    Output: Generated image
    """
    def __init__(self, noise_dim=10, num_classes=10, label_embed_dim=5, channels=3, conv_dim=64):
        """
        Constructor for the class
        :param noise_dim: Dimension of noise vector
        :param num_classes: Number of classes
        :param label_embed_dim: Dimension of label embedding
        :param channels: Number of channels
        :param conv_dim: Convolutional layer's dimension
        """
        super(Generator2, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, label_embed_dim)

        self.input_x = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, conv_dim*4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(conv_dim*4),
            nn.ReLU(),
        )
        self.input_y = nn.Sequential(
            nn.ConvTranspose2d( label_embed_dim, conv_dim*4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(conv_dim*4),
            nn.ReLU(),
        )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(conv_dim*8, conv_dim*4, kernel_size=4, stride=2, padding=1, bias=False),    
            nn.BatchNorm2d(conv_dim*4),
            nn.ReLU(True),
 
            nn.ConvTranspose2d( conv_dim*4, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim*2),
            nn.ReLU(True),
 
            nn.ConvTranspose2d(conv_dim*2, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        for layer in self.modules():
            weight_init(layer)

    def forward(self, noise, label):
        """
        Forward pass
        :param noise: Noise vector
        :param label: Label
        :return: Generated image
        """
        noise = noise.view([noise.shape[0], -1, 1, 1])
        label_embed = self.label_embedding(label)
        label_embed = label_embed.view([label_embed.shape[0], -1, 1, 1])

        noise = self.input_x(noise)
        label_embed = self.input_y(label_embed)

        # late concatenation
        x = torch.cat([noise, label_embed], dim=1)
        output = self.model(x)
        return output
 
 
# Discriminator model

class Discriminator2(nn.Module):
    """
    cDCGANs2 Discriminator
    Input: Image and labels
    Output: Probability of image being real
    """
    def __init__(self, num_classes=10, channels=3, conv_dim=64, device='cuda'):
        """
        Constructor for the class
        :param num_classes: Number of classes
        :param channels: Number of channels
        :param conv_dim: Convolutional layer's dimension
        """
        super(Discriminator2, self).__init__()
        self.image_size = 32
        
        self.num_classes = num_classes
        self.conv_dim = conv_dim
        self.device = device

        self.input_x = nn.Sequential(
            nn.Conv2d(channels, conv_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
        )
        self.input_y = nn.Sequential(
            nn.Conv2d(num_classes, conv_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
        )
        self.model = nn.Sequential(
            nn.Conv2d(conv_dim*2 , conv_dim*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim*4),
            nn.LeakyReLU(0.2),
            
 
            nn.Conv2d(conv_dim*4, conv_dim*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim* 8),
            nn.LeakyReLU(0.2),
 
            nn.Conv2d(conv_dim * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        for layer in self.modules():
            weight_init(layer)

    def forward(self, x, label):
        """
        Forward pass
        :param x: Image
        :param label: Label
        :return: Probability of image being real
        """
        one_hot_encoded = torch.zeros(label.size(0), self.num_classes, self.image_size , self.image_size ).to(self.device)
        label_embed = one_hot_encoded.scatter_(1, label.view(-1, 1, 1, 1), 1)
        x = self.input_x(x)
        label_embed = self.input_y(label_embed)

        # late concatenation
        x = torch.cat((x, label_embed), dim=1)
        output = self.model(x)
        output = output.squeeze()
        return output



# Simple Conv Net for Quality Classification

class SimpleConvNet(nn.Module):
    """
    Simple Convolutional Neural Network
    Input: Image
    Output: Probability of image being in a class
    """
    def __init__(self, num_channels, num_classes):
        """
        Constructor for the class
        :param num_channels: Number of channels
        :param num_classes: Number of classes
        """
        super(SimpleConvNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Linear(16 * 16 * 16, num_classes)

    def forward(self, x):
        """
        Forward pass
        :param x: Image
        :return: Probability of image being in a class
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SoftmaxRegression(nn.Module):
    """
    Softmax Regression
    Input: Image
    Output: Probability of image being in a class"""
    def __init__(self, num_channels, num_classes):
        super(SoftmaxRegression, self).__init__()

        self.features = nn.Sequential(
            nn.Flatten(),
        )

        self.linear = nn.Linear(num_channels * 32 * 32, num_classes)

    def forward(self, x):
        """
        Forward pass
        :param x: Image
        :return: Probability of image being in a class
        """
        x = self.features(x)
        logits = self.linear(x)
        probs = torch.softmax(logits, dim=1)
        return probs
