import argparse
import os
import numpy as np
import torch
import torch.nn as nn

from torchvision import models
from tqdm import tqdm
from models import *
from utils import *
from train import *
from metrics import *
import sys


def main():

    # Add arguments
    parser = argparse.ArgumentParser(description='Conditional GANs and cDCGANs')
    parser.add_argument('--mode', type=str, default='cGAN', help='cGAN or cDCGAN or cDCGAN2')
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--embed_size', type=int, default=5)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_display', type=int, default=10)
    parser.add_argument('--pre_trained', type=bool, default=False)
    parser.add_argument('--data_name', type=str, default='MNIST', help='MNIST or FashionMNIST or CIFAR or CIFAR100 or INTEL')
    parser.add_argument('--use_resnet', type=bool, default=False)
    parser.add_argument('--all_fake', type=bool, default=False)
    parser.add_argument('--all_real', type=bool, default=False)
    args = parser.parse_args()

    printt(args, 'y')

    # Setup Parameters
    mode = args.mode
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    z_dim = args.z_dim
    embed_size = args.embed_size
    num_classes = args.num_classes
    num_display = args.num_display
    pre_trained = args.pre_trained
    data_name = args.data_name
    use_resnet = args.use_resnet
    all_fake = args.all_fake
    all_real = args.all_real

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    printt(f"Device: {device}", "r")

    # Create necessary directories
    data_path, model_path, samples_path = create_dir(data_name, mode)

    # Get data loader
    dataset, num_channels, num_epochs = get_dataset(data_name, data_path)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    # Initializing Models
    if mode == 'cGAN':
        generator_layer_sizes = [256, 512, 1024]
        discriminator_layer_sizes = [1024, 512, 256]
        generator = BaseGenerator(generator_layer_sizes, z_dim, 32, num_classes, num_channels)
        discriminator = BaseDiscriminator(discriminator_layer_sizes, 32, num_classes, num_channels)
    elif mode == 'cDCGAN':
        generator = Generator(z_dim, num_classes, embed_size, num_channels)
        discriminator = Discriminator(num_classes, num_channels)
    elif mode == 'cDCGAN2':
        generator = Generator2(z_dim, num_classes, embed_size, num_channels)
        discriminator = Discriminator2(num_classes, num_channels, 64, device)
    else:
        print("Not a valid mode option")
        raise ValueError

    # Show model summary
    show_model(generator, discriminator)

    # Move to device
    generator, discriminator = generator.to(device), discriminator.to(device)

    # Fix images for sampling during training
    fixed_noise = torch.randn(num_display * num_classes, z_dim).to(device)
    fixed_label = torch.arange(0, num_classes)
    fixed_label = torch.repeat_interleave(fixed_label, num_display).to(device)

    # Load pre-trained models

    if pre_trained:
        generator.load_state_dict(torch.load(os.path.join(model_path, 'generator.pkl')))
        discriminator.load_state_dict(torch.load(os.path.join(model_path, 'discriminator.pkl')))
        printt("Pre-trained models loaded successfully!", "g")

        # Generate images
        generate_sample_images(generator, samples_path, num_display, fixed_noise, fixed_label, "pretrained")
    else:
        # Define optimizers for generator and discriminator
        generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=weight_decay)
        discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=weight_decay)

        # Loss function
        loss_fn = nn.BCELoss()

        # Labels
        real_label = torch.ones(batch_size).to(device)
        fake_label = torch.zeros(batch_size).to(device)

        printt("Conditional GAN training started", "y")
        # Train the conditional GANs
        generator, discriminator = train_conditional_gan(
            data_loader, 
            generator, 
            discriminator, 
            generator_optimizer, 
            discriminator_optimizer, 
            loss_fn, 
            fixed_noise, 
            fixed_label, 
            model_path, 
            samples_path, 
            num_epochs, 
            z_dim, 
            device, 
            real_label, 
            fake_label, 
            num_display
        )


    # Part 2: Train a classifier on fake images to determine quality

    train_dataset, valid_dataset, test_dataset = get_train_val_test(data_name, data_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    if use_resnet:
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(num_channels, batch_size, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes) 
    else:
        if num_channels == 1:
            model = SoftmaxRegression(num_channels, num_classes)#
        else:
            model = SimpleConvNet(num_channels, num_classes)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


    printt("Classifier training started", "y")

    if all_real:
        printt("All Real Images!", 'r')
        model_path = os.path.join('Real', model_path)
        os.makedirs(model_path, exist_ok = True)

        loss_train, acc_train, loss_valid, acc_valid = train_model_on_real_images(
            train_loader, 
            valid_loader, 
            z_dim, 
            device, 
            model, 
            optimizer, 
            criterion, 
            model_path
        )
        acc_test = test_model_on_real_images(test_loader, z_dim, device, model)
        record_statistics(loss_train, loss_valid, acc_train, acc_valid, acc_test, 'None', 'None', model_path)
        done()
        # sys.exit()


    if all_fake:
        printt("All Fake Images!", 'r')

        loss_train, acc_train, loss_valid, acc_valid = train_model_on_fake_images(
            generator, 
            train_loader, 
            valid_loader, 
            z_dim, 
            device, 
            model, 
            optimizer, 
            criterion, 
            model_path
        )
    else:
        printt("Mixed Images!", 'r')
        loss_train, acc_train, loss_valid, acc_valid = train_model_on_combined_images(
            generator, 
            train_loader, 
            valid_loader, 
            z_dim, 
            device, 
            model, 
            optimizer, 
            criterion, 
            model_path
        )


    acc_test = test_model_on_fake_images(generator, test_loader, z_dim, device, model)
    

    if data_name == "INTEL" or data_name == "CIFAR" or data_name == "CIFAR100":

        printt("IS & FID Scores Calculation Started", "y")

        # Calculate IS and FID scores
        fid_score, is_score = calculate_is_fid(test_loader, generator, device, z_dim)
        record_statistics(loss_train, acc_train, loss_valid, acc_valid, acc_test, fid_score, is_score, model_path)
    else:

        record_statistics(loss_train, loss_valid, acc_train, acc_valid, acc_test, 'None', 'None', model_path)

    printt("Training and Testing Done!", "y")

    # Surprise!
    done()


if __name__ == '__main__':
    main()
