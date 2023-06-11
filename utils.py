import os
import json
import subprocess
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import random_split
from dataloader import*
import matplotlib.pyplot as plt



def printt(text, color='w'):
    """
    Custom print function
    :param text: Text to print
    :param color: Color of the text
    :return: None
    """
    # Colors: red, green, blue, yellow
    colors = {
        'r': '\033[31m',
        'g': '\033[32m',
        'b': '\033[34m',
        'y': '\033[33m'
    }
    end_color = '\033[0m'
    if color.lower() in colors:
        print(f"{colors[color.lower()]}{text}{end_color}")
    else:
        print(text)

def generate_sample_images(generator, samples_path, num_per_class, z, fixed_label, epoch=0):
    """
    Generate images from generator and save them to samples_path
    :param generator: Generator model
    :param samples_path: Path to save images
    :param num_per_class: Number of images per class
    :param z: Random noise
    :param fixed_label: Fixed label
    :param epoch: Epoch number
    :return: None
    """
    generator.eval()
    fake_imgs = generator(z, fixed_label)
    fake_imgs = (fake_imgs + 1) / 2
    fake_imgs_ = torchvision.utils.make_grid(fake_imgs, normalize=False, nrow=num_per_class)
    torchvision.utils.save_image(fake_imgs_, os.path.join(samples_path, 'sample_' + str(epoch) + '.png'))


def create_dir(dataset, mode):
    """
    Create directories for storing data, models and samples
    :param dataset: Name of the dataset
    :return: Paths to data, model and samples directories
    """
    db_path = os.path.join('./data', dataset)
    os.makedirs(db_path, exist_ok=True)
    model_path = os.path.join(f'./results/{mode}/model', dataset)
    os.makedirs(model_path, exist_ok=True)
    samples_path = os.path.join(f'./results/{mode}/samples', dataset)
    os.makedirs(samples_path, exist_ok=True)
    return db_path, model_path, samples_path


def get_dataset(name, path):
    """
    Get dataset
    :param name: Name of the dataset
    :param path: Path to store the dataset
    :return: Dataset, number of channels and number of epochs
    """

    transform = transforms.Compose(
        [
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )

    if name == 'MNIST':
        n_channels = 1
        epochs = 10
        dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
    elif name == 'FashionMNIST':
        n_channels = 1
        epochs = 10
        dataset = datasets.FashionMNIST(path, train=True, download=True, transform=transform)
    elif name == 'CIFAR':
        n_channels = 3
        epochs = 100
        dataset = datasets.CIFAR10(path, train=True, download=True, transform=transform)
    elif name == 'CIFAR100':
        n_channels = 3
        epochs = 100
        dataset = datasets.CIFAR100(path, train=True, download=True, transform=transform)
    elif name == 'INTEL':
        if len(os.listdir(path)) == 0:
            subprocess.run(['bash', 'get_dataset.sh'])
        transform = transforms.Compose(
        [
            transforms.Resize([32, 32]),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )
        n_channels = 3
        epochs = 100
        dataset = SegmentationDataset(os.path.join(path, 'seg_train/seg_train'), transform=transform)
        torch.manual_seed(7)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        dataset, test_dataset = random_split(dataset, [train_size, test_size])

    else:
        print("Not a valid dataset option")
        raise ValueError
    return dataset, n_channels, epochs



def get_train_val_test(name, path):
    """
    Get train, validation and test dataset
    :param name: Name of the dataset
    :param path: Path to store the dataset
    :return: Train, validation and test dataset
    """
    transform = transforms.Compose(
        [
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )
    torch.manual_seed(7)
    if name == 'MNIST':
        train_dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, valid_dataset = random_split(train_dataset, [train_size, val_size])
        test_dataset = datasets.MNIST(path, train=False, download=True, transform=transform)
    elif name == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(path, train=True, download=True, transform=transform)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, valid_dataset = random_split(train_dataset, [train_size, val_size])
        test_dataset = datasets.FashionMNIST(path, train=False, download=True, transform=transform)
    elif name == 'CIFAR':
        train_dataset = datasets.CIFAR10(path, train=True, download=True, transform=transform)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, valid_dataset = random_split(train_dataset, [train_size, val_size])
        test_dataset = datasets.CIFAR10(path, train=False, download=True, transform=transform)
    elif name == 'CIFAR100':
        train_dataset = datasets.CIFAR100(path, train=True, download=True, transform=transform)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, valid_dataset = random_split(train_dataset, [train_size, val_size])
        test_dataset = datasets.CIFAR100(path, train=False, download=True, transform=transform)
    elif name == 'INTEL':
        if len(os.listdir(path)) == 0:
            subprocess.run(['bash', 'get_dataset.sh'])
        transform = transforms.Compose(
        [
            transforms.Resize([32, 32]),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )
        train_dataset = SegmentationDataset(os.path.join(path, 'seg_train/seg_train'), transform=transform)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, valid_dataset = random_split(train_dataset, [train_size, val_size])
        test_dataset = SegmentationDataset(os.path.join(path, 'seg_test/seg_test'), transform=transform)

    else:
        print("Not a valid dataset option")
        raise ValueError
    return train_dataset, valid_dataset, test_dataset




def show_model(generator, discriminator):
    """
    Show model summary
    :param gen: Generator model
    :param dis: Discriminator model
    :return: None
    """
    printt("------------------------------------\nGenerator Architecture:\n------------------------------------\n", 'r')
    printt(generator, 'b')
    printt("------------------------------------\nDiscriminator Architecture:\n------------------------------------\n", 'r')
    printt(discriminator, 'b')

def plot_discriminator_generator_loss(all_d_loss, all_g_loss, model_path):
    """
    Plot discriminator and generator loss
    :param all_d_loss: List of discriminator loss
    :param all_g_loss: List of generator loss
    :param model_path: Path to save the plot
    :return: None
    """
    plt.figure(figsize=(8, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(all_g_loss, label="G")
    plt.plot(all_d_loss, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(model_path, 'all_loss.png'))
    plt.close()


def plot_train_validation_loss_and_accuracy(loss_train, acc_train, loss_valid, acc_valid, model_path):
    """
    Plot train and validation loss and accuracy
    :param loss_train: List of train loss
    :param acc_train: List of train accuracy
    :param loss_valid: List of validation loss
    :param acc_valid: List of validation accuracy
    :param model_path: Path to save the plot
    :return: None
    """
    plt.figure(figsize=(7, 4))

    # Plot train and validation loss
    plt.subplot(1, 2, 1)
    plt.title("Train and Validation Loss")
    plt.plot(loss_train, label="train")
    plt.plot(loss_valid, label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot train and validation accuracy
    plt.subplot(1, 2, 2)
    plt.title("Train and Validation Accuracy")
    plt.plot(acc_train, label="train")
    plt.plot(acc_valid, label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(model_path, 'train_validation_loss_and_accuracy.png'))
    plt.close()

def record_statistics(loss_train, loss_valid, acc_train, acc_valid, acc_test, fid_score, is_score, model_path):
    """
    
    Save the information as a json file
    :param loss_train: List of train loss
    :param loss_valid: List of validation loss
    :param acc_train: List of train accuracy
    :param acc_valid: List of validation accuracy
    :param acc_test: test accuracy (single value)
    :param model_path: Path to save the plot
    :param fid_score: FID score
    :param is_score: IS score
    :return: None
    """
    record = {
        'train_loss': loss_train,
        'valid_loss': loss_valid,
        'train_acc': acc_train,
        'valid_acc': acc_valid,
        'test_acc': acc_test,
        'is_score': is_score,
        'fid_score': fid_score
    }
    with open(os.path.join(model_path, 'statistics.json'), 'w') as f:
        json.dump(record, f)



def done():
    """
    Print a cute message when training is done
    :return: None
    """
    printt("       /\\", "r")
    printt("      /  \\", 'g')
    printt("     /    \\", 'b')
    printt("    /______\\", 'y')
    printt("   |        |", 'r')
    printt("   |  O  O  |", 'g')
    printt("   |   ~    |", 'b')
    printt("   | \___/  |", 'y')
    printt("   |________|", 'r')
    printt("  /          \\", 'g')
    printt(" /            \\", 'b')
    printt("/______________\\", 'y')
    printt("You are done! Go and take a break!", 'r')
