from utils import *
from models import *
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os


def train_conditional_gan(data_loader, generator, discriminator, generator_optimizer, discriminator_optimizer, loss_function, fixed_noise, fixed_labels, model_path, samples_path, num_epochs, noise_dim, device, real_label, fake_label, samples_per_class):
    """
    Train conditional GAN
    :param data_loader: Data loader
    :param generator: Generator model
    :param discriminator: Discriminator model
    :param generator_optimizer: Generator optimizer
    :param discriminator_optimizer: Discriminator optimizer
    :param loss_function: Loss function
    :param fixed_noise: Fixed noise for sampling during training
    :param fixed_labels: Fixed labels for sampling during training
    :param model_path: Path to save the model
    :param samples_path: Path to save the samples
    :param num_epochs: Number of epochs
    :param batch_size: Batch size
    :param noise_dim: Noise dimension
    :param device: Device
    :param real_label: Label for real images
    :param fake_label: Label for fake images
    :param samples_per_class: Number of samples per class
    :return: None
    """
    all_discriminator_losses = []
    all_generator_losses = []
    epoch_avg_generator_loss = []
    epoch_avg_discriminator_loss = []

    # Training
    for epoch in tqdm(range(num_epochs)):
        generator.train()
        discriminator.train()

        batch_discriminator_losses = []
        batch_generator_losses = []
        for real_images, real_labels in data_loader:

            # Loading data
            batch_size_curr = real_images.size(0)
            noise = torch.randn(batch_size_curr, noise_dim).to(device)


            real_images = real_images.to(device)
            real_labels = real_labels.to(device)

            # Generate fake data
            fake_images = generator(noise, real_labels)

            # Train Discriminator
            fake_outputs = discriminator(fake_images.detach(), real_labels)
            real_outputs = discriminator(real_images.detach(), real_labels)
            discriminator_loss = (loss_function(fake_outputs, fake_label) + loss_function(real_outputs, real_label)) / 2

            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Train Generator
            fake_outputs = discriminator(fake_images, real_labels)
            generator_loss = loss_function(fake_outputs, real_label)

            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            batch_discriminator_losses.append(discriminator_loss.item())
            batch_generator_losses.append(generator_loss.item())

        all_discriminator_losses.extend(batch_discriminator_losses)
        all_generator_losses.extend(batch_generator_losses)
        epoch_avg_discriminator_loss.append(sum(batch_discriminator_losses) / len(batch_discriminator_losses))
        epoch_avg_generator_loss.append(sum(batch_generator_losses) / len(batch_generator_losses))

        printt(f"\t Epoch: {epoch + 1}/{num_epochs} \t D_loss: {round(epoch_avg_discriminator_loss[-1], 3)} \t G_loss: {round(epoch_avg_generator_loss[-1], 3)}", 'g')

        # Generate sample images for every 5 epochs
        if (epoch + 1) % 5 == 0:
            generate_sample_images(generator, samples_path, samples_per_class, fixed_noise, fixed_labels, epoch=epoch + 1)

    # Save final model and generate sample images
    torch.save(generator.state_dict(), os.path.join(model_path, 'generator.pkl'))
    torch.save(discriminator.state_dict(), os.path.join(model_path, 'discriminator.pkl'))
    plot_discriminator_generator_loss(all_discriminator_losses, all_generator_losses, model_path)
    generate_sample_images(generator, samples_path, samples_per_class, fixed_noise, fixed_labels)
    return generator, discriminator



def train_model_on_fake_images(generator, train_dataloader, val_dataloader, noise_dim, device, model, optimizer, criterion, model_path):
    """
    Train a classifier on fake images
    :param generator: Generator model
    :param train_dataloader: Train data loader
    :param val_dataloader: Validation data loader
    :param noise_dim: Noise dimension
    :param device: Device
    :param model: Model
    :param optimizer: Optimizer
    :param criterion: Loss function
    :param model_path: Path to save the model
    :return: loss_train, acc_train, loss_valid, acc_valid
    """
    epochs = 50
    loss_train = []
    acc_train = []
    loss_valid = []
    acc_valid = []
    best_val_loss = np.inf
    no_improvement = 0

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        correct = 0
        total = 0
        generator.eval()

        for _, real_labels in train_dataloader:
            batch_size_curr = _.size(0)
            noise = torch.randn(batch_size_curr, noise_dim).to(device)

            real_labels = real_labels.to(device)
            fake_images = generator(noise, real_labels)

            optimizer.zero_grad()

            outputs = model(fake_images)
            loss = criterion(outputs, real_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)

            total += real_labels.size(0)
            correct += predicted.eq(real_labels).sum().item()

        train_loss = running_loss / len(train_dataloader)
        train_acc = correct / total
        loss_train.append(train_loss)
        acc_train.append(train_acc)

        printt(f"\t Epoch: {epoch + 1}/{epochs} \t Train Loss: {round(train_loss, 3)} \t Train Acc: {round(train_acc, 3)}", 'g')

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for _, labels in val_dataloader:
                batch_size_curr = _.size(0)
                noise = torch.randn(batch_size_curr, noise_dim).to(device)
                labels = labels.to(device)

                fake_images = generator(noise, labels)

                outputs = model(fake_images)

                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_dataloader)
        val_acc = val_correct / val_total

        loss_valid.append(val_loss)
        acc_valid.append(val_acc)
        printt(f"\t Epoch: {epoch + 1}/{epochs} \t Validation Loss: {round(val_loss, 3)} \t Validation Acc: {round(val_acc, 3)}", 'b')

        # if epoch > 9 and np.mean(loss_valid[-5:]) > np.mean(loss_valid[-10:-5]):
        #     printt("Early Stopped!" , 'r')
        #     break
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= 5:
            printt("Early Stopped!" , 'r')
            break

    plot_train_validation_loss_and_accuracy(loss_train, acc_train, loss_valid, acc_valid, model_path)

    torch.save(model.state_dict(), os.path.join(model_path, 'classifier.pkl'))

    return loss_train, acc_train, loss_valid, acc_valid

def test_model_on_fake_images(generator, test_dataloader, noise_dim, device, model):
    """
    Test a classifier on fake images
    :param generator: Generator model
    :param test_dataloader: Test data loader
    :param noise_dim: Noise dimension
    :param device: Device
    :param model: Model
    :return: test_acc
    """

    model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for _, labels in test_dataloader:
            batch_size_curr = _.size(0)
            noise = torch.randn(batch_size_curr, noise_dim).to(device)
            labels = labels.to(device)

            fake_images = generator(noise, labels)

            outputs = model(fake_images)

            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = test_correct / test_total

    printt(f"Test Accuracy: {round(test_acc, 3)}", 'g')

    return test_acc


def train_model_on_combined_images(generator, train_dataloader, val_dataloader, noise_dim, device, model, optimizer, criterion, model_path):
    """
    Train a classifier on combined images
    :param generator: Generator model
    :param train_dataloader: Train data loader
    :param val_dataloader: Validation data loader
    :param noise_dim: Noise dimension
    :param device: Device
    :param model: Model
    :param optimizer: Optimizer
    :param criterion: Loss function
    :param model_path: Path to save the model
    :return: loss_train, acc_train, loss_valid, acc_valid
    """
    epochs = 50
    loss_train = []
    acc_train = []
    loss_valid = []
    acc_valid = []
    best_val_loss = np.inf
    no_improvement = 0

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        correct = 0
        total = 0
        generator.eval()

        for real_images, real_labels in train_dataloader:
            batch_size_curr = real_images.size(0) // 2
            noise = torch.randn(batch_size_curr, noise_dim).to(device)

            half_real_labels = real_labels[:batch_size_curr].to(device)

            fake_images = generator(noise, half_real_labels)

            half_real_images = real_images[batch_size_curr:].to(device)

            all_img = torch.cat([fake_images, half_real_images], dim=0)
            
            indices = np.random.permutation(real_images.size(0))

            all_img = all_img[indices].to(device)
            real_labels = real_labels[indices].to(device)

            optimizer.zero_grad()

            outputs = model(all_img)
            loss = criterion(outputs, real_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += real_labels.size(0)
            correct += predicted.eq(real_labels).sum().item()

        train_loss = running_loss / len(train_dataloader)
        train_acc = correct / total
        loss_train.append(train_loss)
        acc_train.append(train_acc)

        printt(f"\t Epoch: {epoch + 1}/{epochs} \t Train Loss: {round(train_loss, 3)} \t Train Acc: {round(train_acc, 3)}", 'g')

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for real_images, real_labels in val_dataloader:
                batch_size_curr = real_images.size(0) // 2
                noise = torch.randn(batch_size_curr, noise_dim).to(device)

                half_real_labels = real_labels[:batch_size_curr].to(device)
        
                fake_images = generator(noise, half_real_labels)

                half_real_images = real_images[batch_size_curr:].to(device)

                all_img = torch.cat([fake_images, half_real_images], dim=0)
                
                indices = np.random.permutation(real_images.size(0))

                all_img = all_img[indices].to(device)
                real_labels = real_labels[indices].to(device)

                outputs = model(all_img)

                loss = criterion(outputs, real_labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += real_labels.size(0)
                val_correct += predicted.eq(real_labels).sum().item()
        
        val_loss = val_loss / len(val_dataloader)
        val_acc = val_correct / val_total

        loss_valid.append(val_loss)
        acc_valid.append(val_acc)
        printt(f"\t Epoch: {epoch + 1}/{epochs} \t Validation Loss: {round(val_loss, 3)} \t Validation Acc: {round(val_acc, 3)}", 'b')

        # if epoch > 9 and np.mean(loss_valid[-5:]) > np.mean(loss_valid[-10:-5]):
        #     printt("Early Stopped!" , 'r')
        #     break
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= 5:
            printt("Early Stopped!" , 'r')
            break

    plot_train_validation_loss_and_accuracy(loss_train, acc_train, loss_valid, acc_valid, model_path)

    torch.save(model.state_dict(), os.path.join(model_path, 'classifier.pkl'))

    return loss_train, acc_train, loss_valid, acc_valid


def train_model_on_real_images(train_dataloader, val_dataloader, noise_dim, device, model, optimizer, criterion, model_path):
    """
    Train a classifier on real images
    :param train_dataloader: Train data loader
    :param val_dataloader: Validation data loader
    :param noise_dim: Noise dimension
    :param device: Device
    :param model: Model
    :param optimizer: Optimizer
    :param criterion: Loss function
    :param model_path: Path to save the model
    :return: loss_train, acc_train, loss_valid, acc_valid
    """
    epochs = 50
    loss_train = []
    acc_train = []
    loss_valid = []
    acc_valid = []
    best_val_loss = np.inf
    no_improvement = 0

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        correct = 0
        total = 0

        for real_images, real_labels in train_dataloader:
            
            real_images, real_labels = real_images.to(device), real_labels.to(device)

            optimizer.zero_grad()

            outputs = model(real_images)
            loss = criterion(outputs, real_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += real_labels.size(0)
            correct += predicted.eq(real_labels).sum().item()

        train_loss = running_loss / len(train_dataloader)
        train_acc = correct / total
        loss_train.append(train_loss)
        acc_train.append(train_acc)

        printt(f"\t Epoch: {epoch + 1}/{epochs} \t Train Loss: {round(train_loss, 3)} \t Train Acc: {round(train_acc, 3)}", 'g')

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for real_images, real_labels in val_dataloader:

                real_images, real_labels = real_images.to(device), real_labels.to(device)

                outputs = model(real_images)

                loss = criterion(outputs, real_labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += real_labels.size(0)
                val_correct += predicted.eq(real_labels).sum().item()
        
        val_loss = val_loss / len(val_dataloader)
        val_acc = val_correct / val_total

        loss_valid.append(val_loss)
        acc_valid.append(val_acc)
        printt(f"\t Epoch: {epoch + 1}/{epochs} \t Validation Loss: {round(val_loss, 3)} \t Validation Acc: {round(val_acc, 3)}", 'b')

        # if epoch > 9 and np.mean(loss_valid[-5:]) > np.mean(loss_valid[-10:-5]):
        #     printt("Early Stopped!" , 'r')
        #     break
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= 7:
            printt("Early Stopped!" , 'r')
            break

    plot_train_validation_loss_and_accuracy(loss_train, acc_train, loss_valid, acc_valid, model_path)

    torch.save(model.state_dict(), os.path.join(model_path, 'classifier.pkl'))

    return loss_train, acc_train, loss_valid, acc_valid

def test_model_on_real_images(test_dataloader, noise_dim, device, model):
    """
    Test a classifier on fake images
    :param generator: Generator model
    :param test_dataloader: Test data loader
    :param noise_dim: Noise dimension
    :param device: Device
    :param model: Model
    :return: test_acc
    """

    model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = test_correct / test_total

    printt(f"Test Accuracy: {round(test_acc, 3)}", 'g')

    return test_acc
