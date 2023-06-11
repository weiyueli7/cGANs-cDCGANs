from ignite.metrics import FID, InceptionScore
from ignite.engine import Engine
import ignite.distributed as idist
import torch
import torchvision
from utils import *


def evaluation_step(engine, batch):
    """
    Evaluation step function
    :param engine: Ignite Engine
    :param batch: Batch of data
    :return: FID and IS
    """
    with torch.no_grad():
        net.eval()

        images, labels = batch
        images, labels = images.to(device_), labels.to(device_)
        noise = torch.randn(images.size(0), noise_dim_).to(device_)
        fake_batch = net(noise, labels)
        
        fake = torchvision.transforms.Resize((299, 299), antialias=None)(fake_batch)
        real = torchvision.transforms.Resize((299, 299), antialias=None)(images)
        return fake, real


def calculate_is_fid(test_loader, generator, device, noise_dim):
    """
    Calculate IS and FID scores
    :param test_loader: Data loader
    :param generator: Generator
    :param device: Device
    :param noise_dim: Noise dimension
    :return: IS and FID scores
    """
    global net
    net = generator

    global device_
    device_ = device

    global noise_dim_
    noise_dim_ = noise_dim

    fid_metric = FID(device=device_)
    is_metric = InceptionScore(
        device=device_, 
        output_transform=lambda x: x[0]
    )
    evaluator = Engine(evaluation_step)
    fid_metric.attach(evaluator, "fid")
    is_metric.attach(evaluator, "is")

    evaluator.run(test_loader, max_epochs=1)
    metrics = evaluator.state.metrics
    fid_score = metrics['fid']
    is_score = metrics['is']
    printt(f"*   FID : {fid_score:4f}", 'r')
    printt(f"*    IS : {is_score:4f}", 'r')
    return fid_score, is_score
