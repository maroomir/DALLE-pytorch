import argparse
import math
import os.path

import numpy
import torch
import torchvision.transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Lambda, Resize, CenterCrop, ToTensor
from tqdm import tqdm
from dalle_pytorch import DiscreteVAE


def load_images(img_path: str, img_size: int = 128) -> ImageFolder:
    transform = torchvision.transforms.Compose([
        Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        Resize(img_size),
        CenterCrop(img_size),
        ToTensor()
    ])
    return torchvision.datasets.ImageFolder(img_path, transform=transform)


def _fit(model: DiscreteVAE, data_loader: DataLoader, optimizer, trace=True):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.train()
    bar = tqdm(enumerate(data_loader))
    losses = []
    for i, (_input, _) in bar:
        _input = _input.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()
        loss = model(_input, return_loss=True)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        if trace:
            bar.set_description(f"{i}/{len(data_loader)}: Loss={numpy.mean(losses)}")
    return numpy.mean(losses)


def train(epoch: int,
          model_path: str,
          train_data: ImageFolder,
          batch_size=4,
          num_workers=0,  # 0: CPU / 4: GPU
          learning_rate=1e-3,
          init_epoch=False,
          **kwargs):
    # Check if we can use a GPU device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("{} device activation".format(device.__str__()))
    # Define the training dataset
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    # Define the VAE Model
    model = DiscreteVAE(image_size=kwargs['image_size'],
                        num_layers=kwargs['num_layers'],
                        num_tokens=kwargs['num_tokens'],
                        codebook_dim=kwargs['emb_dim'],
                        hidden_dim=kwargs['hidden_dim'],
                        num_resnet_blocks=kwargs['num_resnet_blocks'],
                        smooth_l1_loss=kwargs['smooth_l1_loss'],
                        kl_div_loss_weight=kwargs['kl_loss_weight']
                        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)

    def load_model(trace):
        if isinstance(model_path, str) and os.path.exists(model_path) and init_epoch is False:
            model_data = torch.load(model_path, map_location=device)
            epo = model_data['epoch']
            model.load_state_dict(model_data['model'])
            optimizer.load_state_dict(model_data['optimizer'])
            if trace:
                print("## Successfully load the model at {} epochs!".format(epo))
                print("Directory of the pre-trained model: {}".format(model_path))
            return epo
        else:
            return 0

    def save_model(_i, _path: str):
        torch.save({'epoch': _i,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, _path)

    start = load_model(trace=True)
    # Train and Test Repeat
    min_loss = 10000.0
    history = []
    for i in range(start, epoch + 1):
        loss = _fit(model=model, data_loader=train_loader, optimizer=optimizer)
        history.append(loss)
        # Change the learning rate
        scheduler.step()
        # Rollback the model when loss is NaN
        if math.isnan(loss):
            load_model(trace=False)
            print("## Rollback the Model for prevent vanishing")
        # Save the optimal model
        elif loss < min_loss:
            min_loss = loss
            save_model(i, model_path)
        elif i % 100 == 0:
            save_model(i, f'vae_{i}epoch.pth')
    return history


def test(model_path: str,
         test_data: ImageFolder,
         num_worker=0,  # 0: CPU / 4: GPU
         **kwargs):
    # Check if we can use a GPU device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("{} device activation".format(device.__str__()))
    # Define the test dataset
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=num_worker)
    # Load the model
    model = DiscreteVAE(image_size=kwargs['image_size'],
                        num_layers=kwargs['num_layers'],
                        num_tokens=kwargs['num_tokens'],
                        codebook_dim=kwargs['emb_dim'],
                        hidden_dim=kwargs['hidden_dim'],
                        num_resnet_blocks=kwargs['num_resnet_blocks'],
                        smooth_l1_loss=kwargs['smooth_l1_loss'],
                        kl_div_loss_weight=kwargs['kl_loss_weight']
                        ).to(device)
    model.eval()
    model_data = torch.load(model_path)
    model.load_state_dict(model_data['model'])
    print("Successfully load the Model in path")
    # Start the test sequence
    bar = tqdm(test_loader)
    print("Length of data = ", len(bar))
    outputs = []
    for i, _input in enumerate(bar):
        _input = _input.type(torch.FloatTensor).to(device)
        output = model(_input)
        outputs.append(output.detach().cpu().numpy())
    return outputs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True,
                        help='path to your folder of images for learning the discrete VAE and its codebook')
    parser.add_argument('--image_size', type=int, required=False, default=128,
                        help='image size')
    model_group = parser.add_argument_group('Model settings')
    model_group.add_argument('--num_tokens', type=int, default=8192, help='number of image tokens')
    model_group.add_argument('--num_layers', type=int, default=3, help='number of layers (should be 3 or above)')
    model_group.add_argument('--num_resnet_blocks', type=int, default=2, help='number of residual net blocks')
    model_group.add_argument('--smooth_l1_loss', dest='smooth_l1_loss', action='store_true')
    model_group.add_argument('--emb_dim', type=int, default=512, help='embedding dimension')
    model_group.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension')
    model_group.add_argument('--kl_loss_weight', type=float, default=0., help='KL loss weight')
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    opt = parse_opt()
    dataset = load_images(img_path=opt['image_folder'], img_size=opt['image_size'])
    train(epoch=500, model_path='/model/vae_best.pth', **opt)
