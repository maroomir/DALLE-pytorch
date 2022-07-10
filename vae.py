import argparse
import math
import os
import os.path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy
import torch
import matplotlib.pyplot

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dalle_pytorch import DiscreteVAE
from loader import load_dataset, load_images


def _fit(model: DiscreteVAE, data_loader: DataLoader, optimizer, trace=True):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    bar = tqdm(enumerate(data_loader))
    losses = []
    for i, (_input, _) in bar:
        _input = _input.type(torch.FloatTensor).to(device)
        model.train()
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
          train_data: Dataset,
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
    print(f"{device.__str__()} device activation")
    # Define the training dataset
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    # Construct the VAE params to save the hyperparameter
    hparams = dict(image_size=kwargs['image_size'],
                   num_layers=kwargs['num_layers'],  # number of downsamples - ex.256 / (2 ** 3) = (32 X 32 feature map)
                   num_tokens=kwargs['num_tokens'],  # number of visual tokens. in the paper, they used 8192
                   codebook_dim=kwargs['emb_dim'],  # codebook dimension
                   hidden_dim=kwargs['hidden_dim'],  # hidden dimension
                   num_resnet_blocks=kwargs['num_resnet_blocks'],  # number of resnet blocks
                   smooth_l1_loss=kwargs['smooth_l1_loss'],
                   kl_div_loss_weight=kwargs['kl_loss_weight'])
    # Define the VAE Model
    model = DiscreteVAE(**hparams).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)

    def load_model(trace):
        if isinstance(model_path, str) and os.path.exists(model_path) and init_epoch is False:
            model_data = torch.load(model_path, map_location=device)
            epo, weights, optim = model_data['epoch'], model_data['weights'], model_data['optimizer']
            model.load_state_dict(weights)
            optimizer.load_state_dict(optim)
            if trace:
                print(f"## Successfully load the model at {epo} epochs!")
                print(f"Directory of the pre-trained model: {model_path}")
            return epo
        else:
            return 0

    def save_model(_i, _path: str):
        root_path = os.path.dirname(_path)
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        torch.save({'epoch': _i,
                    'hparams': hparams,
                    'weights': model.state_dict(),
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
            _path = os.path.join(os.path.dirname(model_path), f'vae_{i}epoch.pth')
            save_model(i, _path)
    return history


def test(model_path: str,
         test_data: Dataset,
         num_worker=0  # 0: CPU / 4: GPU
         ):
    # Check if we can use a GPU device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("{} device activation".format(device.__str__()))
    # Define the test dataset
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=num_worker)
    # Load the model
    model_data = torch.load(model_path)
    hparams, weights = model_data['hparams'], model_data['weights']
    model = DiscreteVAE(**hparams).to(device)
    model.load_state_dict(weights)
    model.eval()
    print("Successfully load the Model in path")
    # Start the test sequence
    bar = tqdm(test_loader)
    print("Length of data = ", len(bar))
    codes, outputs = [], []
    for _input, _ in bar:
        _input = _input.type(torch.FloatTensor).to(device)
        with torch.no_grad():
            code = model.get_codebook_indices(_input)
            output = model.decode(code)
            codes.append(code.detach().cpu().numpy())
            outputs.append(output.detach().cpu().numpy())
    return codes, outputs


def translate(inputs: list):
    return [im.transpose(0, 2, 3, 1).squeeze(axis=0) for im in inputs]


def _parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=128, help='image size')
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


def _trace(code: list):
    code = numpy.concatenate(code, axis=0)
    sample_len, code_len = code.shape
    print(f'ALL {opt["image_size"]} size images are compress by {code_len} codes')
    return sample_len, code_len


def _random_verify(preds: list, targets: list, num_samples: int, num_random=10, fig_size=(10, 10)):
    indexes = numpy.random.choice(num_samples, size=num_random)
    preds, targets = numpy.array(preds), numpy.array(targets)
    preds, targets = numpy.concatenate(preds[indexes, ...], axis=1), numpy.concatenate(targets[indexes, ...], axis=1)
    fair = numpy.concatenate((preds, targets), axis=0)
    matplotlib.pyplot.figure(figsize=fig_size)
    matplotlib.pyplot.imshow(fair)
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.show()


if __name__ == "__main__":
    opt = _parse_opt()
    train_path = './data/train'
    test_path = './data/test'
    model_pth = './model/vae_best.pth'
    if not os.path.exists(model_pth):
        train_set = load_dataset(img_path=train_path, img_size=opt['image_size'])
        logs = train(epoch=500, train_data=train_set, model_path=model_pth, learning_rate=0.001, **opt)
        matplotlib.pyplot.plot(logs)
        matplotlib.pyplot.show()
    test_set = load_dataset(img_path=test_path, img_size=opt['image_size'])
    test_images = load_images(source=test_set)
    codes, decodes = test(model_path=model_pth, test_data=test_set, **opt)
    pred_images = translate(decodes)
    len_sample, _ = _trace(codes)
    _random_verify(pred_images, test_images, len_sample)
