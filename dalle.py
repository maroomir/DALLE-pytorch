import argparse
import itertools
import math
import os.path

import numpy
import torch

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dalle_pytorch import DALLE, DiscreteVAE
from loader import load_dataset, load_images, load_labels


class Token:
    PARSE_TO_DEFAULT = "default"
    PARSE_TO_NDARRAY = "ndarray"
    PARSE_TO_TENSOR = "tensor"

    def __init__(self, labels: list):
        self._org = labels
        self._all = list(sorted(frozenset(list(itertools.chain.from_iterable(self._org)))))

    @property
    def pairs(self):
        return dict(zip(self._all, range(1, len(self._all) + 1)))

    @property
    def num_pairs(self):
        return len(self.pairs) + 1

    @property
    def sequence_len(self):
        return max(len(cap) for cap in self._org)

    def parse(self, parse_type='default'):
        if parse_type == self.PARSE_TO_DEFAULT:
            return [[self.pairs[w] for w in cap] for cap in self._org]
        elif parse_type == self.PARSE_TO_NDARRAY:
            source = self.parse(self.PARSE_TO_DEFAULT)
            res = numpy.zeros((len(source), self.sequence_len), dtype=numpy.int64)
            for i in range(len(source)):
                res[i, :len(source[i])] = source[i]
            return res
        elif parse_type == self.PARSE_TO_TENSOR:
            return torch.from_numpy(self.parse(self.PARSE_TO_NDARRAY))

    def caption_mask(self):
        return self.parse(parse_type=self.PARSE_TO_TENSOR) != 0


def load_vae_model(model_path: str, device):
    assert os.path.exists(model_path), 'VAE model file does not exist'
    vae_data = torch.load(vae_path)
    hparams, weights = vae_data['hparams'], vae_data['weights']
    vae = DiscreteVAE(**hparams).to(device)
    vae.load_state_dict(weights)
    return vae


def _fit(model: DALLE, data_loader: DataLoader, optimizer, trace=True):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    bar = tqdm(enumerate(data_loader))
    losses = []
    for i, (text_code, image, mask) in bar:
        text_code = text_code.type(torch.FloatTensor).to(device)
        image = image.type(torch.FloatTensor).to(device)
        model.train()
        optimizer.zero_grad()
        loss = model(text=text_code, image=image, mask=mask, return_loss=True)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        if trace:
            bar.set_description(f"{i}/{len(data_loader)}: Loss={numpy.mean(losses)}")
    return numpy.mean(losses)


def train(epoch: int,
          vae_path: str,
          dalle_path: str,
          train_data: Dataset,
          token: Token,
          batch_size=4,
          num_workers=0,  # 0: CPU / 4: GPU
          learning_rate=0.001,
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
    # Construct the DALLE params to save the hyperparameter
    hparams = dict(dim=kwargs['dim'],
                   num_text_tokens=token.num_pairs,  # vocab size for text
                   text_seq_len=token.sequence_len,  # text sequence length
                   depth=kwargs['depth'],  # should aim to be 64
                   heads=kwargs['heads'],  # attention heads
                   dim_head=kwargs['dim_head'],  # attention head dimension
                   attn_dropout=kwargs['attn_dropout'],  # attention dropout
                   ff_dropout=kwargs['ff_dropout']  # feedforward dropout
                   )
    # Define the DALLE Model
    # automatically infer (1) image sequence length and (2) number of image tokens
    model = DALLE(vae=load_vae_model(vae_path, device), **hparams).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)

    def load_model(trace):
        if isinstance(dalle_path, str) and os.path.exists(dalle_path) and init_epoch is False:
            model_data = torch.load(dalle_path, map_location=device)
            epo, weights, optim = model_data['epoch'], model_data['weights'], model_data['optimizer']
            model.load_state_dict(weights)
            optimizer.load_state_dict(optim)
            if trace:
                print(f"## Successfully load the model at {epo} epochs!")
                print(f"Directory of the pre-trained model: {dalle_path}")
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
            save_model(i, dalle_path)
        elif i % 100 == 0:
            save_model(i, f'vae_{i}epoch.pth')
    return history


def test(vae_path: str,
         dalle_path: str,
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
    model_data = torch.load(dalle_path)
    hparams, weights = model_data['hparams'], model_data['weights']
    model = DALLE(vae=load_vae_model(vae_path, device),
                  num_text_tokens=token.num_pairs,
                  text_seq_len=token.sequence_len,
                  **hparams).to(device)
    model.load_state_dict(weights)
    model.eval()
    print("Successfully load the Model in path")
    # Start the test sequence
    pass


def _parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', type=int, default=128, help='image size')
    model_group = parser.add_argument_group('Model Settings')
    model_group.add_argument('--dim', default=1024, type=int, help='Model dimension')
    model_group.add_argument('--depth', default=12, type=int, help='Model depth')
    model_group.add_argument('--heads', default=16, type=int, help='Model number of heads')
    model_group.add_argument('--dim_head', default=64, type=int, help='Model head dimension')
    model_group.add_argument('--ff_dropout', default=0.1, type=float, help='Feed forward dropout')
    model_group.add_argument('--attn_dropout', default=0.1, type=float, help='Feed forward dropout')
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    opt = _parse_opt()
    train_path = './data/train'
    test_path = './data/test'
    vae_path = './model/vae_best.pth'
    dalle_path = './model/dalle_best.pth'
    train_set = load_dataset(img_path=train_path, img_size=opt['image_size'])
    train_labels = load_labels(source=train_set)
    token = Token(train_labels)
    print(token.pairs)
    print(token.sequence_len)
    print(token.parse())
    print(token.parse(token.PARSE_TO_TENSOR))
    print(token.caption_mask())
    if not os.path.exists(dalle_path):
        train_set = load_dataset(img_path=train_path, img_size=opt['image_size'])
        logs = train(epoch=500, train_data=train_set, dalle_path=dalle_path, vae_path=vae_path, learning_rate=0.001,
                     token=token, **opt)
