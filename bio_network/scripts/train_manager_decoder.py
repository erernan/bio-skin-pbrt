from collections import namedtuple
from itertools import product

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from pytorch_msssim import ssim

import data
import net


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

class run_builder():
    @staticmethod
    def get_runs(parameters):
        run = namedtuple("Run",parameters.keys())
        runs = []
        for v in product(*parameters.values()):
            runs.append(run(*v))
        return runs

def train(parameters):
    # TODO: other train parameters:
    epoch = 100
    index = 26

    device = get_device()
    # dataset
    dataset_path = "/data/wq/biological_smoother/data/"

    train_set = data.decoder_dataset(dataset_path=dataset_path)

    torch.set_grad_enabled(True)
    # train all product of hyperparams
    for run in run_builder.get_runs(parameters):
        torch.cuda.empty_cache()
        # decoder instance
        decoder = net.Decoder().to(device)
        # tb name
        comment = f'-{run} index:{index}'
        # load data
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size = run.batch_size,
            shuffle = True,
            num_workers = 4
        )
        # optimizer
        optimizer = torch.optim.Adam(decoder.parameters(), lr=run.lr, weight_decay = 1e-4)
        # tensorboard
        tb = SummaryWriter(comment=comment)

        for epoch in trange(epoch):
            decoder.train()
            count = 0
            average_loss = 0

            # train one epoch
            for batch in train_loader:
                biological,albedos = batch
                albedos = albedos.to(device)
                biological = biological.view(-1,3).to(device)

                preds = decoder(biological)
                preds = preds.view(-1,256,256,3)

                # TODO: change loss

                loss =   F.l1_loss(preds,albedos)
                loss_ssim = 0.1 * (1 - ssim(preds,albedos,data_range=1.0))
                loss += loss_ssim

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                average_loss += loss.item()
                count += 1

            tb.add_scalar('Loss', average_loss / count, epoch)
            print("epoch: ",epoch,"  average loss: ", average_loss / count)
        
        tb.close()

        # save module
        torch.save(decoder,f"/data/wq/biological_smoother/model/decoder_{index}.pt")

        



    
