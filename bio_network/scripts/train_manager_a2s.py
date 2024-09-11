from collections import namedtuple
from itertools import product

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import data
import net

def position_encoding(tensor,num_encoding_function):
    encoding = [tensor]
    frequency_bands = torch.linspace(
        2.0 ** 0.0,
        2.0 ** (num_encoding_function - 1),
        num_encoding_function,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    for freq in frequency_bands:
        for func in [torch.sin,torch.cos]:
            encoding.append(func(tensor*freq))
    return torch.cat(encoding,dim=-1) 

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
    index = 30
    dataset = "albedo only"

    device = get_device()
    # dataset
    dataset_path = "/data/wq/biological_smoother/data/"
    if  dataset ==  "albedo only":
        train_set = data.alb_dataset(dataset_path=dataset_path)
    else:
        train_set = data.alb_bio_dataset(dataset_path=dataset_path)
    torch.set_grad_enabled(True)
    # train all product of hyperparams
    for run in run_builder.get_runs(parameters):
        torch.cuda.empty_cache()
        # unet instance
        a2s_net = net.Net().to(device)
        # tb name
        comment = f'-{run} index:{index}'
        # load data
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size = run.batch_size,
            shuffle = True,
            num_workers = 8
        )
        # optimizer
        optimizer = torch.optim.Adam(a2s_net.parameters(), lr=run.lr, weight_decay = 1e-4)
        # tensorboard
        tb = SummaryWriter(comment=comment)

        for epoch in trange(epoch):
            a2s_net.train()
            count = 0
            average_loss = 0
            melanin_loss = 0
            hemoglobin_loss = 0
            beta_loss = 0
            # train one epoch
            for batch in train_loader:
                albedos,gt_cms,gt_chs,gt_betas = batch
                # reshape albedos,cm,ch,beta to MPL input batch size
                temp0 = albedos[:,0]
                temp1 = albedos[:,1]
                temp2 = albedos[:,2]
                albedos = torch.stack((temp0.contiguous().view(-1),temp1.contiguous().view(-1),temp2.contiguous().view(-1)),dim=1)
                albedos = position_encoding(albedos,2).to(device)
                # print(albedos.shape)
                gt_cms = gt_cms.to(device).view(-1)
                gt_chs = gt_chs.to(device).view(-1)
                gt_betas = gt_betas.to(device).view(-1)

                preds = a2s_net(albedos)

                # l1 loss
                loss_melanin = F.l1_loss(preds[:,0],gt_cms)
                loss_hemoglobin = F.l1_loss(preds[:,1],gt_chs)
                loss_beta = F.l1_loss(preds[:,2],gt_betas)

                loss =   loss_melanin + loss_hemoglobin + loss_beta + 20.0 * F.mse_loss(F.relu(-1.0 * preds),torch.zeros_like(preds).to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                average_loss += loss.item()
                melanin_loss += loss_melanin.item()
                hemoglobin_loss += loss_hemoglobin.item()
                beta_loss += loss_beta.item()
                count += 1

            tb.add_scalar('Loss', average_loss / count, epoch)
            tb.add_scalar('Melanin Loss', melanin_loss / count, epoch)
            tb.add_scalar('Hemoglobin Loss', hemoglobin_loss / count, epoch)
            tb.add_scalar('Beta Loss', beta_loss / count, epoch)
            print("epoch: ",epoch,"  average loss: ", average_loss / count)
        
        tb.close()

        # save module
        torch.save(a2s_net,f"/data/wq/biological_smoother/model/a2s_net_{index}.pt")

        



    
