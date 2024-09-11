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
import vggloss


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
    index = 31
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
        if dataset == "albedo only":
            unet = net.UNet().to(device)
        else:
            unet = net.UNet(in_ch=6,out_ch=3).to(device)
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
        optimizer = torch.optim.Adam(unet.parameters(), lr=run.lr, weight_decay = 1e-4)
        # tensorboard
        tb = SummaryWriter(comment=comment)
        # vgg
        vgg = vggloss.VGGPerceptualLoss().to(device)

        for epoch in trange(epoch):
            unet.train()
            count = 0
            average_loss = 0
            melanin_loss = 0
            hemoglobin_loss = 0
            beta_loss = 0
            # train one epoch
            for batch in train_loader:
                albedos,gt_cms,gt_chs,gt_betas = batch
                albedos = albedos.to(device)
                gt_cms = gt_cms.to(device)
                gt_chs = gt_chs.to(device)
                gt_betas = gt_betas.to(device)

                preds = unet(albedos)

                # TODO: change loss
                
                # l2 loss
                loss_melanin = F.mse_loss(preds[:,0],gt_cms)
                loss_hemoglobin = F.mse_loss(preds[:,1],gt_chs)
                loss_beta = F.mse_loss(preds[:,2],gt_betas)


                # ssim loss
                # gt_cms = torch.unsqueeze(gt_cms,dim=1)
                # gt_chs = torch.unsqueeze(gt_chs,dim=1)
                # gt_betas = torch.unsqueeze(gt_betas,dim=1)
                # biological = torch.concat((gt_cms,gt_chs,gt_betas),dim=1)
                # loss_ssim = 0.1 * (1 - ssim(preds,biological,data_range=1.0))

                # vgg loss
                loss_vgg_melanin = vgg(preds[:,0].unsqueeze(1),gt_cms.unsqueeze(1))
                loss_vgg_hemoglobin = vgg(preds[:,1].unsqueeze(1),gt_chs.unsqueeze(1))
                loss_vgg_beta = vgg(preds[:,2].unsqueeze(1),gt_betas.unsqueeze(1))
                loss_melanin += 0.01*loss_vgg_melanin
                loss_hemoglobin += 0.01*loss_vgg_hemoglobin
                loss_beta += 0.01*loss_vgg_beta

                loss =   loss_melanin + loss_hemoglobin + loss_beta 

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
        torch.save(unet,f"/data/wq/biological_smoother/model/unet_{index}.pt")

        



    
