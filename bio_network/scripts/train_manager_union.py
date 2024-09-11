from collections import namedtuple
from itertools import product

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import data
import net
import vggloss

def reshape_tensor(tensor):
    R = tensor[:,:,:,0]
    G = tensor[:,:,:,1]
    B = tensor[:,:,:,2]
    return torch.stack([R,G,B],dim=1)


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
    index = 32
    dataset = "albedo only"

    device = get_device()
    # dataset
    dataset_path = "/data/wq/biological_smoother/data/"
    if  dataset ==  "albedo only":
        train_set = data.alb_dataset(dataset_path=dataset_path)
    else:
        train_set = data.enforce_dataset(dataset_path=dataset_path)
    torch.set_grad_enabled(True)
    # train all product of hyperparams
    for run in run_builder.get_runs(parameters):
        torch.cuda.empty_cache()
        # unet and a2s net instance
        a2s_net = torch.load("/data/wq/biological_smoother/model/a2s_net_30.pt").to(device)
        # unet = torch.load("/data/wq/biological_smoother/model/unet_31.pt").to(device)
        # a2s_net = net.Net().to(device)
        unet = net.UNet(in_ch=6).to(device)
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
        optimizer = torch.optim.Adam([
            {'params': a2s_net.parameters(), 'lr': 0.001, 'weight_decay':1e-4}, 
	        {'params': unet.parameters(), 'lr': 0.0001, 'weight_decay':1e-4}
	        ])
        # tensorboard
        tb = SummaryWriter(comment=comment)
        # vgg
        vgg = vggloss.VGGPerceptualLoss().to(device)

        for epoch in trange(epoch):
            a2s_net.train()
            unet.train()

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
                albedos_a2s = torch.stack((temp0.contiguous().view(-1),temp1.contiguous().view(-1),temp2.contiguous().view(-1)),dim=1)
                albedos_a2s = position_encoding(albedos_a2s,2).to(device)

                gt_cms = gt_cms.to(device)
                gt_chs = gt_chs.to(device)
                gt_betas = gt_betas.to(device)

                preds_a2s = reshape_tensor(a2s_net(albedos_a2s).view(-1,256,256,3))
                albedos = albedos.to(device)
                union_input = torch.cat([albedos,preds_a2s],dim=1)

                preds_unet = unet(union_input)

                # l2 loss
                loss_melanin = F.mse_loss(preds_unet[:,0],gt_cms)
                loss_hemoglobin = F.mse_loss(preds_unet[:,1],gt_chs)
                loss_beta = F.mse_loss(preds_unet[:,2],gt_betas)

                # vgg loss
                loss_vgg_melanin = vgg(preds_unet[:,0].unsqueeze(1),gt_cms.unsqueeze(1))
                loss_vgg_hemoglobin = vgg(preds_unet[:,1].unsqueeze(1),gt_chs.unsqueeze(1))
                loss_vgg_beta = vgg(preds_unet[:,2].unsqueeze(1),gt_betas.unsqueeze(1))
                loss_melanin += 0.01*loss_vgg_melanin
                loss_hemoglobin += 0.01*loss_vgg_hemoglobin
                loss_beta += 0.01*loss_vgg_beta

                loss = loss_melanin + loss_hemoglobin + loss_beta

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
        torch.save(a2s_net,f"/data/wq/biological_smoother/model/union_net_a2s_{index}.pt")
        torch.save(unet,f"/data/wq/biological_smoother/model/union_net_unet_{index}.pt")

        



    
