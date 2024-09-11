from cgi import print_arguments
import torch
from torch.utils.data import Dataset
import pyexr
import numpy as np
import os
from tqdm import trange

def reshape(array):
    R = array[:,:,0]
    G = array[:,:,1]
    B = array[:,:,2]
    array = []
    array.append(R)
    array.append(G)
    array.append(B)
    array = np.array(array)
    return array




# only aledo dataset
class alb_dataset(Dataset):
    def __init__(self,dataset_path) -> None:
        super().__init__()
        self.albedos = []
        self.gt_melanins = []
        self.gt_hemoglobins = []
        self.gt_betas = []

        print("load albedo data begin")
        albedo_path = os.path.join(dataset_path,"train_set/albedo_new_patch/")
        melanin_path = os.path.join(dataset_path,"GT/melanin_new_patch/")
        hemoglobin_path = os.path.join(dataset_path,"GT/hemoglobin_new_patch/")
        beta_path = os.path.join(dataset_path,"GT/beta_new_patch/")

        albedos = os.listdir(albedo_path)

        self.length = len(albedos)
        self.length = int(self.length / 4.0)
        print(self.length)

        for i in trange(self.length):
            init = 3
            self.albedos.append(reshape(pyexr.open(os.path.join(albedo_path,f"{i*4+init}.exr")).get()))
            self.gt_melanins.append(pyexr.open(os.path.join(melanin_path,f"{i*4+init}.exr")).get().squeeze())
            self.gt_hemoglobins.append(pyexr.open(os.path.join(hemoglobin_path,f"{i*4+init}.exr")).get().squeeze())
            self.gt_betas.append(pyexr.open(os.path.join(beta_path,f"{i*4+init}.exr")).get().squeeze())
        
        # do not to tensor here!

        print("load data end")

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        albedo = torch.tensor(self.albedos[index])
        gt_melanin = torch.tensor(self.gt_melanins[index])
        gt_hemoglobin = torch.tensor(self.gt_hemoglobins[index])
        gt_beta = torch.tensor(self.gt_betas[index])

        
        return albedo,gt_melanin,gt_hemoglobin,gt_beta

# aledo + a2s biological dataset
class alb_bio_dataset(Dataset):
    def __init__(self,dataset_path) -> None:
        super().__init__()
        self.albedos = []
        self.gt_melanins = []
        self.gt_hemoglobins = []
        self.gt_betas = []
        self.a2s_melanins = []
        self.a2s_hemoglobins = []
        self.a2s_betas = []


        print("load albedo and biological data begin")
        albedo_path = os.path.join(dataset_path,"train_set/albedo/")
        a2s_melanin_path = os.path.join(dataset_path,"train_set/melanin")
        a2s_hemoglobin_path = os.path.join(dataset_path,"train_set/hemoglobin")
        a2s_beta_path = os.path.join(dataset_path,"train_set/beta")
        melanin_path = os.path.join(dataset_path,"GT/melanin/")
        hemoglobin_path = os.path.join(dataset_path,"GT/hemoglobin/")
        beta_path = os.path.join(dataset_path,"GT/beta/")

        albedos = os.listdir(albedo_path)

        self.length = len(albedos)

        for i in trange(self.length):
            # train set
            self.albedos.append(reshape(pyexr.open(os.path.join(albedo_path,f"{i+1}.exr")).get()))
            self.a2s_melanins.append(pyexr.open(os.path.join(a2s_melanin_path,f"{i+1}.exr")).get().squeeze())
            self.a2s_hemoglobins.append(pyexr.open(os.path.join(a2s_hemoglobin_path,f"{i+1}.exr")).get().squeeze())
            self.a2s_betas.append(pyexr.open(os.path.join(a2s_beta_path,f"{i+1}.exr")).get().squeeze())

            # ground truth
            self.gt_melanins.append(pyexr.open(os.path.join(melanin_path,f"{i+1}_melanin.exr")).get().squeeze())
            self.gt_hemoglobins.append(pyexr.open(os.path.join(hemoglobin_path,f"{i+1}_hemoglobin.exr")).get().squeeze())
            self.gt_betas.append(pyexr.open(os.path.join(beta_path,f"{i+1}_beta.exr")).get().squeeze())
        
        # do not to tensor here!

        print("load data end")

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # combine albedo, melanin, hemoglobin and beta to one tensor
        albedo = self.albedos[index]
        a2s_melanin = np.expand_dims(self.a2s_melanins[index],0)
        a2s_hemoglobin = np.expand_dims(self.a2s_hemoglobins[index],0)
        a2s_beta = np.expand_dims(self.a2s_betas[index],0)
        albedo = torch.tensor(np.concatenate((albedo,a2s_melanin,a2s_hemoglobin,a2s_beta),axis=0))

        gt_melanin = torch.tensor(self.gt_melanins[index])
        gt_hemoglobin = torch.tensor(self.gt_hemoglobins[index])
        gt_beta = torch.tensor(self.gt_betas[index])

        return albedo,gt_melanin,gt_hemoglobin,gt_beta


# load decoder data
class decoder_dataset(Dataset):
    def __init__(self,dataset_path) -> None:
        super().__init__()
        self.albedos = []
        self.melanins = []
        self.hemoglobins = []
        self.betas = []


        print("load decoder data begin")
        albedo_path = os.path.join(dataset_path,"train_set/albedo/")
        biological_path = os.path.join(dataset_path,"train_set/biological_decoder/")
        albedos = os.listdir(albedo_path)
        self.length = len(albedos)


        for i in trange(self.length):
            self.albedos.append(pyexr.open(os.path.join(albedo_path,f"{i+1}.exr")).get())
            self.melanins.append(pyexr.open(os.path.join(biological_path,f"{i+1}_melanin.exr")).get().squeeze())
            self.hemoglobins.append(pyexr.open(os.path.join(biological_path,f"{i+1}_hemoglobin.exr")).get().squeeze())
            self.betas.append(pyexr.open(os.path.join(biological_path,f"{i+1}_beta.exr")).get().squeeze())
        print("load data end")

    def __len__(self):
        return self.length


    def __getitem__(self, index):
        albedo = torch.tensor(self.albedos[index])
        melanin = torch.tensor(self.melanins[index])
        hemoglobin = torch.tensor(self.hemoglobins[index])
        beta = torch.tensor(self.betas[index])
        biological_input = torch.stack([melanin,hemoglobin,beta],dim=2)

        
        return biological_input,albedo


# enforce dataset
class enforce_dataset(Dataset):
    def __init__(self,dataset_path) -> None:
        super().__init__()
        self.albedos = []
        self.gt_melanins = []
        self.gt_hemoglobins = []
        self.gt_betas = []

        print("load enforce data begin")
        albedo_path = os.path.join(dataset_path,"train_set/albedo_enforce/")
        melanin_path = os.path.join(dataset_path,"GT/melanin_enforce/")
        hemoglobin_path = os.path.join(dataset_path,"GT/hemoglobin_enforce/")
        beta_path = os.path.join(dataset_path,"GT/beta_enforce/")

        albedos = os.listdir(albedo_path)

        self.length = len(albedos)

        for i in trange(self.length):
            self.albedos.append(reshape(pyexr.open(os.path.join(albedo_path,f"{i+1}.exr")).get()))
            self.gt_melanins.append(pyexr.open(os.path.join(melanin_path,f"{i+1}.exr")).get().squeeze())
            self.gt_hemoglobins.append(pyexr.open(os.path.join(hemoglobin_path,f"{i+1}.exr")).get().squeeze())
            self.gt_betas.append(pyexr.open(os.path.join(beta_path,f"{i+1}.exr")).get().squeeze())
        
        # do not to tensor here!

        print("load data end")

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        albedo = torch.tensor(self.albedos[index])
        gt_melanin = torch.tensor(self.gt_melanins[index])
        gt_hemoglobin = torch.tensor(self.gt_hemoglobins[index])
        gt_beta = torch.tensor(self.gt_betas[index])

        
        return albedo,gt_melanin,gt_hemoglobin,gt_beta

    