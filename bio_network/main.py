from collections import OrderedDict

import torch

import scripts as sc

if __name__ == "__main__":
    # mode
    mode = "train_union_net"
    # mode = "train_union_decoder_net"
    # mode = "generate trainset 256"

    if mode == "train_unet":
        parameter_set = OrderedDict(
            lr = [1e-5],
            batch_size = [42] # the max batch size 50, do not change, except vgg loss
        )
        sc.train_unet(parameter_set)
    
    elif mode == "train_a2s_net":
        parameter_set = OrderedDict(
            lr = [0.001],
            batch_size = [128]
        )
        sc.train_a2s_net(parameter_set)      

    elif mode == "train_union_net":
        parameter_set = OrderedDict(
            # lr = [0.001], 
            batch_size = [32] # the max batch size 50, do not change, expect vgg loss
        )
        sc.train_union_net(parameter_set)
    elif mode == "train_decoder_net":
        parameter_set = OrderedDict(
            lr = [0.001],
            batch_size = [128] 
        )
        sc.train_decoder(parameter_set)   
    elif mode == "train_union_decoder_net":
        parameter_set = OrderedDict(
            # lr = [0.001],
            batch_size = [30] 
        )
        sc.train_union_decoder_net(parameter_set)  

    elif mode == "generate trainset 256":
        # albedo patch path
        albedo_patch_path = "/data/wq/biological_smoother/data/train_set/albedo/"
        # trainset path
        trainset_path = "/data/wq/biological_smoother/data/train_set/"
        # module path
        a2s_module_path = "/data/wq/biological_smoother/model/a2s_39.pt"
        # resolution
        res = 256
        # generate function
        sc.generate_trainset(albedo_patch_path,trainset_path,a2s_module_path,res)
    
    else:
        print("Nothing happend")

# AVALIABLE GPU ID: 4 5 6 7


