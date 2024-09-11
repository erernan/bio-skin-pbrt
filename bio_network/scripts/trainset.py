import torch
import os
import pyexr
import numpy
from tqdm import tqdm
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

# reshape the loaded img to 1 dim array, so that it matchs the a2s nn input shape
def load_img(image_path):
    img = pyexr.open(image_path).get()
    img = img.reshape(-1,3)
    img = torch.tensor(img).to("cuda")
    return img

# proess the a2s generated results
def process_result(result,result_path,resolution,name):
    result = torch.reshape(result,(resolution,resolution,3))
    result = result.to("cpu").detach().numpy()
    result_melanin = result[:,:,0]
    result_hemoglobin = result[:,:,1]
    result_beta = result[:,:,2]

    pyexr.write(result_path + "melanin/"+ name, result_melanin)
    pyexr.write(result_path + "hemoglobin/" + name, result_hemoglobin)
    pyexr.write(result_path + "beta/" + name, result_beta)
    return  


# reference a2s nn
def generate_trainset(albedo_path,result_path,nn_path,resolution):
    # load nn module
    a2s_nn = torch.load(nn_path).to("cuda")
    a2s_nn.eval()

    # reference each img
    albedos_name = os.listdir(albedo_path)
    for albedo_name in tqdm(albedos_name):
        alb_path = os.path.join(albedo_path,albedo_name)
        albedo = load_img(alb_path)
        albedo = position_encoding(albedo,2)
        result = a2s_nn(albedo)
        process_result(result,result_path,resolution,albedo_name)
    return