import sys
sys.path.append("./scripts")
sys.path.append("./deeplearning/biological_smoother/scripts")

import train_manager
import train_manager_a2s
import train_manager_union
import train_manager_decoder
import train_manager_union_decoder
import data
import trainset
import net

generate_trainset = trainset.generate_trainset
train_unet = train_manager.train
train_a2s_net = train_manager_a2s.train
train_union_net = train_manager_union.train
train_decoder = train_manager_decoder.train
train_union_decoder_net = train_manager_union_decoder.train
