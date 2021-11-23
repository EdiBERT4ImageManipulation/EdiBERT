import torch
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--checkpoint_source', type=str, default='logs/2021-04-23T18-19-01_ffhq_transformer/checkpoints/last.ckpt')
parser.add_argument('--checkpoint_target', type=str, default='logs/2021-04-23T18-19-01_ffhq_transformer/checkpoints/vqgan.ckpt')

args = parser.parse_args()

pl_sd = torch.load(args.checkpoint_source, map_location="cpu")
print(pl_sd.keys())
## pl_sd is a dictionary
## pl_sd['state_dict'] is an Ordered Dict (<class 'collections.OrderedDict'>) with keys=names; values=weights
## Inspect keys with pl_sd['state_dict'].keys()
## Need to translate: 'first_stage_model.encoder.conv_in.weight' --> 'encoder.conv_in.weight'
## !!!! Maybe for a lighter model --> remove pl_sd['optimizer_states']

keys = pl_sd['state_dict'].keys()

new_pl_sd = pl_sd['state_dict'].copy()
for key in keys:
    if key.startswith('first_stage_model.'):
        print(key)
        new_key = key.replace('first_stage_model.','')
        print(new_key)
        new_pl_sd[new_key] = pl_sd['state_dict'][key]
    else:
        print('noooo: ',key)
    _ = new_pl_sd.pop(key)
pl_sd['state_dict'] = new_pl_sd

torch.save(pl_sd,args.checkpoint_target)
