import argparse, os, sys, glob
import torch
import time
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import repeat
import glob
from scipy import ndimage
import torchvision
import lpips
from torch.autograd import Variable

from main import instantiate_from_config
from taming.modules.transformer.mingpt import sample_with_past

rescale = lambda x: (x + 1.) / 2.

def chw_to_pillow(x):
    return Image.fromarray((255*rescale(x.detach().cpu().numpy().transpose(1,2,0))).clip(0,255).astype(np.uint8))

def save_from_logs(logs, logdir, base_count, key="samples", cond_key=None):
    xx = logs[key]
    for i, x in enumerate(xx):
        x = chw_to_pillow(x)
        count = base_count + i
        if cond_key is None:
            rand_int = np.random.randint(0,100000000)
            x.save(os.path.join(logdir, f"{rand_int:06}.png"))
        else:
            condlabel = cond_key[i]
            if type(condlabel) == torch.Tensor: condlabel = condlabel.item()
            os.makedirs(os.path.join(logdir, str(condlabel)), exist_ok=True)
            x.save(os.path.join(logdir, str(condlabel), f"{count:06}.png"))

def save_batch_images(batch, logdir, name_img, key="samples", cond_key=None):
    name_img = name_img.split('/')[-1]
    extension = name_img.split('.')[-1]
    name_img = name_img.replace('.'+extension,'')
    for i, x in enumerate(batch):
        x = chw_to_pillow(x)
        name_save = name_img+f"_{i:06}"+"."+extension
        x.save(os.path.join(logdir, name_save))

def get_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        nargs="?",
        help="path where the samples will be logged to.",
        default=""
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        nargs="?",
        help="num_samples to draw",
        default=50000
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the batch size",
        default=25
    )
    parser.add_argument(
        "--num_optim_steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="?",
        help="number of sampling steps",
        default=2
    )
    parser.add_argument(
        "--top_k_min_selection",
        type=int,
        default=None
    )
    parser.add_argument(
        "--temp_llk",
        type=int,
        default=1
    )
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        nargs="?",
        help="top-k value to sample with",
        default=250,
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        nargs="?",
        help="temperature value to sample with",
        default=1.0
    )
    parser.add_argument(
        "-p",
        "--top_p",
        type=float,
        nargs="?",
        help="top-p value to sample with",
        default=1.0
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="?",
        help="specify comma-separated classes to sample from. Uses 1000 classes per default.",
        default="imagenet"
    )
    parser.add_argument(
        "--bert",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--keep_img",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--mask_collage",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--mask_inference_token",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--random_order",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--collage_frequency",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--dilation_masking",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--dilation_sampling",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--erase_mask_influence",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--progressive_ordering",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--gaussian_smoothing_collage",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--device",
        type=int,
        help="Cuda device id",
        default=0,
    )
    parser.add_argument(
        "--image_list",
        type=str,
        help="Path to folder containing noisy images",
        default=0,
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        help="Path to folder containing noisy images",
        default=0,
    )
    parser.add_argument(
        "--mask_folder",
        type=str,
        help="Path to folder containing noisy images",
        default=0,
    )
    return parser

def create_mask_for_transformer(model,forbidden_tokens):
    mask = torch.ones(model.transformer.block_size,
                model.transformer.block_size)
    mask[:,forbidden_tokens] = 0
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask

@torch.no_grad()
def maximize_token_likelihood(model, img, mask, opt):
    z_code,z_indices = model.encode_to_z(img)
    _,c_indices = model.encode_to_c(img)
    cidx = c_indices
    cz_indices = torch.cat((c_indices, z_indices), dim=1)
    n_toks = model.first_stage_model.quantize.n_e
    toksX,toksY = 16,16
    e_dim = model.first_stage_model.quantize.e_dim

    mask_interpolation = torch.nn.functional.interpolate(mask, (16))[:,0,:]
    mask_interpolation_np = mask_interpolation.cpu().numpy()

    if opt.dilation_sampling>0:
        diff_mask = 1-ndimage.binary_dilation(1-mask_interpolation_np, iterations=opt.dilation_sampling).astype(mask_interpolation_np.dtype)

    unsampled_tokens = np.where(diff_mask[0].flatten()==1)[0]
    sampled_tokens = np.where(diff_mask[0].flatten()==0)[0]
    attention_mask = None

    if opt.dilation_masking>0:
        mask_interpolation_np = 1-ndimage.binary_dilation(1-mask_interpolation_np, iterations=opt.dilation_masking).astype(mask_interpolation_np.dtype)

    mask_interpolation = torch.from_numpy(mask_interpolation_np).to(model.device)
    mask_interpolation = mask_interpolation.view(mask_interpolation.shape[0],-1)

    steps_per_epoch = sampled_tokens.shape[0]
    collage_every = int(steps_per_epoch / opt.collage_frequency)
    if opt.erase_mask_influence:
        r_indices = torch.randint_like(z_indices, 0, model.transformer.config.vocab_size)
        z_indices = mask_interpolation.long()*z_indices + (1-mask_interpolation).long()*r_indices
        r_indices = r_indices.reshape(z_code.shape[0],z_code.shape[2],z_code.shape[3])
    idx = z_indices
    idx = idx.reshape(z_code.shape[0],z_code.shape[2],z_code.shape[3])

    if opt.gaussian_smoothing_collage:
        sigma = 10
        kernel_size = 15
        blurrer = torchvision.transforms.GaussianBlur(kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma))

        mask_collage_Image = mask.cpu().numpy()
        mask_collage_Image = 1-ndimage.binary_dilation(1-mask_collage_Image, iterations=int(kernel_size/2)).astype(mask_interpolation_np.dtype)
        mask_collage_Image = torch.from_numpy(mask_collage_Image).to(device=model.device)
        mask_collage_Image = 1-blurrer(1-mask_collage_Image)
        mask_collage_Image = torch.min(mask_collage_Image,mask)

    if opt.progressive_ordering:
        assert opt.random_order != True
        sampled_tokens_progressive = []
        prev_diff_mask = diff_mask
        while True:
            diff_mask = ndimage.binary_dilation(diff_mask, iterations=1).astype(mask_interpolation_np.dtype)
            dilated_area = diff_mask-prev_diff_mask
            prev_diff_mask = diff_mask
            sampled_tokens_i = np.where(dilated_area[0].flatten()==1)[0]
            for st_i in sampled_tokens_i:
                sampled_tokens_progressive.append(st_i )
            if np.sum(1-diff_mask) == 0:
                break
        sampled_tokens = np.array(sampled_tokens_progressive)

    for e in range(opt.epochs):
        if opt.random_order:
            np.random.shuffle(sampled_tokens)

        if steps_per_epoch <=1:
            break

        for u in range(steps_per_epoch):
            k = e * steps_per_epoch + u
            if opt.mask_collage and u%collage_every == 0 \
                      and (steps_per_epoch - u >= collage_every) \
                      and k != 0:
                xz = model.decode_to_img(idx, z_code.shape)

                if opt.gaussian_smoothing_collage:
                    xz_mask = img * mask_collage_Image + (1-mask_collage_Image) * (xz)
                else:
                    xz_mask = img*mask + (1-mask) * (xz)
                z_code,z_indices = model.encode_to_z(xz_mask)
                z_indices = z_indices.reshape(z_code.shape[0],z_code.shape[2],z_code.shape[3])
                z_indices = z_indices.reshape(z_indices.shape[0],-1)
                idx = z_indices

            z_indices = idx.reshape(idx.shape[0],-1)

            if opt.mask_inference_token:
                jm = (k)%(sampled_tokens.shape[0])
                idx_change = sampled_tokens[jm]
                if not mask_interpolation[:,idx_change]:
                    z_indices[:, idx_change] = torch.randint(model.transformer.config.vocab_size,(1,))
            cz_indices = torch.cat((c_indices, z_indices), dim=1)

            logits, _ = model.transformer.forward(cz_indices[:, :])
            logits = logits[:,c_indices.shape[1]:,]
            logits = logits/opt.temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)

            idx = idx.reshape(idx.shape[0],-1)

            jm = (k)%(sampled_tokens.shape[0])
            idx_change = sampled_tokens[jm]
            logits_s_idx = logits[:,idx_change,:]
            if opt.top_k is not None:
                logits_s_idx = model.top_k_logits(logits_s_idx, opt.top_k)
            probs = torch.nn.functional.softmax(logits_s_idx, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)  ## shape: batch_size X 1
            idx[:,idx_change] = ix[:,0]

            idx = idx.reshape(z_code.shape[0],16,16)

    z = idx.reshape(idx.shape[0],-1)
    z = torch.nn.functional.one_hot(z, n_toks).float()
    z = z @ model.first_stage_model.quantize.embedding.weight
    z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    return z

def load_model_from_config(config, sd, gpu=True, eval_mode=True, device=0):
    model = instantiate_from_config(config)
    if sd is not None:
        model.load_state_dict(sd)
    if gpu:
        model.cuda(device)
    if eval_mode:
        model.eval()
    return {"model": model}


def load_model(config, ckpt, gpu, eval_mode, device):
    # load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
        print(f"loaded model from global step {global_step}.")
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model, pl_sd["state_dict"], gpu=gpu, eval_mode=eval_mode, device=device)["model"]
    return model, global_step

def perceptual_optimization(config, model, z, img, mask, opt):
    xz = model.first_stage_model.decode(z).detach()
    if opt.num_optim_steps > 0:
        z = Variable(z.data, requires_grad=True)
        optimizer = torch.optim.AdamW([z], lr=0.1)
        for j in range(opt.num_optim_steps):
            xz_j = model.first_stage_model.decode(z)
            loss = (opt.lperc(xz_j*mask,img*mask) + opt.lperc(xz_j*(1-mask),xz*(1-mask))).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        xz = xz_j
    return xz


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    parser = get_parser()

    opt, unknown = parser.parse_known_args()
    assert opt.resume

    ckpt = None

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        paths = opt.resume.split("/")
        try:
            idx = len(paths)-paths[::-1].index("logs")+1
        except ValueError:
            idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
        logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), opt.resume
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
    opt.base = base_configs+opt.base
    print('path opt.base: ',opt.base)
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    opt.config = config

    model, global_step = load_model(config, ckpt, gpu=True, eval_mode=True, device=opt.device)
    opt.lperc = lpips.LPIPS(net='vgg').to(model.device)

    if opt.outdir:
        print(f"Switching logdir from '{logdir}' to '{opt.outdir}'")
        logdir = opt.outdir

    if opt.classes == "imagenet":
        given_classes = [i for i in range(1000)]
    else:
        cls_str = opt.classes
        assert not cls_str.endswith(","), 'class string should not end with a ","'
        given_classes = [int(c) for c in cls_str.split(",")]

    base_str = f"top_k_{opt.top_k}_temp_{opt.temperature:.2f}_e_{opt.epochs}"
    if opt.dilation_masking > 0:
        base_str = base_str + f"_dilM{opt.dilation_masking}"
    if opt.dilation_sampling > 0:
        base_str = base_str + f"_dilS{opt.dilation_sampling}"
    if opt.random_order:
        base_str = base_str + "_random"
    if opt.erase_mask_influence:
        base_str =  base_str + "_ers_msk_infl"
    if opt.mask_collage:
        base_str =  base_str + f"_collage_{opt.collage_frequency}"
    else:
        base_str =  base_str + "_noMaskColl"
    if opt.progressive_ordering:
        base_str = base_str + '_prog_ord'
    if opt.gaussian_smoothing_collage:
        base_str = base_str + '_gauss_smooth'
    if opt.mask_inference_token:
        base_str = base_str + '_mask_infToken'
    if opt.num_optim_steps > 0:
        base_str = base_str + f"_nOptim_{opt.num_optim_steps}"
    logdir = os.path.join(logdir, "samples", opt.mask_folder.split('/')[-2], base_str,
                      f"{global_step}")

    print(f"Logging to {logdir}")
    os.makedirs(logdir, exist_ok=True)

    with open(opt.image_list) as f:
        image_list = f.readlines()
        img_names = [os.path.join(opt.image_folder, img_name.replace('\n','')) for img_name in image_list]
        mask_names = [os.path.join(opt.mask_folder, img_name.replace('\n','')) for img_name in image_list]

    for i in range(0,len(mask_names)): #assuming gif
        imgname = img_names[i]
        img = Image.open(imgname).convert('RGB')
        img = np.array(img)
        img = torch.tensor(img.transpose(2,0,1)[None]).to(dtype=torch.float32, device=model.device)
        img = img/127.5-1
        img = torch.nn.functional.interpolate(img, size=(256))

        maskname = mask_names[i]
        mask = Image.open(maskname).convert('RGB')
        mask = np.array(mask)
        mask = torch.tensor(mask.transpose(2,0,1)[None]).to(dtype=torch.float32, device=model.device)
        mask = mask/255
        mask = torch.nn.functional.interpolate(mask, size=(256))
        mask = (mask > 0.5).float()

        if not opt.keep_img:
            img = img * mask + torch.zeros_like(img) * (1-mask)
        img, mask = img.repeat(opt.batch_size,1,1,1), mask.repeat(opt.batch_size,1,1,1)
        print("Image: ", i, imgname, maskname, flush=True)
        z = maximize_token_likelihood(model, img, mask, opt)
        xz = perceptual_optimization(model, z, img, mask, opt)

        save_batch_images(completed_img, logdir, imgname, key="samples", cond_key=None)

    print("done.")
