import argparse
import math
import os

import torch
import torchvision
from torch import optim
from tqdm import tqdm

from criteria.id_loss import IDLoss
from criteria.lpips.lpips import LPIPS

from models.stylegan2.model import Generator

# from utils import ensure_checkpoint_exists


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def main(args, mean_latent=None, tar_img=None):

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)

    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    mean_latent = g_ema.mean_latent(4096)

    if args.mode == "real":
        # loop through all images in the folder
        target_images = []
        for img in os.listdir(args.img_dir):
            img_path = os.path.join(args.img_dir, img)
            img = torchvision.io.read_image(img_path).unsqueeze(0).cuda().float()
            img = img / 127.5 - 1
            img = img * 2.0
            tar_img = img
            target_images.append(img)
    if args.mode == "synthetic":
        tar_z_codes = torch.randn(args.n_samples, 512).cuda()
        with torch.no_grad():
            target_images, _, _ = g_ema(
                [tar_z_codes], return_latents=True, truncation=args.truncation, truncation_latent=mean_latent
            )

    final_cat_results = []
    inverted_images = []
    for tar_img in target_images:
        tar_img = tar_img.unsqueeze(0)
        latent_code_init = mean_latent.detach().clone().repeat(1, 18, 1)

        with torch.no_grad():
            img_orig, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)

        latent = latent_code_init.detach().clone()
        latent.requires_grad = True

        id_loss = IDLoss(args)
        lpips_loss = LPIPS(net_type="alex").to("cuda:0").eval()

        if args.work_in_stylespace:
            optimizer = optim.Adam(latent, lr=args.lr)
        else:
            optimizer = optim.Adam([latent], lr=args.lr)

        pbar = tqdm(range(args.step))

        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr

            img_gen, _ = g_ema(
                [latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace
            )

            loss_lpips = lpips_loss(img_gen, img_orig)
            if args.id_lambda > 0:
                i_loss = id_loss(img_gen, tar_img)[0]
            else:
                i_loss = 0

            if args.work_in_stylespace:
                l2_loss = sum([((latent_code_init[c] - latent[c]) ** 2).sum() for c in range(len(latent_code_init))])
            else:
                l2_loss = ((latent_code_init - latent) ** 2).sum()
            loss = args.lpips_lambada * loss_lpips + args.l2_lambda * l2_loss + args.id_lambda * i_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description((f"loss: {loss.item():.4f} --- id_loss: {i_loss:.4f}"))

        final_result = torch.cat([tar_img, img_orig, img_gen])
        inverted_images.append(img_gen)
        final_cat_results.append(final_result)

    return final_cat_results, inverted_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default="./pretrained_models/stylegan2-ffhq-config-f.pt",
        help="pretrained StyleGAN2 weights",
    )
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--step", type=int, default=40, help="number of optimization steps")
    parser.add_argument(
        "--mode",
        type=str,
        default="synthetic",
        choices=["real", "synthetic"],
        help="choose between inverting a real image an generate a free one",
    )
    parser.add_argument("--n_samples", type=int, default=8, help="number of samples to generate")
    parser.add_argument(
        "--l2_lambda", type=float, default=0.005, help="weight of the latent distance (used for editing only)"
    )
    parser.add_argument("--lpips_lambada", type=float, default=0.5, help="weight of lpips loss")
    parser.add_argument("--id_lambda", type=float, default=2, help="weight of id loss (used for editing only)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--truncation",
        type=float,
        default=0.7,
        help="used only for the initial latent vector, and only when a latent code path is" "not provided",
    )
    parser.add_argument("--work_in_stylespace", default=False, action="store_true")
    parser.add_argument(
        "--save_intermediate_image_every",
        type=int,
        default=20,
        help="if > 0 then saves intermidate results during the optimization",
    )
    parser.add_argument("--results_dir", type=str, default="inversion_results")
    parser.add_argument(
        "--ir_se50_weights",
        default="./pretrained_models/model_ir_se50.pth",
        type=str,
        help="Path to facial recognition network used in ID loss",
    )

    args = parser.parse_args()

    result_images, _ = main(args)

    for i in range(len(result_images)):
        result_image = result_images[i]
        torchvision.utils.save_image(
            result_image.detach().cpu(),
            os.path.join(args.results_dir, f"final_result_{i}.jpg"),
            normalize=True,
            scale_each=True,
            range=(-1, 1),
        )
