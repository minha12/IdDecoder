import argparse

from argparse import Namespace
import torch
from mapper.styleclip_mapper import StyleCLIPMapper
from criteria.id_loss import IDLoss
import torchvision.utils
import os


def main(args):
    # make result directory if not present, exists_ok=True is for concurrent runs

    os.makedirs(args.save_dir, exist_ok=True)
    # update test options with options used during training
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    opts = ckpt["opts"]
    opts.update(vars(args))
    opts = Namespace(**opts)
    # print(opts)
    net = StyleCLIPMapper(opts)
    net.eval()
    net.cuda()
    print("Loaded mapper")

    mean_latent = torch.load("init_latent.pt")[0].cuda()

    tar_z_codes = torch.randn(args.n_samples, 1, 512).cuda()

    with torch.no_grad():
        tar_imgs, latent_code_tars, _ = net.decoder(
            [tar_z_codes], return_latents=True, truncation=0.7, truncation_latent=mean_latent.unsqueeze(0)
        )
    img_gens = []

    id_loss = IDLoss(args)
    x = tar_imgs
    with torch.no_grad():
        embeds = id_loss.extract_feats(x)

    avg_latent = mean_latent.repeat(embeds.shape[0], 1, 1)
    with torch.no_grad():
        w_hat = avg_latent + net.mapper(embeds)

        x_hat, mapper_codes, _ = net.decoder(
            [w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1
        )

    for i in range(args.n_samples):
        tar_img_normalized = tar_imgs[i] / 255.0 if tar_imgs[i].dtype == torch.uint8 else tar_imgs[i]
        x_hat_normalized = x_hat[i] / 255.0 if x_hat[i].dtype == torch.uint8 else x_hat[i]
        # Concatenating the two rows to form a single image
        final_image = torch.cat((tar_img_normalized, x_hat_normalized), 2)  # Concatenate along width

        # Saving the concatenated image
        torchvision.utils.save_image(
            (final_image + 1) / 2, value_range=(0, 1), fp=f"{args.save_dir}/final_image_{i}.png"
        )


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="StyleCLIP Mapper and Optimizer Options")

    # Mapper options
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./pretrained_models/none_identical_clone.pt",
        help="Path to mapper checkpoint",
    )
    parser.add_argument("--n_samples", type=int, default=4, help="Number of samples")
    parser.add_argument("--work_in_stylespace", action="store_true", help="Work in StyleSpace")
    parser.add_argument("--couple_outputs", action="store_true", help="Couple outputs")
    parser.add_argument("--mapper_type", type=str, default="EmbedMapper", help="Type of mapper")
    parser.add_argument("--no_coarse_mapper", action="store_true", help="Exclude coarse mapper")
    parser.add_argument("--no_medium_mapper", action="store_true", help="Exclude medium mapper")
    parser.add_argument("--no_fine_mapper", action="store_true", help="Exclude fine mapper")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--test_batch_size", type=int, default=4, help="Batch size for testing")
    parser.add_argument("--test_workers", type=int, default=4, help="Number of workers for testing")
    parser.add_argument("--stylespace", action="store_true", help="Use StyleSpace")
    parser.add_argument(
        "--ir_se50_weights", type=str, default="pretrained_models/model_ir_se50.pth", help="Path to IR-SE50 weights"
    )
    parser.add_argument("--save_dir", type=str, default="fa_results", help="Path to save results")

    args = parser.parse_args()

    main(args)
