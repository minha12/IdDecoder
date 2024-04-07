import os
from argparse import Namespace

import torchvision
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import time

from tqdm import tqdm

from mapper.training.train_utils import convert_s_tensor_to_list

sys.path.append(".")
sys.path.append("..")

from mapper.datasets.latents_dataset import LatentsDatasets, StyleSpaceLatentsDataset

from mapper.options.test_options import TestOptions
from mapper.styleclip_mapper import StyleCLIPMapper
from torchvision import transforms


def run(test_opts):
    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    os.makedirs(out_path_results, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)
    # print(opts)
    net = StyleCLIPMapper(opts)
    net.eval()
    net.cuda()

    test_src_latents = torch.load(opts.latents_test_path)
    vec_to_inject = np.random.randn(test_src_latents.shape[0], 512).astype('float32')
    print(vec_to_inject.shape)
    test_tar_latents = torch.tensor([]).cuda()
    #print(torch.from_numpy(vec_to_inject[0]).shape)
    for i in range(test_src_latents.shape[0]):
        _, latent_code = net(torch.from_numpy(vec_to_inject[i]).to("cuda").unsqueeze(0), input_code=True,return_latents=True,randomize_noise=False)
        test_tar_latents = torch.cat( (test_tar_latents, latent_code.detach()), dim=0)
    print(f'target latent shape {test_tar_latents.shape}')
    if opts.work_in_stylespace:
        dataset = StyleSpaceLatentsDataset(latents=[l.cpu() for l in test_latents], opts=opts)
    else:
        dataset = LatentsDatasets(src_latents=test_src_latents.cpu(),
                                  tar_latents=test_tar_latents.cpu(),
                                  mask_root=opts.mask_dataset_path,
                                  opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)
    
    global_i = 0
    global_time = []
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            input_cuda = input_batch
            

            tic = time.time()
            result_batch = run_on_batch(input_cuda, net, opts.couple_outputs, opts.work_in_stylespace)
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(opts.test_batch_size):
            im_path = str(global_i).zfill(5)
            if test_opts.couple_outputs:
                couple_output = torch.cat([result_batch[2][i].unsqueeze(0), result_batch[0][i].unsqueeze(0)])
                torchvision.utils.save_image(couple_output, os.path.join(out_path_results, f"{im_path}.jpg"), normalize=True, range=(-1, 1))
            else:
                torchvision.utils.save_image(result_batch[0][i], os.path.join(out_path_results, f"{im_path}.jpg"), normalize=True, range=(-1, 1))
            torch.save(result_batch[1][i].detach().cpu(), os.path.join(out_path_results, f"latent_{im_path}.pt"))

            global_i += 1

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)

from editings import latent_editor

ganspace_pca = torch.load('editings/ganspace_pca/ffhq_pca.pt')
directions = {
    'asianess': (16,5,8,-3),
    'eye_openness': (16,7,8,0)
}
editor = latent_editor.LatentEditor()
def run_on_batch(inputs, net, couple_outputs=False, stylespace=False):
    w,w_t,mask = inputs
    w,w_t = w.to('cuda'), w_t.to('cuda')
    with torch.no_grad():
        #w_hat = (1 - net.mapper( w,w_t ) )* w + net.mapper( w,w_t ) * w_t
        w_hat = w + 0.1 * net.mapper( w )
        w_hat = editor.apply_ganspace(w_hat, ganspace_pca, [directions["asianess"]])
        w_hat = editor.apply_ganspace(w_hat, ganspace_pca, [directions["eye_openness"]])
        x_hat, w_hat, _ = net.decoder([w_hat], input_is_latent=True, return_latents=True, 
                                           randomize_noise=False, truncation=1)
        result_batch = (x_hat, w_hat)
        if couple_outputs:
            x, _ = net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1, input_is_stylespace=stylespace)
            result_batch = (x_hat, w_hat, x)
    return result_batch


if __name__ == '__main__':
	test_opts = TestOptions().parse()
	run(test_opts)
