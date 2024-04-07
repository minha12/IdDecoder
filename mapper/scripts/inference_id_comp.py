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

from mapper.datasets.latents_dataset import EmbedNamesDataset

from mapper.options.test_options import TestOptions
from mapper.styleclip_mapper import StyleCLIPMapper
from torchvision import transforms

def run(test_opts, file_names):
    print('number of file_names: {}'.format(len(file_names)))
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

    dataset = EmbedNamesDataset(embed_names=file_names,
                                root=opts.latents_test_path)
    #print('root: ', opts.latents_test_path)
    #print("Number of training samples: {}".format(len(train_dataset)))
    print("Number of test samples: {}".format(len(dataset)))
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
            result_batch = run_on_batch(input_cuda, net)
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(opts.test_batch_size):
            #index_i = test_dataset.indices[global_i]
            im_path = os.path.basename(dataset.embed_paths[global_i]).split('.')[0]
            #print('saving ', im_path)
            #print('saving ', im_path.split('/')[-2] + '.png')
            torchvision.utils.save_image(result_batch[0][i], os.path.join(out_path_results, f"{im_path}.png"), normalize=True, range=(-1, 1))
            #torch.save(result_batch[1][i].detach().cpu(), os.path.join(out_path_results, f"latent_{im_path}.pt"))

            global_i += 1

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)

def run_on_batch(inputs, net):
    embeds = inputs
    embeds = embeds.to('cuda')
    with torch.no_grad():
        #w_hat = (1 - net.mapper( w,w_t ) )* w + net.mapper( w,w_t ) * w_t
        w_hat = net.mapper( embeds )

        x_hat, w_hat, _ = net.decoder([w_hat], input_is_latent=True, return_latents=True, 
                                           randomize_noise=False, truncation=1)
        result_batch = (x_hat, w_hat)
    return result_batch

if __name__ == '__main__':
	test_opts = TestOptions().parse()
	run(test_opts)
