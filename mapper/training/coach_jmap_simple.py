import os

import clip
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import criteria.clip_loss as clip_loss
from criteria import id_loss
from criteria import landmark_loss
from mapper.datasets.latents_dataset import LatentsDataset, StyleSpaceLatentsDataset, LatentsDatasets, LatentsMappedDatasets
from mapper.datasets.mask_dataset import MaskDataset
from mapper.styleclip_mapper import StyleCLIPMapper
from mapper.training.ranger import Ranger
from mapper.training import train_utils
from mapper.training.train_utils import convert_s_tensor_to_list
from torchvision import transforms

class Coach:
    def __init__(self, opts):
        self.opts = opts

        self.global_step = 0

        self.device = 'cuda:0'
        self.opts.device = self.device

        # Initialize network
        self.net = StyleCLIPMapper(self.opts).to(self.device)
        print(self.net.mapper)

        # Initialize loss
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss_t(self.opts).to(self.device).eval()
        if self.opts.latent_l2_lambda > 0:
            self.latent_l2_loss = nn.MSELoss().to(self.device).eval()

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets_t()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)

        #self.text_inputs = torch.cat([clip.tokenize(self.opts.description)]).cuda()

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

    def train(self):
        self.net.train()
        print('w_truth = self.net.mapper([w,w_t])')
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                w,w_t,w_truth = batch
                print(f'w shape: {w.shape}')
                w,w_t,w_truth = w.to(self.device), w_t.to(self.device), w_truth.to(self.device)
                with torch.no_grad():
                    x, _ = self.net.decoder([w],   input_is_latent=True, randomize_noise=False, truncation=1, input_is_stylespace=self.opts.work_in_stylespace)
                #w_hat = w + 0.1 * self.net.mapper( torch.cat([w,w_t], dim=2) ) 
                alpha = self.net.mapper( w,w_t ) 
                w_hat = (1 - alpha)* w + alpha * w_t
                print(f'w_hat shape {w_hat.shape}')
                x_hat, _, _ = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
                loss, loss_dict = self.calc_loss_t( x_hat, x, w_hat, w_truth )
                loss.backward()
                self.optimizer.step()

                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 1000 == 0):
                    with torch.no_grad():
                        x, _ = self.net.decoder([w],   input_is_latent=True, randomize_noise=False, 
                                                truncation=1, input_is_stylespace=self.opts.work_in_stylespace)
                        y, _ = self.net.decoder([w_t], input_is_latent=True, randomize_noise=False, 
                                                truncation=1, input_is_stylespace=self.opts.work_in_stylespace)
                        x_hat, _, _ = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, 
                                                           randomize_noise=False, truncation=1)
                    self.parse_and_log_images(x,y, x_hat, title='images_train')
                    #print(alpha[0][5][:10])
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            if batch_idx > 200:
                break

            w,w_t,w_truth = batch
            w,w_t,w_truth = w.to(self.device), w_t.to(self.device), w_truth.to(self.device)

            with torch.no_grad():
                x, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=False, 
                                        truncation=1, input_is_stylespace=self.opts.work_in_stylespace)
                y, _ = self.net.decoder([w_t], input_is_latent=True, randomize_noise=False, 
                                        truncation=1, input_is_stylespace=self.opts.work_in_stylespace)
                
                alpha = self.net.mapper( w,w_t ) 
                w_hat = (1 - alpha)* w + alpha * w_t
                
                x_hat, _ , _ = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, 
                                                   randomize_noise=False, truncation=1)
                loss, cur_loss_dict = self.calc_loss_t(x_hat, x, w_hat, w_truth)
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            
            self.parse_and_log_images(x, y, x_hat, title='images_val', index=batch_idx)

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        params = list(self.net.mapper.parameters())
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer


    def configure_datasets_t(self):
        
        full_dataset = LatentsMappedDatasets(src_root=self.opts.latents_train_path,
                                       tar_root=self.opts.latents_tar_path,
                                       truth_root = self.opts.latents_truth_path,
                                       opts=self.opts)
        print('full dataset size: ', len(full_dataset))
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [int(len(full_dataset)*0.9), int(len(full_dataset)*0.1)])
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset
    
    def calc_loss_t(self, x_hat, x, w_hat, w_truth):
        loss_dict = {}
        loss = 0.0
        if self.opts.id_lambda > 0:
            loss_id, dist_improvement = self.id_loss(x_hat, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_dist'] = float(dist_improvement)
            loss = loss_id * self.opts.id_lambda
        if self.opts.latent_l2_lambda > 0:
            loss_l2_latent = self.latent_l2_loss(w_hat, w_truth)
            loss_dict['loss_l2_latent'] = float(loss_l2_latent)
            loss += loss_l2_latent * self.opts.latent_l2_lambda
        loss_dict['loss'] = float(loss)
        return loss, loss_dict

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            #pass
            print(f"step: {self.global_step} \t metric: {prefix}/{key} \t value: {value}")
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, x, y, x_hat, title, index=None):
        if index is None:
            path = os.path.join(self.log_dir, title, 
                                f'{str(self.global_step).zfill(5)}.jpg')
        else:
            path = os.path.join(self.log_dir, title, 
                                f'{str(self.global_step).zfill(5)}_{str(index).zfill(5)}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        p = transforms.Compose([transforms.Resize((256,256))])
        log_img = torch.cat([p(x).detach().cpu(),p(y).detach().cpu(), 
                             p(x_hat).detach().cpu()])
        torchvision.utils.save_image(p(log_img), path,
                                     normalize=True, scale_each=True, 
                                     range=(-1, 1), nrow=self.opts.batch_size)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts)
        }
        return save_dict