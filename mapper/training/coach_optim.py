import os

import clip
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import criteria.clip_loss as clip_loss
from criteria.clip_loss import CLIPLoss
from criteria import id_loss
from criteria import landmark_loss
from mapper.datasets.latents_dataset import LatentsDataset, StyleSpaceLatentsDataset, LatentsDatasets
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
        self.direction = torch.rand(18,512).to(self.device)
        self.direction.requires_grad = True
        print(self.direction)

        # Initialize loss
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss_t(self.opts).to(self.device).eval()
        if self.opts.clip_lambda > 0:
            self.clip_loss = torch.nn.DataParallel(clip_loss.CLIPLoss(opts))
        if self.opts.latent_l2_lambda > 0:
            self.latent_l2_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.landmark_l2_lambda > 0:
            self.landloss = landmark_loss.LMLoss(self.opts).to(self.device).eval()

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

        self.text_inputs = torch.cat([clip.tokenize(self.opts.description)]).cuda()

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
        self.net.eval()
        print('w_hat = w + self.net.mapper( torch.cat([w,w_t], dim=2) )')
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                if self.opts.work_in_stylespace:
                    w = convert_s_tensor_to_list(batch)
                    w = [c.to(self.device) for c in w]
                else:
                    w,w_t = batch
                    w,w_t = w.to(self.device), w_t.to(self.device)
                with torch.no_grad():
                    x, _ = self.net.decoder([w],   input_is_latent=True, randomize_noise=False, truncation=1, input_is_stylespace=self.opts.work_in_stylespace)
                    y, _ = self.net.decoder([w_t], input_is_latent=True, randomize_noise=False, truncation=1, input_is_stylespace=self.opts.work_in_stylespace)
                if self.opts.work_in_stylespace:
                    delta = self.net.mapper(w)
                    w_hat = [c + 0.1 * delta_c for (c, delta_c) in zip(w, delta)]
                    x_hat, _, w_hat = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1, input_is_stylespace=True)
                else:
                    # input_ = torch.cat([w,w_t], dim=2)
                    # print(input_.shape)
                    # mask = (self.direction > 0.5).float()
                    #mask = self.direction
                    mask = torch.sigmoid(self.direction)
                    w_hat_0 = 0.001 * w * mask + (1- 0.001* mask) * w_t
                    x_hat, w_hat, _ = self.net.decoder([w_hat_0], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
                    #print('Comparing w_hat ...', torch.eq(w_hat, w_hat_0))
                    
                loss, loss_dict = self.calc_loss_t(w, x, y, w_hat, x_hat)
                loss.backward()
                self.optimizer.step()
                # print('update gradient')
                # print(self.direction)

                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 1000 == 0):
                    self.parse_and_log_images(x,y, x_hat, title='images_train')
                    print(self.direction)
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

            if self.opts.work_in_stylespace:
                w = convert_s_tensor_to_list(batch)
                w = [c.to(self.device) for c in w]
            else:
                w,w_t = batch
                w,w_t = w.to(self.device), w_t.to(self.device)

            with torch.no_grad():
                x, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1, input_is_stylespace=self.opts.work_in_stylespace)
                y, _ = self.net.decoder([w_t], input_is_latent=True, randomize_noise=False, truncation=1, input_is_stylespace=self.opts.work_in_stylespace)
                if self.opts.work_in_stylespace:
                    delta = self.net.mapper(w)
                    w_hat = [c + 0.1 * delta_c for (c, delta_c) in zip(w, delta)]
                    x_hat, _, w_hat = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1, input_is_stylespace=True)
                else:
                    mask = torch.sigmoid(self.direction)
                    w_hat = w * mask + (1-mask) * w_t
                    x_hat, w_hat, _ = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
                loss, cur_loss_dict = self.calc_loss_t(w, x, y, w_hat, x_hat)
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            self.parse_and_log_images(x, y, x_hat, title='images_val', index=batch_idx)

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.net.eval()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.eval()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.direction
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        #params = list(self.net.mapper.parameters())
        
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam([self.direction], lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        if self.opts.latents_train_path:
            train_latents = torch.load(self.opts.latents_train_path) #load latents
        else:
            train_latents_z = torch.randn(self.opts.train_dataset_size, 512).cuda() #random sampling in z
            train_latents = []
            for b in range(self.opts.train_dataset_size // self.opts.batch_size):
                with torch.no_grad():
                    _, train_latents_b = self.net.decoder([train_latents_z[b: b + self.opts.batch_size]],
                                                          truncation=0.7, truncation_latent=self.net.latent_avg, return_latents=True) #generate 
                    train_latents.append(train_latents_b) #colect w latent codes
            train_latents = torch.cat(train_latents) #concatenate all

        if self.opts.latents_test_path:
            test_latents = torch.load(self.opts.latents_test_path)
        else:
            test_latents_z = torch.randn(self.opts.train_dataset_size, 512).cuda()
            test_latents = []
            for b in range(self.opts.test_dataset_size // self.opts.test_batch_size):
                with torch.no_grad():
                    _, test_latents_b = self.net.decoder([test_latents_z[b: b + self.opts.test_batch_size]],
                                                      truncation=0.7, truncation_latent=self.net.latent_avg, return_latents=True)
                    test_latents.append(test_latents_b)
            test_latents = torch.cat(test_latents)

        if self.opts.work_in_stylespace:
            train_dataset_celeba = StyleSpaceLatentsDataset(latents=[l.cpu() for l in train_latents],
                                                            opts=self.opts)
            test_dataset_celeba = StyleSpaceLatentsDataset(latents=[l.cpu() for l in test_latents],
                                                 opts=self.opts)
        else:
            train_dataset_celeba = LatentsDataset(latents=train_latents.cpu(),
                                                  opts=self.opts)
            test_dataset_celeba = LatentsDataset(latents=test_latents.cpu(),
                                                  opts=self.opts)
        train_dataset = train_dataset_celeba
        test_dataset = test_dataset_celeba
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset

    def configure_datasets_t(self):
        if self.opts.latents_src_path:
            src_latents = torch.load(self.opts.latents_src_path)
            tar_latents = torch.load(self.opts.latents_tar_path)
            train_src_latents = src_latents[:int(src_latents.shape[0]*0.9)]
            train_tar_latents = tar_latents[:int(tar_latents.shape[0]*0.9)]
        else:
            train_latents_z = torch.randn(self.opts.train_dataset_size, 512).cuda() #random sampling in z
            train_latents = []
            for b in range(self.opts.train_dataset_size // self.opts.batch_size):
                with torch.no_grad():
                    _, train_latents_b = self.net.decoder([train_latents_z[b: b + self.opts.batch_size]],
                                                          truncation=0.7, truncation_latent=self.net.latent_avg, return_latents=True) #generate 
                    train_latents.append(train_latents_b) #colect w latent codes
            train_latents = torch.cat(train_latents) #concatenate all

        if self.opts.latents_src_path:
            test_src_latents = src_latents[:int(src_latents.shape[0]*0.1)]
            test_tar_latents = tar_latents[:int(tar_latents.shape[0]*0.1)]
        else:
            test_latents_z = torch.randn(self.opts.train_dataset_size, 512).cuda()
            test_latents = []
            for b in range(self.opts.test_dataset_size // self.opts.test_batch_size):
                with torch.no_grad():
                    _, test_latents_b = self.net.decoder([test_latents_z[b: b + self.opts.test_batch_size]],
                                                      truncation=0.7, truncation_latent=self.net.latent_avg, return_latents=True)
                    test_latents.append(test_latents_b)
            test_latents = torch.cat(test_latents)

        if self.opts.work_in_stylespace:
            train_dataset_celeba = StyleSpaceLatentsDataset(latents=[l.cpu() for l in train_latents],
                                                            opts=self.opts)
            test_dataset_celeba = StyleSpaceLatentsDataset(latents=[l.cpu() for l in test_latents],
                                                 opts=self.opts)
        else:
            train_dataset_celeba = LatentsDatasets(src_latents=train_src_latents.cpu(),
                                                  tar_latents=train_tar_latents.cpu(),
                                                  opts=self.opts)
            test_dataset_celeba = LatentsDatasets(src_latents=test_src_latents.cpu(),
                                                 tar_latents=test_tar_latents.cpu(),
                                                  opts=self.opts)
        train_dataset = train_dataset_celeba
        test_dataset = test_dataset_celeba
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset
    
    def calc_loss(self, w, x, w_hat, x_hat):
        loss_dict = {}
        loss = 0.0
        if self.opts.id_lambda > 0:
            loss_id, sim_improvement = self.id_loss(x_hat, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda
        if self.opts.landmark_l2_lambda > 0:
            loss_landmark = self.landloss(x_hat, x)
            loss_dict['loss_l2_landmark'] = float(loss_landmark)
            loss += loss_landmark * self.opts.landmark_l2_lambda
        if self.opts.clip_lambda > 0:
            loss_clip = self.clip_loss(x_hat, self.text_inputs).mean()
            loss_dict['loss_clip'] = float(loss_clip)
            loss += loss_clip * self.opts.clip_lambda
        if self.opts.latent_l2_lambda > 0:
            if self.opts.work_in_stylespace:
                loss_l2_latent = 0
                for c_hat, c in zip(w_hat, w):
                    loss_l2_latent += self.latent_l2_loss(c_hat, c)
            else:
                loss_l2_latent = self.latent_l2_loss(w_hat, w)
            loss_dict['loss_l2_latent'] = float(loss_l2_latent)
            loss += loss_l2_latent * self.opts.latent_l2_lambda
        loss_dict['loss'] = float(loss)
        return loss, loss_dict
    
    def calc_loss_t(self, w, x, y, w_hat, x_hat):
        loss_dict = {}
        loss = 0.0
        if self.opts.id_lambda > 0:
            loss_id, sim_improvement = self.id_loss(x_hat, x, y)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda
        if self.opts.landmark_l2_lambda > 0:
            loss_landmark = self.landloss(x_hat, x)
            loss_dict['loss_l2_landmark'] = float(loss_landmark)
            loss += loss_landmark * self.opts.landmark_l2_lambda
        if self.opts.clip_lambda > 0:
            #loss_clip = self.clip_loss(x_hat, self.text_inputs).mean()
            target_clip = self.clip_loss.module.encode(y)
            target_clip = target_clip / target_clip.norm(dim=-1, keepdim=True)
            mod_clip = self.clip_loss.module.encode(x_hat)
            mod_clip = mod_clip / mod_clip.norm(dim=-1, keepdim=True)
            similarity_gap = mod_clip @ target_clip.T #matmul
            loss_clip = 1 - similarity_gap.mean()
            loss_dict['loss_clip'] = float(loss_clip)
            loss += loss_clip * self.opts.clip_lambda
        if self.opts.latent_l2_lambda > 0:
            if self.opts.work_in_stylespace:
                loss_l2_latent = 0
                for c_hat, c in zip(w_hat, w):
                    loss_l2_latent += self.latent_l2_loss(c_hat, c)
            else:
                loss_l2_latent = self.latent_l2_loss(w_hat, w)
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
            path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}.jpg')
        else:
            path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}_{str(index).zfill(5)}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        p = transforms.Compose([transforms.Resize((256,256))])
        log_img = torch.cat([p(x).detach().cpu(),p(y).detach().cpu(), p(x_hat).detach().cpu()])
        torchvision.utils.save_image(p(log_img), path,
                                     normalize=True, scale_each=True, range=(-1, 1), nrow=self.opts.batch_size)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts)
        }
        return save_dict
    
    def get_lr(self, t, initial_lr, rampdown=0.75, rampup=0.005):
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)

        return initial_lr * lr_ramp