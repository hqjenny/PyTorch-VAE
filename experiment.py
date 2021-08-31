import math
import torch
import cosa_dataset
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

import time
import traceback
import argparse
 
class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict,
                 hparams: argparse.Namespace) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.hparams = hparams
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        M_N = self.params['batch_size']/ self.num_train_imgs
        outp, inp, mu, _ = results 

        print(f'train_input, {inp}')
        print(f'train_output, {outp}')
        print(f'train_latent, {mu}')

        train_loss = self.model.loss_function(*results,
                                              M_N = 0.001,
                                              optimizer_idx=optimizer_idx,
                                              labels = labels, 
                                              batch_idx = batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        # self.curr_device = real_img.device

        # simba_final_modified row in arch_dataset_12.csv
        # 4121961,5571578486.230001,16,1024,1024,1,128,64,128,16384,64,1024,1,65536

        # exit(0)

        results = self.forward(real_img, labels = labels)
        outp, inp, mu, _ = results 

        print(f'val_input, {inp}')
        print(f'val_output, {outp}')
        print(f'val_latent, {mu}')
        #M_N = self.params['batch_size']/ self.num_val_imgs,
        val_loss = self.model.loss_function(*results,
                                            M_N = 0.001,
                                            optimizer_idx = optimizer_idx,
                                            labels = labels, 
                                            batch_idx = batch_idx)

        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        # self.curr_device = real_img.device

        # simba_final_modified row in arch_dataset_12.csv
        # 4121961,5571578486.230001,16,1024,1024,1,128,64,128,16384,64,1024,1,65536

        # exit(0)

        print(f'real_img, {batch}')
        results = self.forward(real_img, labels = labels)
        print(f'latent vector, {results}')
        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            labels = labels, 
                                            batch_idx = batch_idx)

        return val_loss

    def test_end(self, outputs):
        # real_img_manual = torch.tensor([
        #                                 [16, 512, 512, 1, 128, 64, 128, 16384, 64, 1024, 1, 65536],
        #                                 [16, 1024, 1024, 1, 128, 64, 128, 16384, 64, 1024, 1, 65536],
        #                                 [16, 2048, 2048, 1, 128, 64, 128, 16384, 64, 1024, 1, 65536],
        #                                 [16, 4096, 4096, 1, 128, 64, 128, 16384, 64, 1024, 1, 65536]  
        #                                ],
        #                                dtype=torch.float64)
        # labels_manual = torch.tensor([
        #                                 4114280,
        #                                 4121961,
        #                                 4114280,
        #                                 4120834
        #                              ])

        # Test encoding 
        real_img_manual = torch.tensor([
                                        # [16, 1024, 1024, 1, 128, 64, 128, 16384, 64, 1024, 1, 262144]
                                        [16, 4096, 4096, 64, 16384, 1024, 65536]
                                       ],
                                       dtype=torch.float64)
        labels_manual = torch.tensor([
                                        3607319
                                     ])
        # result_manual = self.model.encode(real_img_manual)
        result_manual = self.forward(real_img_manual, labels=labels_manual)
        mu = result_manual[2]
        log_var = result_manual[3]
        z = self.model.reparameterize(mu, log_var)

        print("result_manual:")
        print(result_manual)

        print("z:")
        print(z)

        decoded = self.model.decode(z)
        print("decoded:")
        print(decoded)

        exit(0)

        # Test decoding
        # z = torch.tensor([[0.1, -0.9964, -0.8327, -0.3403]]).double()
        # z = torch.tensor([[2.4483,  2.7174,  2.4963, -0.6122]]).double()
        # z = torch.tensor([[ 1.2228,  3.1153,  2.7553, -0.2200]]).double()

        from mpl_toolkits import mplot3d
        import numpy as np
        import matplotlib.pyplot as plt

        input_name = "entries_globalbuf"
        input_idx = 6

        # L1 = [1, 1.1, 1.2, 1.3, 1.4]
        # L2 = [2.9, 3, 3.1, 3.2, 3.3]
        L1 = np.linspace(-4, 4, 80)
        L2 = np.linspace(-4, 4, 80)
        X, Y = np.meshgrid(L1, L2)
        Z = np.zeros(X.shape)

        # [16, 1024, 1024, 64, 16384, 1024, 65536] -> [0.1280, -0.4072,  1.3486,  1.1342]
        for yidx in range(X.shape[0]):
            for xidx in range(X.shape[1]):
                z = torch.tensor([[ X[yidx][xidx],  Y[yidx][xidx], 1.3486, 1.1342]]).double()
                decoded = self.model.decode(z)

                arch_val = decoded[0][input_idx]
                Z[yidx][xidx] = arch_val

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(np.array(L1), np.array(L2), np.array(Z), 80, cmap='binary')
        ax.set_xlabel('l1')
        ax.set_ylabel('l2')
        ax.set_zlabel(input_name)

        ax.view_init(35, 35)
        plt.savefig('contour_l1_l2.png', bbox_inches='tight')

        L3 = np.linspace(-3, 5, 80)
        L4 = np.linspace(-3, 5, 80)
        X, Y = np.meshgrid(L3, L4)
        Z = np.zeros(X.shape)

        # [16, 1024, 1024, 64, 16384, 1024, 65536] -> [0.1280, -0.4072,  1.3486,  1.1342]
        for yidx in range(X.shape[0]):
            for xidx in range(X.shape[1]):
                z = torch.tensor([[ 0.1280, -0.4072, X[yidx][xidx], Y[yidx][xidx]]]).double()
                decoded = self.model.decode(z)

                arch_val = decoded[0][input_idx]
                Z[yidx][xidx] = arch_val

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(np.array(L3), np.array(L4), np.array(Z), 80, cmap='binary')
        ax.set_xlabel('l3')
        ax.set_ylabel('l4')
        ax.set_zlabel(input_name)

        # ax.view_init(35, 35)
        plt.savefig('contour_l3_l4.png', bbox_inches='tight')

        # exit(0)

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    # def test_step(self, batch, batch_idx, optimizer_idx = 0):
    #     pass
    
    # def test_end(self, outputs):
    #     z = torch.tensor([[2.5, -1.8951, -4.8044,  0.3893]]).double()
    #     decoded = self.model.decode(z)
    #     print("decoded:")
    #     print(decoded)

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))

        # print("test_input:")
        # print(test_input)
        # print("test_label:")
        # print(test_label)
        # exit(0)

        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        recons = self.model.generate(test_input, labels = test_label)
        fn =  f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/recons_{self.logger.name}_{self.current_epoch}.csv"
        fn_input =  f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/input_{self.logger.name}_{self.current_epoch}.csv"

        with open(fn, 'w') as f:
            result = recons.data.tolist()
            f.write(f'{result}\n')


        with open(fn_input, 'w') as f:
            result =test_input.data.tolist()
            f.write(f'{result}\n')



        # Vutils.save_image(recons.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"recons_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        try:
            samples = self.model.sample(16,
                                        self.curr_device,
                                        labels = test_label.double())

            # vutils.save_image(samples.cpu().data,
            #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
            #                   f"{self.logger.name}_{self.current_epoch}.png",
            #                   normalize=True,
            #                   nrow=12)
            fn =  f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/{self.logger.name}_{self.current_epoch}.csv"

            with open(fn, 'w') as f:
                result = samples.cpu().data.tolist()
                f.write(f'{result}\n')

        except Exception as e:
            # print(traceback.format_exc())
            pass

        # exit(0)

        del test_input, recons #, samples


    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        if self.params['dataset'] == 'celeba':
            transform = self.data_transforms()
            dataset = CelebA(root = self.params['data_path'],
                             split = "train",
                             transform=transform,
                             download=False)	
        elif self.params['dataset'] == 'cosa':
            dataset = cosa_dataset.CoSADataset(root = self.params['data_path'], split= "train")
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size= self.params['batch_size'],
                          shuffle = True,
                          drop_last=True)

    @data_loader
    def val_dataloader(self):

        if self.params['dataset'] == 'celeba':
            transform = self.data_transforms()
            dataset = CelebA(root = self.params['data_path'], split = "test", transform=transform,
						    download=False)
        elif self.params['dataset'] == 'cosa':
            dataset = cosa_dataset.CoSADataset(root = self.params['data_path'], split= "valid")

        else:
            raise ValueError('Undefined dataset type')
        
        
        self.sample_dataloader =  DataLoader(dataset,
					     batch_size= 4,
					     shuffle = True,
					     drop_last=True)
        self.num_val_imgs = len(self.sample_dataloader)

        return self.sample_dataloader

    @data_loader
    def test_dataloader(self):

        if self.params['dataset'] == 'celeba':
            transform = self.data_transforms()
            dataset = CelebA(root = self.params['data_path'], split = "test", transform=transform,
						    download=False)
        elif self.params['dataset'] == 'cosa':
            dataset = cosa_dataset.CoSADataset(root = self.params['data_path'], split= "test")

        else:
            raise ValueError('Undefined dataset type')
        
        
        self.sample_dataloader =  DataLoader(dataset,
					     batch_size= 4,
					     shuffle = True,
					     drop_last=True)
        self.num_val_imgs = len(self.sample_dataloader)

        return self.sample_dataloader

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        else:
            raise ValueError('Undefined dataset type')
        return transform

