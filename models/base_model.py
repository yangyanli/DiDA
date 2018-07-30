import os
import torch
from collections import OrderedDict


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    

    # save models to the disk
    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # load DAmodels from the disk
    def load_networks_DA(self, which_usename_DA, which_epochs_DA):
        which_usename_DA = 'checkpoints/' + which_usename_DA
        for name in self.load_model_names_DA:
            if isinstance(name, str):
                save_filename = 'best%s_net_%s.pth' % (which_epochs_DA, name)

                save_path = os.path.join(which_usename_DA, save_filename)
                net = getattr(self, 'net' + name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.module.load_state_dict(torch.load(save_path))
                else:
                    net.load_state_dict(torch.load(save_path))

    # load Dimodels from the disk
    def load_networks_Di(self, which_usename_Di, which_epochs_Di):
        which_usename_Di = 'checkpoints/' + which_usename_Di
        for name in self.load_model_names_Di:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epochs_Di, name)

                save_path = os.path.join(which_usename_Di, save_filename)
                net = getattr(self, 'net' + name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.module.load_state_dict(torch.load(save_path))
                else:
                    net.load_state_dict(torch.load(save_path))

    # load DAmodels from the disk
    def load_networks_Pre(self, which_usename_DA, which_epochs_DA):
        which_usename_DA = 'checkpoints/' + which_usename_DA
        for name in self.load_model_names_Pre:
            if isinstance(name, str):
                save_filename = 'best%s_net_%s.pth' % (which_epochs_DA, name)

                save_path = os.path.join(which_usename_DA, save_filename)
                net = getattr(self, 'net' + name + 'Pre')
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.module.load_state_dict(torch.load(save_path))
                else:
                    net.load_state_dict(torch.load(save_path))


    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
