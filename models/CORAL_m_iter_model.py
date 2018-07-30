import torch
from torch.autograd import Variable
import itertools
from .base_model import BaseModel
from . import networks
import torch.nn as nn


class CORAL_m_iter_Model(BaseModel):
    def name(self):
        return 'CORAL_m_iter_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['Domain', 'C']
        
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks

        self.model_names = ['SE']
        self.load_model_names_DA = ['SE']
        self.load_model_names_Di = ['ZE', 'De']
        self.load_model_names_Pre = ['SE']

        # load/define networks
        self.netSEPre = networks.define_SE(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.which_method, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netSE = networks.define_SE(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.which_method, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netZE = networks.define_ZE(opt.input_nc, opt.output_nc, opt.which_model_netZE, opt.norm,
                                        not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netAC = networks.define_AC(opt.output_nc, opt.input_nc, opt.which_model_netAC, opt.norm, not opt.no_dropout,
                                        opt.init_type, self.gpu_ids)
        self.netDe = networks.define_De(opt.output_nc, opt.input_nc, opt.which_model_netDe, opt.which_method, opt.norm,
                                        not opt.no_dropout, opt.init_type, self.gpu_ids)

        self.load_networks_DA(opt.which_usename_DA, opt.which_epochs_DA)
        self.load_networks_Di(opt.which_usename_Di, opt.which_epochs_Di)
        self.load_networks_Pre(opt.which_usename_DA, opt.which_epochs_DA)
        self.nets = [self.netSE]
        



        if self.isTrain:
                                                
            # initialize optimizers
            self.optimizer = torch.optim.SGD([
        {'params': self.netSE.module.sharedNet.parameters()},
        {'params': self.netSE.module.fc.parameters(), 'lr': 10*opt.lr/opt.lr_de},
], lr=opt.lr/opt.lr_de, momentum=0.9)

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer)
            
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    def set_input(self, opt, s_img, s_label, t_img):

        
        if len(self.gpu_ids) > 0:
            s_img = s_img.cuda(self.gpu_ids[0], async=True)
            s_label = s_label.cuda(self.gpu_ids[0], async=True).view(-1)
            t_img = t_img.cuda(self.gpu_ids[0], async=True)
                        
        self.s_img = Variable(s_img)
        self.t_img = Variable(t_img)        
        self.s_label = Variable(s_label)
        

        s_img1, s_img2 = torch.split(s_img, s_img.shape[0]/2, 0)
        E_s1, E_s2 = self.netSEPre(s_img1, s_img2)
        self.mid_s = torch.cat((E_s1, E_s2), 0)
        self.mid_z = self.netZE(self.t_img)

        self.mid_code = torch.cat((self.mid_s, self.mid_z), 1)#.contiguous()
        self.mid_img = Variable(self.netDe(self.mid_code).view(-1, 3, 28, 28).data)
        self.mid_label = self.s_label

        self.s_mid_label = torch.cat((self.s_label, self.mid_label), 0)
        self.s_mid_img = torch.cat((self.s_img, self.mid_img), 0)

        self.img_G = torch.cat((self.s_img, self.mid_img, self.t_img), 0)

        return self.img_G

    def forward_CORAL(self):
        self.E_s_mid, self.E_t = self.netSE(self.s_mid_img, self.t_img)

    def backward_CORAL(self, _lambda):
        C_loss = torch.nn.functional.cross_entropy(self.E_s_mid, self.s_mid_label)
        Domain_loss = networks.CORAL_Loss(self.E_s_mid, self.E_t)
        sum_loss = _lambda * Domain_loss + C_loss
        sum_loss.backward()
        return C_loss.data[0], Domain_loss.data[0], sum_loss.data[0]
    
    def optimize_parameters(self, _lambda):
        
        self.forward_CORAL()                
        C_loss, Domain_loss, sum_loss = self.backward_CORAL(_lambda)        
        self.optimizer.step()
        self.optimizer.zero_grad()

        return C_loss, Domain_loss
        
    def test(self, t_img, t_label):

        if len(self.gpu_ids) > 0:
            t_img = t_img.cuda(self.gpu_ids[0], async=True)
            t_label = t_label.cuda(self.gpu_ids[0], async=True).view(-1)
                                           
        t_img = Variable(t_img)        
        t_label = Variable(t_label)

        t_img1, t_img2 = torch.split(t_img, t_img.shape[0]/2, 0)
        E_t1, E_t2 = self.netSE(t_img1, t_img2)
        E_t = torch.cat((E_t1, E_t2), 0)
        
        pred = E_t.data.max(1)[1]
        correct_t = pred.eq(t_label.data).cpu().sum()

        return correct_t
        
