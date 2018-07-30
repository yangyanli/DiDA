import torch
from torch.autograd import Variable
import itertools
from .base_model import BaseModel
from . import networks
import torch.nn as nn


class CORAL_m_Model(BaseModel):
    def name(self):
        return 'CORAL_m_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['Domain', 'C']
        
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['SE']
        else:  # during test time, only load Gs
            self.model_names = ['SE']

        # load/define networks                
        self.netSE = networks.define_SE(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.which_method_netSE, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)


        if self.isTrain:
                                                
            # initialize optimizers
            #self.optimizer_C = torch.optim.SGD(itertools.chain(self.netSE.parameters()), lr=opt.lr, momentum = 0.9)
            self.optimizer = torch.optim.SGD([
        {'params': self.netSE.module.sharedNet.parameters()},
        {'params': self.netSE.module.fc.parameters(), 'lr': 10*opt.lr},
], lr=opt.lr, momentum=0.9)
            
            #self.optimizer = torch.optim.SGD(itertools.chain(self.netSE.parameters()), lr=opt.lr, momentum = 0.9)

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
        #self.img_G = torch.cat((self.t_img, self.s_img), 0)


    def forward_CORAL(self):
        self.E_s, self.E_t = self.netSE(self.s_img, self.t_img)

    def backward_CORAL(self, _lambda):
        C_loss = torch.nn.functional.cross_entropy(self.E_s, self.s_label)
        Domain_loss = networks.CORAL_Loss(self.E_s, self.E_t)
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
        
