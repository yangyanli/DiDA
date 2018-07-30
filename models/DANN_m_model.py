import torch
from torch.autograd import Variable
import itertools
from .base_model import BaseModel
from . import networks
import torch.nn as nn


class DANN_m_Model(BaseModel):
    def name(self):
        return 'DANN_m_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['Domain', 'C']
        
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['SE', 'C', 'DP']
        else:  # during test time, only load Gs
            self.model_names = ['SE', 'C']

        # load/define networks                
        self.netSE = networks.define_SE(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netC = networks.define_C(opt.output_nc, opt.input_nc, opt.which_model_netC, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netDP = networks.define_DP(opt.output_nc, opt.input_nc, opt.which_model_netDP, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)



        if self.isTrain:
                                                
            # initialize optimizers
            self.optimizer_C = torch.optim.SGD(itertools.chain(self.netSE.parameters(), self.netC.parameters()), lr=opt.lr, momentum = 0.9)
            self.optimizer_Domain = torch.optim.SGD(itertools.chain(self.netSE.parameters(), self.netDP.parameters()), lr=opt.lr, momentum = 0.9)
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_C)
            self.optimizers.append(self.optimizer_Domain)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    def set_input(self, opt, s_img, s_label, t_img):

        domain_t_label = torch.LongTensor(opt.batchSize/2).zero_() + 1
        #domain_t_label.cuda()
        domain_s_label = torch.LongTensor(opt.batchSize/2).zero_()
        #domain_s_label.cuda()
        domain_label = torch.cat((domain_t_label, domain_s_label), 0)#.cuda()
        
        if len(self.gpu_ids) > 0:
            s_img = s_img.cuda(self.gpu_ids[0], async=True)
            s_label = s_label.cuda(self.gpu_ids[0], async=True).view(-1)
            t_img = t_img.cuda(self.gpu_ids[0], async=True)
            domain_label = domain_label.cuda(self.gpu_ids[0], async=True)
            
        self.s_img = Variable(s_img)
        self.t_img = Variable(t_img)        
        self.s_label = Variable(s_label)
        self.img_G = torch.cat((self.t_img, self.s_img), 0)
        self.domain_label = Variable(domain_label)

    def forward_C(self):
        self.E_s = self.netSE(self.s_img)
        self.pred_label = self.netC(self.E_s)

    def forward_Domain(self):
        self.E_G = self.netSE(self.img_G)
        self.domain_pred = self.netDP(self.E_G)

    def backward_C(self):
        C_loss = torch.nn.functional.cross_entropy(self.pred_label, self.s_label)
        C_loss.backward()
        return C_loss.data[0]

    def backward_Domain(self):
        Domain_loss = torch.nn.functional.cross_entropy(self.domain_pred, self.domain_label)
        Domain_loss.backward()
        return Domain_loss.data[0]

    
    def optimize_parameters(self):
        
        self.forward_C()                
        C_loss = self.backward_C()        
        self.optimizer_C.step()
        self.optimizer_C.zero_grad()

        self.forward_Domain()                
        Domain_loss = self.backward_Domain()        
        self.optimizer_Domain.step()
        self.optimizer_Domain.zero_grad()

        return C_loss, Domain_loss
        
    def test(self, t_img, t_label):

        if len(self.gpu_ids) > 0:
            t_img = t_img.cuda(self.gpu_ids[0], async=True)
            t_label = t_label.cuda(self.gpu_ids[0], async=True).view(-1)
                                           
        t_img = Variable(t_img)        
        t_label = Variable(t_label)

        E_t = self.netSE(t_img)
        pred_label_t = self.netC(E_t)
        
        pred = pred_label_t.data.max(1)[1]
        correct_t = pred.eq(t_label.data).cpu().sum()

        return correct_t
        
