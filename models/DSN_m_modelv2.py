import torch
from torch.autograd import Variable
import itertools
from .base_model import BaseModel
from . import networks
import torch.nn as nn


class DSN_m_Modelv2(BaseModel):
    def name(self):
        return 'DSN_m_Modelv2'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['Domain', 'C', 'Diff', 'Recon']
        
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['SE', 'PE_S', 'PE_T', 'C', 'DP', 'Re']
        else:  # during test time, only load Gs
            self.model_names = ['SE', 'PE_S', 'PE_T', 'C', 'Re']

        # load/define networks                
        self.netSE = networks.define_SE(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.which_method, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netPE_S = networks.define_SE(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.which_method, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netPE_T = networks.define_SE(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.which_method, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netC = networks.define_C(opt.output_nc, opt.input_nc, opt.which_model_netC, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netDP = networks.define_DP(opt.output_nc, opt.input_nc, opt.which_model_netDP, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netRe = networks.define_Re(opt.output_nc, opt.input_nc, opt.which_model_netZE, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)



        if self.isTrain:
                                                
            # initialize optimizers
            self.optimizer = torch.optim.SGD(itertools.chain(self.netSE.parameters(), self.netC.parameters(), self.netPE_S.parameters(), self.netPE_T.parameters(), self.netDP.parameters(), self.netRe.parameters()), lr=opt.lr, momentum = 0.9)
            #self.optimizer_Domain = torch.optim.SGD(itertools.chain(self.netSE.parameters(), self.netDP.parameters()), lr=opt.lr, momentum = 0.9)
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer)
            #self.optimizers.append(self.optimizer_Domain)
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
        #domain_label = torch.cat((domain_t_label, domain_s_label), 0)#.cuda()
        
        if len(self.gpu_ids) > 0:
            s_img = s_img.cuda(self.gpu_ids[0], async=True)
            s_label = s_label.cuda(self.gpu_ids[0], async=True).view(-1)
            t_img = t_img.cuda(self.gpu_ids[0], async=True)
            self.domain_t_label = domain_t_label.cuda(self.gpu_ids[0], async=True)
            self.domain_s_label = domain_s_label.cuda(self.gpu_ids[0], async=True)
            
        self.s_img = Variable(s_img)
        self.t_img = Variable(t_img)        
        self.s_label = Variable(s_label)
        self.img_G = torch.cat((self.t_img, self.s_img), 0)
        #self.domain_label = Variable(domain_label)

    def forward_before_domain_t(self):
        self.E_t = self.netSE(self.t_img)
        self.P_t = self.netPE_T(self.t_img)
        self.code_t = torch.cat((self.E_t, self.P_t), 1)
        self.Re_t = self.netRe(self.code_t).view(-1, 3, 28, 28)
      
    def forward_after_domain_t(self):
        self.E_t = self.netSE(self.t_img)
        self.P_t = self.netPE_T(self.t_img)
        self.code_t = torch.cat((self.E_t, self.P_t), 1)
        self.Re_t = self.netRe(self.code_t).view(-1, 3, 28, 28)
        self.domain_pred_t = self.netDP(self.E_t)

    def backward_before_domain_t(self):
        self.diff_loss_t = networks.DiffLoss(self.E_t, self.P_t)
        self.Re_loss1_t = networks.MSE(self.Re_t, self.t_img)
        self.Re_loss2_t = networks.SIMSE(self.Re_t, self.t_img)
        Domain_loss =  0.005 * self.diff_loss_t + 0.01 * (self.Re_loss1_t + self.Re_loss2_t)
        Domain_loss.backward()
        return Domain_loss.data[0]

    def backward_after_domain_t(self):
        self.diff_loss_t = networks.DiffLoss(self.E_t, self.P_t)
        self.Re_loss1_t = networks.MSE(self.Re_t, self.t_img)
        self.Re_loss2_t = networks.SIMSE(self.Re_t, self.t_img)        
        self.Domain_loss_t = torch.nn.functional.cross_entropy(self.domain_pred_t, self.domain_t_label)
        Domain_loss =  0.005 * self.diff_loss_t + 0.01 * (self.Re_loss1_t + self.Re_loss2_t) + 0.25 * self.Domain_loss_t
        Domain_loss.backward()
        return Domain_loss.data[0]

    def forward_before_domain_s(self):
        self.E_s = self.netSE(self.s_img)
        self.pred_label = self.netC(self.E_s)
        self.P_s = self.netPE_S(self.s_img)
        self.code_s = torch.cat((self.E_s, self.P_s), 1)
        self.Re_s = self.netRe(self.code_s).view(-1, 3, 28, 28)

    def forward_after_domain_s(self):
        self.E_s = self.netSE(self.s_img)
        self.pred_label = self.netC(self.E_s)
        self.P_s = self.netPE_S(self.s_img)
        self.code_s = torch.cat((self.E_s, self.P_s), 1)
        self.Re_s = self.netRe(self.code_s).view(-1, 3, 28, 28)
        self.domain_pred_s = self.netDP(self.E_s)

    def backward_before_domain_s(self):
        self.diff_loss_s = networks.DiffLoss(self.E_s, self.P_s)
        self.Re_loss1_s = networks.MSE(self.Re_s, self.s_img)
        self.Re_loss2_s = networks.SIMSE(self.Re_s, self.s_img)
        self.Class_loss = torch.nn.functional.cross_entropy(self.pred_label, self.s_label)
        C_loss =  self.Class_loss + 0.005 * self.diff_loss_s + 0.01 * (self.Re_loss1_s + self.Re_loss2_s)        
        C_loss.backward()
        return C_loss.data[0]

    def backward_after_domain_s(self):
        self.Domain_loss_s = torch.nn.functional.cross_entropy(self.domain_pred_s, self.domain_s_label)
        self.diff_loss_s = networks.DiffLoss(self.E_s, self.P_s)
        self.Re_loss1_s = networks.MSE(self.Re_s, self.s_img)
        self.Re_loss2_s = networks.SIMSE(self.Re_s, self.s_img)
        self.Class_loss = torch.nn.functional.cross_entropy(self.pred_label, self.s_label)
        C_loss =  self.Class_loss + 0.005 * self.diff_loss_s + 0.01 * (self.Re_loss1_s + self.Re_loss2_s)  + 0.25 * self.Domain_loss_s        
        C_loss.backward()
        return C_loss.data[0]

    
    def optimize_parameters(self, step):
        active_domain_step = 10000
        
        if step < active_domain_step:
        	self.forward_before_domain_t()                
        	Domain_loss = self.backward_before_domain_t()
                self.optimizer.step()
                self.optimizer.zero_grad() 
       
        	self.forward_before_domain_s()                
        	C_loss = self.backward_before_domain_s()
                self.optimizer.step()
                self.optimizer.zero_grad()        
        	
        else:
        	self.forward_after_domain_t()                
        	Domain_loss = self.backward_after_domain_t()
                self.optimizer.step()
                self.optimizer.zero_grad() 
       
        	self.forward_after_domain_s()                
        	C_loss = self.backward_after_domain_s()
                self.optimizer.step()
                self.optimizer.zero_grad()   
        	
        return C_loss, Domain_loss, self.Re_t, self.Re_s
        
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
        
