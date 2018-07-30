import torch
from torch.autograd import Variable
import itertools
from .base_model import BaseModel
from . import networks
import torch.nn as nn


class DSN_m_iter_Model(BaseModel):
    def name(self):
        return 'DSN_m_iter_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['Domain', 'C']
        
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks

        self.model_names = ['SE', 'PE_S', 'PE_T', 'C', 'DP', 'Re']
        self.load_model_names_DA = ['SE', 'PE_S', 'PE_T', 'C', 'Re']
        self.load_model_names_Di = ['ZE', 'De']
        self.load_model_names_Pre = ['SE']

        # load/define networks
        self.netSEPre = networks.define_SE(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.which_method, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netSE = networks.define_SE(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.which_method, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netPE_S = networks.define_SE(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.which_method, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netPE_T = networks.define_SE(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.which_method, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netC = networks.define_C(opt.output_nc, opt.input_nc, opt.which_model_netC, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netDP = networks.define_DP(opt.output_nc, opt.input_nc, opt.which_model_netDP, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netRe = networks.define_Re(opt.output_nc, opt.input_nc, opt.which_model_netZE, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netZE = networks.define_ZE(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.norm,
                                        not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netAC = networks.define_AC(opt.output_nc, opt.input_nc, opt.which_model_netC, opt.norm, not opt.no_dropout,
                                        opt.init_type, self.gpu_ids)
        self.netDe = networks.define_De(opt.output_nc, opt.input_nc, opt.which_model_netDe,  opt.which_method, opt.norm, 
                                        not opt.no_dropout, opt.init_type, self.gpu_ids)

        self.load_networks_DA(opt.which_usename_DA, opt.which_epochs_DA)
        self.load_networks_Di(opt.which_usename_Di, opt.which_epochs_Di)
        self.load_networks_Pre(opt.which_usename_DA, opt.which_epochs_DA)
        self.nets = [self.netSE, self.netDP, self.netC, self.netPE_T, self.netPE_S, self.netRe]
        



        if self.isTrain:
                                                
            # initialize optimizers
            self.params = list(self.netSE.parameters()) + list(self.netC.parameters()) +list(self.netDP.parameters()) + list(self.netPE_T.parameters()) + list(self.netPE_S.parameters()) + list(self.netRe.parameters())
            
            self.optimizer = torch.optim.SGD(self.params, lr=opt.lr/opt.lr_de, momentum = 0.9)#10
            
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer)
            
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    def set_input(self, opt, s_img, s_label, t_img):

        domain_t_label = torch.LongTensor(opt.batchSize/2).zero_() + 1
        domain_s_label = torch.LongTensor(opt.batchSize/2*2).zero_()
        
        
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

        self.mid_s = self.netSEPre(self.s_img)
        self.mid_z = self.netZE(self.t_img)

        self.mid_code = torch.cat((self.mid_s, self.mid_z), 1)#.contiguous()
        self.mid_img = Variable(self.netDe(self.mid_code).view(-1, 3, 28, 28).data)
        self.mid_label = self.s_label

        self.s_mid_label = torch.cat((self.s_label, self.mid_label), 0)
        self.s_mid_img = torch.cat((self.s_img, self.mid_img), 0)

        self.img_G = torch.cat((self.s_img, self.mid_img, self.t_img), 0)
        

        return self.img_G

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
        Domain_loss =  0.0001 * self.diff_loss_t + 0.01 * (self.Re_loss1_t + self.Re_loss2_t)
        Domain_loss.backward()
        return Domain_loss.data[0]

    def backward_after_domain_t(self):
        self.diff_loss_t = networks.DiffLoss(self.E_t, self.P_t)
        self.Re_loss1_t = networks.MSE(self.Re_t, self.t_img)
        self.Re_loss2_t = networks.SIMSE(self.Re_t, self.t_img)        
        self.Domain_loss_t = torch.nn.functional.cross_entropy(self.domain_pred_t, self.domain_t_label)
        Domain_loss =  0.0001 * self.diff_loss_t + 0.01 * (self.Re_loss1_t + self.Re_loss2_t) + 0.25 * self.Domain_loss_t
        Domain_loss.backward()
        return Domain_loss.data[0]

    def forward_before_domain_s(self):
        self.E_s = self.netSE(self.s_img)
        self.pred_label = self.netC(self.E_s)
        self.P_s = self.netPE_S(self.s_img)
        self.code_s = torch.cat((self.E_s, self.P_s), 1)
        self.Re_s = self.netRe(self.code_s).view(-1, 3, 28, 28)

    def forward_after_domain_s(self):
        self.E_s_mid = self.netSE(self.s_mid_img)
        self.E_s = self.netSE(self.s_img)
        self.pred_label = self.netC(self.E_s_mid)
        self.P_s = self.netPE_S(self.s_img)
        self.code_s = torch.cat((self.E_s, self.P_s), 1)
        self.Re_s = self.netRe(self.code_s).view(-1, 3, 28, 28)
        self.domain_pred_s_mid = self.netDP(self.E_s_mid)

    def backward_before_domain_s(self):
        self.diff_loss_s = networks.DiffLoss(self.E_s, self.P_s)
        self.Re_loss1_s = networks.MSE(self.Re_s, self.s_img)
        self.Re_loss2_s = networks.SIMSE(self.Re_s, self.s_img)
        self.Class_loss = torch.nn.functional.cross_entropy(self.pred_label, self.s_label)
        C_loss =  self.Class_loss + 0.0001 * self.diff_loss_s + 0.01 * (self.Re_loss1_s + self.Re_loss2_s)        
        C_loss.backward()
        return C_loss.data[0]

    def backward_after_domain_s(self):
        self.Domain_loss_s = torch.nn.functional.cross_entropy(self.domain_pred_s_mid, self.domain_s_label)
        self.diff_loss_s = networks.DiffLoss(self.E_s, self.P_s)
        self.Re_loss1_s = networks.MSE(self.Re_s, self.s_img)
        self.Re_loss2_s = networks.SIMSE(self.Re_s, self.s_img)
        self.Class_loss = torch.nn.functional.cross_entropy(self.pred_label, self.s_mid_label)
        C_loss =  self.Class_loss + 0.0001 * self.diff_loss_s + 0.01 * (self.Re_loss1_s + self.Re_loss2_s)  + 0.75 * self.Domain_loss_s        
        C_loss.backward()
        return C_loss.data[0]


    
    def optimize_parameters(self):
        
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
        
