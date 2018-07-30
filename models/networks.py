import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable, Function
from torch.optim import lr_scheduler
import numpy as np



def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'nopolicy':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.2):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'uniform':
                init.uniform(m.weight.data)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):

    #opt_manualSeed = 100#random.randint(1, 10000)  # fix seed
    #print("Random Seed: ", opt_manualSeed)
    #np.random.seed(opt_manualSeed)
    #torch.manual_seed(opt_manualSeed)
    #torch.cuda.manual_seed_all(opt_manualSeed)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    #init_weights(net, init_type)
    return net


"""=========NetSE============"""
def define_SE(input_nc, output_nc, which_model_netSE, which_method, norm='Instance', use_dropout=False, init_type='normal', gpu_ids=[]):
    netSE = None
    norm_layer = get_norm_layer(norm_type=norm)


    if which_method == 'CORAL':
        if which_model_netSE == 'mnist_mnistm':
            netSE = CORAL_SE_mnist_mnistm(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
        elif which_model_netSE == 'mnist_usps':
            netSE = CORAL_SE_mnist_usps(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
        elif which_model_netSE == 'svhn_mnist':
            netSE = CORAL_SE_svhn_mnist(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
        else:
            raise NotImplementedError('SE_model name [%s] is not recognized' % which_model_netSE)

    
    #elif which_method == ('DANN' or 'DSN'):
    else:
        if which_model_netSE == 'mnist_mnistm':
            netSE = SE_mnist_mnistm(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
        elif which_model_netSE == 'mnist_usps':
            netSE = SE_mnist_usps(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
        elif which_model_netSE == 'svhn_mnist':
            netSE = SE_svhn_mnist(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
        else:
            raise NotImplementedError('SE_model name [%s] is not recognized' % which_model_netSE)


    return init_net(netSE, init_type, gpu_ids)

"""=========Domain_Pred============"""
def define_DP(input_nc, output_nc, which_model_netDP, norm='Instance', use_dropout=False, init_type='normal', gpu_ids=[]):
    netDP = None
    norm_layer = get_norm_layer(norm_type=norm)

   
    if which_model_netDP == 'mnist_mnistm':
        netDP = DP_mnist_mnistm(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netDP == 'mnist_usps':
        #netDP = DP_mnist_usps(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
        netDP = DP_mnist_mnistm(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netDP == 'svhn_mnist':
        #netDP = DP_svhn_mnist(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
        netDP = DP_mnist_mnistm(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
   
    else:
        raise NotImplementedError('DP_model name [%s] is not recognized' % which_model_netDP)
    return init_net(netDP, init_type, gpu_ids)


"""=========NetZE============"""
def define_ZE(input_nc, output_nc, which_model_netZE, norm='Instance', use_dropout=False, init_type='normal', gpu_ids=[]):
    netZE = None
    norm_layer = get_norm_layer(norm_type=norm)

   
    if which_model_netZE == 'mnist_mnistm':
        netZE = ZE_mnist_mnistm(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netZE == 'mnist_usps':
        netZE = ZE_mnist_usps(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netSE == 'svhn_mnist':
        netZE = ZE_svhn_mnist(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
   
    else:
        raise NotImplementedError('ZE_model name [%s] is not recognized' % which_model_netZE)
    return init_net(netZE, init_type, gpu_ids)

"""=========NetRe============"""
def define_Re(input_nc, output_nc, which_model_netRe, norm='Instance', use_dropout=False, init_type='normal', gpu_ids=[]):
    netRe = None
    norm_layer = get_norm_layer(norm_type=norm)

   
    if which_model_netRe == 'mnist_mnistm':
        netRe = Re_mnist_mnistm(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netRe == 'mnist_usps':
        netRe = Re_mnist_usps(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netRe == 'svhn_mnist':
        netRe = Re_svhn_mnist(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
   
    else:
        raise NotImplementedError('Re_model name [%s] is not recognized' % which_model_netRe)
    return init_net(netRe, init_type, gpu_ids)



"""==========Classifier========"""
def define_C(input_nc, output_nc, which_model_netC, norm='Instance', use_dropout=False, init_type='normal', gpu_ids=[]):
    netC = None
    norm_layer = get_norm_layer(norm_type=norm)

   
    if which_model_netC == 'mnist_mnistm':
        netC = C_mnist_mnistm(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netC == 'mnist_usps':
        netC = C_mnist_usps(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netC == 'svhn_mnist':
        netC = C_svhn_mnist(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
   
    else:
        raise NotImplementedError('C_model name [%s] is not recognized' % which_model_netC)
    return init_net(netC, init_type, gpu_ids)


"""==========AClassifier========"""
def define_AC(input_nc, output_nc, which_model_netAC, norm='Instance', use_dropout=False, init_type='normal', gpu_ids=[]):
    netAC = None
    norm_layer = get_norm_layer(norm_type=norm)

   
    if which_model_netAC == 'mnist_mnistm':
        netAC = AC_mnist_mnistm(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netAC == 'mnist_usps':
        netAC = AC_mnist_usps(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netAC == 'svhn_mnist':
        netAC = AC_svhn_mnist(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
   
    else:
        raise NotImplementedError('AC_model name [%s] is not recognized' % which_model_netAC)
    return init_net(netAC, init_type, gpu_ids)

"""==========NetDe========"""
def define_De(input_nc, output_nc, which_model_netDe, which_method, norm='Instance', use_dropout=False, init_type='normal', gpu_ids=[]):
    netDe = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_method == 'DANN':
      if which_model_netDe == 'mnist_mnistm':
        netDe = De_mnist_mnistm(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
      elif which_model_netDe == 'mnist_usps':
        netDe = De_mnist_usps(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
      elif which_model_netDe == 'svhn_mnist':
        netDe = De_svhn_mnist(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
      else:
        raise NotImplementedError('De_model name [%s] is not recognized' % which_model_netDe)

    if which_method == 'DSN':
      if which_model_netDe == 'mnist_mnistm':
        netDe = De_mnist_mnistm(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
      elif which_model_netDe == 'mnist_usps':
        netDe = De_mnist_usps(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
      elif which_model_netDe == 'svhn_mnist':
        netDe = De_svhn_mnist(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
      else:
        raise NotImplementedError('De_model name [%s] is not recognized' % which_model_netDe)

    if which_method == 'CORAL':
      if which_model_netDe == 'mnist_mnistm':
        netDe = De_CORAL_mnist_mnistm(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
      elif which_model_netDe == 'mnist_usps':
        netDe = De_mnist_usps(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
      elif which_model_netDe == 'svhn_mnist':
        netDe = De_svhn_mnist(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
   
      else:
        raise NotImplementedError('De_model name [%s] is not recognized' % which_model_netDe)
    return init_net(netDe, init_type, gpu_ids)




"""==========net SE_mnist_mnistm========="""
class SE_mnist_mnistm(nn.Module):
   
    def __init__(self, input_nc, output_nc, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(SE_mnist_mnistm, self).__init__() 
        self.input_nc = input_nc
        self.output_nc = output_nc
   
        #28,28
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 5, stride = 1, padding = 2)
        self.norm_layer1 = nn.InstanceNorm2d(32)       
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 48, kernel_size = 5, stride = 1, padding = 0)
        self.norm_layer2 = nn.InstanceNorm2d(48)        
        self.relu2 = nn.ReLU(True)
	self.pool2 = nn.MaxPool2d(2)        

        self.fc1 = nn.Linear(1200, 48)
       
    def save_network(self, network, network_label, epoch_label, save_dir):
        save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        
    def forward(self, input):

        out = self.conv1(input)
        #print(out.size())
        out = self.norm_layer1(out)
        out = self.relu1(out)
    	out = self.pool1(out)
        #print(out.size())

        out = self.conv2(out)
        #print(out.size())
        out = self.norm_layer2(out)
        out = self.relu2(out)
    	out = self.pool2(out)

       
        out = out.view(-1, 1200)

        out = self.fc1(out)
       
        return out

"""==========net Re_mnist_mnistm========="""
class Re_mnist_mnistm(nn.Module):
   
    def __init__(self, input_nc, output_nc, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(Re_mnist_mnistm, self).__init__() 
        self.input_nc = input_nc
        self.output_nc = output_nc
   
        self.fc1 = nn.Linear(96, 1024)
        self.lrelu1 = nn.LeakyReLU(0.2, True)


        self.fc2 = nn.Linear(1024, 1024)
        self.lrelu2 = nn.LeakyReLU(0.2, True)

        self.fc3 = nn.Linear(1024, 2048)
        self.lrelu3 = nn.LeakyReLU(0.2, True)
        

        self.fc4 = nn.Linear(2048, 2352)
        self.Ta = nn.Tanh()
        


    #def save_network(self, network, network_label, epoch_label, gpu_ids):
    def save_network(self, network, network_label, epoch_label, save_dir):
        save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        #if len(gpu_ids) and torch.cuda.is_available():
            #network.cuda(device_id=gpu_ids[0])

   
    #def forward(self, input):
    def forward(self, x):



        out = self.fc1(x)
        out = self.lrelu1(out)

        out = self.fc2(out)
        out = self.lrelu2(out)

        out = self.fc3(out)
        out = self.lrelu3(out)

        out = self.fc4(out)
        out = self.Ta(out)

       
        return out

def DiffLoss(input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.mm(input2_l2.t()).pow(2)))

        return diff_loss

def MSE(pred, real):
  
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


def SIMSE(pred, real):

        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse





"""==========net CORAL_SE_mnist_mnistm========="""
def CORAL_Loss(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    #xc = xm.t() @ xm
    xc = torch.matmul(xm.t(),xm)

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    #xct = xmt.t() @ xmt
    xct = torch.matmul(xmt.t(), xmt)

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)

    return loss

class CORAL_SE_mnist_mnistm(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(CORAL_SE_mnist_mnistm, self).__init__()
        self.sharedNet = CORAL_mnist_mnistm(input_nc, output_nc, norm_layer=nn.InstanceNorm2d, use_dropout=False)
        self.fc = nn.Linear(48, 10)

        # initialize according to CORAL paper experiment
        self.fc.weight.data.normal_(0, 0.005)

    def forward(self, source, target):
        source = self.sharedNet(source)
        source = self.fc(source)

        target = self.sharedNet(target)
        target = self.fc(target)

        return source, target

class CORAL_mnist_mnistm(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(CORAL_mnist_mnistm, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc

        # 28,28
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.norm_layer1 = nn.InstanceNorm2d(32)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=0)
        self.norm_layer2 = nn.InstanceNorm2d(48)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(1200, 48)
        # self.lrelu1 = nn.LeakyReLU(0.2, True)

        # self.fc2 = nn.Linear(48, 10)
        # self.lrelu2 = nn.LeakyReLU(0.2, True)


    def save_network(self, network, network_label, epoch_label, save_dir):
        save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def forward(self, input):
        out = self.conv1(input)
        # print(out.size())
        out = self.norm_layer1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        # print(out.size())

        out = self.conv2(out)
        # print(out.size())
        out = self.norm_layer2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        #print(out.size())
        out = out.view(-1, 1200)

        out = self.fc1(out)
        # out = self.lrelu1(out)
        # out = self.fc2(out)
        # out = self.lrelu2(out)

        return out

"""=========DP_mnist_mnistm======="""
class DP_mnist_mnistm(nn.Module):

	def __init__(self, input_nc, output_nc, norm_layer=nn.InstanceNorm2d, use_dropout=False):
		super(DP_mnist_mnistm, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc

		self.fc1 = nn.Linear(48, 100)
                self.relu1 = nn.LeakyReLU(0.2, True) 

                self.fc2 = nn.Linear(100, 2)
  
	def save_network(self, network, network_label, epoch_label, save_dir ):
		save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
    	        save_path = os.path.join(save_dir, save_filename)
    	        torch.save(network.state_dict(), save_path)

        def forward(self, x):

            x = flip_gradient(x)

    	    out = self.fc1(x)
    	   
      	    out = self.relu1(out)

    	    out = self.fc2(out)

    	    return out

class FlipGrad(Function):
       def forward(self, x):
              
              return x.view_as(x)
       def backward(self, grad_output):

             
          
              return grad_output.neg()
def flip_gradient(x):
       return FlipGrad()(x)



"""==========net C_mnist_mnistm========="""
class C_mnist_mnistm(nn.Module):
   
    def __init__(self, input_nc, output_nc, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(C_mnist_mnistm, self).__init__() 
        self.input_nc = input_nc
        self.output_nc = output_nc          

        self.fc1 = nn.Linear(48, 100)
        self.relu1 = nn.LeakyReLU(0.2, True)  

        self.fc2 = nn.Linear(100, 100)
        self.relu2 = nn.LeakyReLU(0.2, True)
  
        self.fc3 = nn.Linear(100, 10) 

    
    def save_network(self, network, network_label, epoch_label, save_dir):
        save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        
    def forward(self, x):

    	#out = self.fc1(x)
    	    
        x = x.contiguous()

    	out = self.fc1(x)
    	   
      	out = self.relu1(out)

    	out = self.fc2(out)
      	out = self.relu2(out)

    	out = self.fc3(out)
       
        return out


"""==========net ZE_mnist_mnistm========="""
class ZE_mnist_mnistm(nn.Module):
   
    def __init__(self, input_nc, output_nc, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(ZE_mnist_mnistm, self).__init__() 
        self.input_nc = input_nc
        self.output_nc = output_nc
   
        #28,28
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 5, stride = 1, padding = 2)
        self.norm_layer1 = nn.InstanceNorm2d(32)       
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5, stride = 1, padding = 0)
        self.norm_layer2 = nn.InstanceNorm2d(64)        
        self.relu2 = nn.ReLU(True)
	self.pool2 = nn.MaxPool2d(2) 

        self.conv3 = nn.Conv2d(64, 16, kernel_size = 3, stride = 1, padding = 0)
        self.norm_layer3 = nn.InstanceNorm2d(16)        
        self.relu3 = nn.ReLU(True)
	#self.pool3 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(144, 144)
        self.lrelu1 = nn.LeakyReLU(0.2, True)


        self.fc2 = nn.Linear(144, 144)
        #self.lrelu1 = nn.LeakyReLU(0.2, True)
        


    
    def save_network(self, network, network_label, epoch_label, save_dir):
        save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        #if len(gpu_ids) and torch.cuda.is_available():
            #network.cuda(device_id=gpu_ids[0])

   
    
    def forward(self, x):

        out = self.conv1(x)
        #print(out.size())
        out = self.norm_layer1(out)
        out = self.relu1(out)
    	out = self.pool1(out)
        #print(out.size())

        out = self.conv2(out)
        #print(out.size())
        out = self.norm_layer2(out)
        out = self.relu2(out)
    	out = self.pool2(out)

        out = self.conv3(out)
        #print(out.size())
        out = self.norm_layer3(out)
        out = self.relu3(out)

        #print(out.size())
        out = out.view(-1, 144)

        out = self.fc1(out)
        out = self.lrelu1(out)

        out = self.fc2(out)
        

       
        return out


"""==========net De_mnist_mnistm========="""
class De_mnist_mnistm(nn.Module):

    def __init__(self, input_nc, output_nc, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(De_mnist_mnistm, self).__init__() 
        self.input_nc = input_nc
        self.output_nc = output_nc
   
       
        self.fc1 = nn.Linear(192, 1024)
        self.lrelu1 = nn.LeakyReLU(0.2, True)


        self.fc2 = nn.Linear(1024, 1024)
        self.lrelu2 = nn.LeakyReLU(0.2, True)

        self.fc3 = nn.Linear(1024, 2048)
        self.lrelu3 = nn.LeakyReLU(0.2, True)
        

        self.fc4 = nn.Linear(2048, 2352)
        self.Ta = nn.Tanh()
        


    #def save_network(self, network, network_label, epoch_label, gpu_ids):
    def save_network(self, network, network_label, epoch_label, save_dir):
        save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        #if len(gpu_ids) and torch.cuda.is_available():
            #network.cuda(device_id=gpu_ids[0])

   
    #def forward(self, input):
    def forward(self, x):



        out = self.fc1(x)
        out = self.lrelu1(out)

        out = self.fc2(out)
        out = self.lrelu2(out)

        out = self.fc3(out)
        out = self.lrelu3(out)

        out = self.fc4(out)
        out = self.Ta(out)

       
        return out

class De_CORAL_mnist_mnistm(nn.Module):

    def __init__(self, input_nc, output_nc, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(De_CORAL_mnist_mnistm, self).__init__() 
        self.input_nc = input_nc
        self.output_nc = output_nc
   
       
        self.fc1 = nn.Linear(154, 1024)
        self.lrelu1 = nn.LeakyReLU(0.2, True)


        self.fc2 = nn.Linear(1024, 1024)
        self.lrelu2 = nn.LeakyReLU(0.2, True)

        self.fc3 = nn.Linear(1024, 2048)
        self.lrelu3 = nn.LeakyReLU(0.2, True)
        

        self.fc4 = nn.Linear(2048, 2352)
        self.Ta = nn.Tanh()
        


    #def save_network(self, network, network_label, epoch_label, gpu_ids):
    def save_network(self, network, network_label, epoch_label, save_dir):
        save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        #if len(gpu_ids) and torch.cuda.is_available():
            #network.cuda(device_id=gpu_ids[0])

   
    #def forward(self, input):
    def forward(self, x):



        out = self.fc1(x)
        out = self.lrelu1(out)

        out = self.fc2(out)
        out = self.lrelu2(out)

        out = self.fc3(out)
        out = self.lrelu3(out)

        out = self.fc4(out)
        out = self.Ta(out)

       
        return out



"""==========net AC_mnist_mnistm========="""
class AC_mnist_mnistm(nn.Module):

    def __init__(self, input_nc, output_nc, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(AC_mnist_mnistm, self).__init__() 
	self.input_nc = input_nc
	self.output_nc = output_nc

	self.fc1 = nn.Linear(144, 48)
        self.relu1 = nn.LeakyReLU(0.2, True)

	self.fc2 = nn.Linear(48, 48)
        self.relu2 = nn.LeakyReLU(0.2, True) 

        self.fc3 = nn.Linear(48, 10)
  
    def save_network(self, network, network_label, epoch_label, save_dir ):
	save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
    	save_path = os.path.join(save_dir, save_filename)
    	torch.save(network.state_dict(), save_path)

    def forward(self, x):
            
        out = self.fc1(x)
    	   
        out = self.relu1(out)

        out = self.fc2(out)
    	   
        out = self.relu2(out)

        out = self.fc3(out)

    	return out



"""==========net SE_mnist_usps========="""
class SE_mnist_usps(nn.Module):
   
    def __init__(self, input_nc, output_nc, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(NetSE_mnist_usps, self).__init__() 
        self.input_nc = input_nc
        self.output_nc = output_nc
   
        #28,28
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 5, stride = 1, padding = 2)
        self.norm_layer1 = nn.InstanceNorm2d(32)       
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 48, kernel_size = 5, stride = 1, padding = 0)
        self.norm_layer2 = nn.InstanceNorm2d(48)        
        self.relu2 = nn.ReLU(True)
	self.pool2 = nn.MaxPool2d(2)        

        self.fc1 = nn.Linear(1200, 48)
       
    def save_network(self, network, network_label, epoch_label, save_dir):
        save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        
    def forward(self, input):

        out = self.conv1(input)
        #print(out.size())
        out = self.norm_layer1(out)
        out = self.relu1(out)
    	out = self.pool1(out)
        #print(out.size())

        out = self.conv2(out)
        #print(out.size())
        out = self.norm_layer2(out)
        out = self.relu2(out)
    	out = self.pool2(out)

       
        out = out.view(-1, 1200)

        out = self.fc1(out)
       
        return out


"""==========net SE_svhn_mnist========="""
class SE_svhn_mnist(nn.Module):
   
    def __init__(self, input_nc, output_nc, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(NetSE_svhn_mnist, self).__init__() 
        self.input_nc = input_nc
        self.output_nc = output_nc
   
        #28,28
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 5, stride = 1, padding = 2)
        self.norm_layer1 = nn.InstanceNorm2d(32)       
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 48, kernel_size = 5, stride = 1, padding = 0)
        self.norm_layer2 = nn.InstanceNorm2d(48)        
        self.relu2 = nn.ReLU(True)
	self.pool2 = nn.MaxPool2d(2)        

        self.fc1 = nn.Linear(1200, 48)
       
    def save_network(self, network, network_label, epoch_label, save_dir):
        save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        
    def forward(self, input):

        out = self.conv1(input)
        #print(out.size())
        out = self.norm_layer1(out)
        out = self.relu1(out)
    	out = self.pool1(out)
        #print(out.size())

        out = self.conv2(out)
        #print(out.size())
        out = self.norm_layer2(out)
        out = self.relu2(out)
    	out = self.pool2(out)

       
        out = out.view(-1, 1200)

        out = self.fc1(out)
       
        return out



