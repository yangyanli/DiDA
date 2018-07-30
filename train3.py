import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from tensorboardX import SummaryWriter
import torch
import torchvision.utils as utils


writer = SummaryWriter()

if __name__ == '__main__':
    opt = TrainOptions().parse()

    """
    opt.dataroot = './traindata'
    opt.name = 'DANN_miter1step1'
    opt.batchSize = 64
    opt.lr = 0.00001
    opt.model = 'DANN_m_iter'
    opt.which_epochs_DA = 1
    opt.which_usename_DA = 'DANN_mstep1without'
    opt.which_epochs_Di = 10
    opt.which_usename_Di = 'DANN_mv3step2'
    opt.gpu_ids = [0]
    opt.save_epoch_freq = 100
    """

    mnist_data_loader, mnistm_data_loader, eval_data_loader = CreateDataLoader(opt)
    mnist_dataset, mnistm_dataset, eval_dataset = mnist_data_loader.load_data(), mnistm_data_loader.load_data(), eval_data_loader.load_data()
    mnist_dataset_size = len(mnist_data_loader)
    mnistm_dataset_size = len(mnistm_data_loader)
    eval_dataset_size = len(eval_data_loader)
    
    print('#mnist training images = %d' % mnist_dataset_size)
    print('#mnistm training images = %d' % mnistm_dataset_size)
    print('#eval training images = %d' % eval_dataset_size)
    print('#eval training images = %d' % len(eval_dataset))

    model = create_model(opt)
    best_acc = 0    
    total_steps = 0
    i = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        j = 0

        """=====test=====""" 
        correct_t = 0
        for test_step, (t_img, t_label, _) in enumerate(eval_dataset):          
  	    correct_t += model.test(t_img, t_label)
        acc = float(correct_t)/float(len(eval_dataset)) 
        print('epoch %d / %d \t acc: %8f' % (epoch, opt.niter + opt.niter_decay, acc))
        writer.add_scalar('eval/acc', acc, epoch)
  	if acc >= best_acc:
   	     best_acc = acc 
   	     print('saving the best model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
             model.save_networks('best'+str(epoch))

        """=====train====="""       
        mnistm_dataloader = iter(mnistm_dataset)
        for step, (s_img, s_label, _) in enumerate(mnist_dataset):
             
             i = i+1
             
             if j < len(mnist_dataset)/opt.batchSize-1:    
                 j = j +1
                 

                 t_img, _, _ = mnistm_dataloader.next()
            
                 img_G = model.set_input(opt, s_img, s_label, t_img)
                 if (opt.which_method == 'CORAL'):
                     _lamda = (epoch) / (opt.niter + opt.niter_decay)
                     C_loss, Domain_loss = model.optimize_parameters(_lamda)
                 elif (opt.which_method == 'DSN'):
                     C_loss, Domain_loss, Re_t, Re_s = model.optimize_parameters()
                 else:
                     C_loss, Domain_loss = model.optimize_parameters()

                 writer.add_scalar('data/C_loss', C_loss, i)
                 writer.add_scalar('data/Domain_loss', Domain_loss, i)

             if i % 100 == 0:

                #img_G = torch.cat((s_img, t_img), 0)
                imgvis = img_G.view(-1, 3, 28, 28)
        	img = utils.make_grid(imgvis, nrow = 10, padding = 2, normalize=True, scale_each=False, pad_value=1)   
       	  	writer.add_image('img/img_G', img, global_step = i)

                if (opt.which_method == 'DSN'):
                	img_Re = torch.cat((Re_s, Re_t), 0)
                	imgvis = img_Re.view(-1, 3, 28, 28)
        		img = utils.make_grid(imgvis, nrow = 10, padding = 2, normalize=True, scale_each=False, pad_value=1)   
       	  		writer.add_image('img/img_Re', img, global_step = i)




    	     if step % 10 == 0:

                if (opt.model == 'DANN_m_iter'):
        	     print ('DAANN_M_iter(%s):Epoch [%d/%d],Step [%d/%d]' %(opt.name, epoch, opt.niter + opt.niter_decay, step, len(mnist_dataset)/opt.batchSize/2))
                if (opt.model == 'DANN_m_iterv2'):
        	     print ('DAANN_M_iterv2(%s):Epoch [%d/%d],Step [%d/%d]' %(opt.name, epoch, opt.niter + opt.niter_decay, step, len(mnist_dataset)/opt.batchSize/2))
                if (opt.model == 'DANN_m_iterv3'):
        	     print ('DAANN_M_iterv3(%s):Epoch [%d/%d],Step [%d/%d]' %(opt.name, epoch, opt.niter + opt.niter_decay, step, len(mnist_dataset)/opt.batchSize/2))
                if (opt.model == 'DANN_m_iterv3'):
        	     print ('DAANN_M_iterv3(%s):Epoch [%d/%d],Step [%d/%d]' %(opt.name, epoch, opt.niter + opt.niter_decay, step, len(mnist_dataset)/opt.batchSize/2))
                if (opt.model == 'CORAL_m_iter'):
        	     print ('CORAL_m_iter(%s):Epoch [%d/%d],Step [%d/%d]' %(opt.name, epoch, opt.niter + opt.niter_decay, step, len(mnist_dataset)/opt.batchSize/2))
                if (opt.model == 'DSN_m_iter'):
        	     print ('DSN_m_iter(%s):Epoch [%d/%d],Step [%d/%d]' %(opt.name, epoch, opt.niter + opt.niter_decay, step, len(mnist_dataset)/opt.batchSize/2))




        if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

        iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
        	print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
                model.save_networks('latest')
                model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        #model.update_learning_rate()
