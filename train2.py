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
    mnist_data_loader, mnistm_data_loader, eval_data_loader = CreateDataLoader(opt)
    mnist_dataset, mnistm_dataset, eval_dataset = mnist_data_loader.load_data(), mnistm_data_loader.load_data(), eval_data_loader.load_data()
    mnist_dataset_size = len(mnist_data_loader)
    mnistm_dataset_size = len(mnistm_data_loader)
    
    
    print('#mnist training images = %d' % mnist_dataset_size)
    print('#mnistm training images = %d' % mnistm_dataset_size)
    

    model = create_model(opt)
    best_acc = 0    
    total_steps = 0
    i = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        j = 0



        """=====train====="""       
        mnistm_dataloader = iter(mnistm_dataset)
        for step, (s_img, s_label, _) in enumerate(mnist_dataset):
             
             
             #print(len(mnist_dataset))
             if j < len(mnist_dataset)/opt.batchSize -1:    
                 j = j + 1
                 i = i + 1

                 t_img, _, _ = mnistm_dataloader.next()
            
                 model.set_input(opt, s_img, s_label, t_img)
                 Reimg, AC_loss, Re_loss, loss_t = model.optimize_parameters(step)

                 writer.add_scalar('data/AC_loss', AC_loss, i)
                 writer.add_scalar('data/Re_loss', Re_loss, i)
                 writer.add_scalar('data/loss_t', loss_t, i)

                 Comimg, Commid = model.combine(opt)

                 if i % 100 == 0:

                    img_G = torch.cat((s_img, t_img), 0)
                    imgvis = img_G.view(-1, 3, 28, 28)
        	    img = utils.make_grid(imgvis, nrow = 8, padding = 2, normalize=True, scale_each=False, pad_value=1)
       	  	    writer.add_image('img/img_G', img, global_step = i)

                    cycle_imgvis = Reimg.view(-1, 3, 28, 28)
                    cycle_img = utils.make_grid(cycle_imgvis.data, nrow=8, padding=2, normalize=True, scale_each=False,
                                         pad_value=1)
                    writer.add_image('img/cycle_img', cycle_img, i)

                    combine_imgvis = Comimg.view(-1, 3, 28, 28)
                    cycle_img = utils.make_grid(combine_imgvis.data, nrow=8, padding=2, normalize=True, scale_each=False,
                                         pad_value=1)
                    writer.add_image('img/combine_img', cycle_img, i)

                    combine_imgvis = Commid.view(-1, 3, 28, 28)
                    cycle_img = utils.make_grid(combine_imgvis.data, nrow=8, padding=2, normalize=True, scale_each=False,
                                         pad_value=1)
                    writer.add_image('img/mid_img', cycle_img, i)



                 if step % 10 == 0:

                    if (opt.model == 'DANN_m'):
        	     print ('DAANN_M:Epoch [%d/%d],Step [%d/%d]' %(epoch, opt.niter + opt.niter_decay, step, len(mnistm_dataset)/opt.batchSize/2))
                    if (opt.model == 'DANN_mv2'):
        	     print ('DAANN_Mv2:Epoch [%d/%d],Step [%d/%d]' %(epoch, opt.niter + opt.niter_decay, step, len(mnistm_dataset)/opt.batchSize/2))
                    if (opt.model == 'Di_m'):
        	     print ('Di_m(%s):Epoch [%d/%d],Step [%d/%d]' %(opt.name, epoch, opt.niter + opt.niter_decay, step, len(mnist_dataset)/opt.batchSize/2))
                    if (opt.model == 'Di_mv2'):
        	     print ('Di_mv2(%s):Epoch [%d/%d],Step [%d/%d]' %(opt.name, epoch, opt.niter + opt.niter_decay, step, len(mnist_dataset)/opt.batchSize/2))
                    if (opt.model == 'Di_mv3'):
        	     print ('Di_mv3(%s):Epoch [%d/%d],Step [%d/%d]' %(opt.name, epoch, opt.niter + opt.niter_decay, step, len(mnist_dataset)/opt.batchSize/2))
                    if (opt.model == 'Di_iter_m'):
        	     print ('Di_iter_m(%s):Epoch [%d/%d],Step [%d/%d]' %(opt.name, epoch, opt.niter + opt.niter_decay, step, len(mnist_dataset)/opt.batchSize/2))
                    if (opt.model == 'Di_CORAL_m'):
        	     print ('Di_CORAL_m(%s):Epoch [%d/%d],Step [%d/%d]' %(opt.name, epoch, opt.niter + opt.niter_decay, step, len(mnist_dataset)/opt.batchSize/2))
                    if (opt.model == 'Di_iter_CORAL_m'):
        	     print ('Di_iter_CORAL_m(%s):Epoch [%d/%d],Step [%d/%d]' %(opt.name, epoch, opt.niter + opt.niter_decay, step, len(mnist_dataset)/opt.batchSize/2))
                    if (opt.model == 'Di_DSN_m'):
        	     print ('Di_DSN_m(%s):Epoch [%d/%d],Step [%d/%d]' %(opt.name, epoch, opt.niter + opt.niter_decay, step, len(mnist_dataset)/opt.batchSize/2))
                    if (opt.model == 'Di_iter_DSN_m'):
        	     print ('Di_iter_DSN_m(%s):Epoch [%d/%d],Step [%d/%d]' %(opt.name, epoch, opt.niter + opt.niter_decay, step, len(mnist_dataset)/opt.batchSize/2))
                    


        if i % opt.save_latest_freq == 0:
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
