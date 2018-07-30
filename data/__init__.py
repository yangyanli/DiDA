import torch.utils.data
import dataloader
import dataloaderm
import dataloaderm_eval

from data.base_data_loader import BaseDataLoader


def CreateDataLoader(opt):
    #data_loader = CustomDatasetDataLoader()
    mnist_data_loader = Mnist_DataLoader()
    mnistm_data_loader = MnistM_DataLoader()
    eval_data_loader = eval_DataLoader()
    print(mnist_data_loader.name())
    print(mnistm_data_loader.name())
    print(eval_data_loader.name())
    mnist_data_loader.initialize(opt)
    mnistm_data_loader.initialize(opt)
    eval_data_loader.initialize(opt)
    return mnist_data_loader, mnistm_data_loader, eval_data_loader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data

class Mnist_DataLoader(BaseDataLoader):
    def name(self):
        return 'Mnist_DataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        root_path = "traindata/"
        
        self.mnistdataset = dataloader.get_training_set(root_path)
        self.mnistdataloader = dataloader.DataLoader(dataset=self.mnistdataset, num_workers=int(opt.nThreads), batch_size = opt.batchSize/2, shuffle= True)

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.mnistdataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.mnistdataloader):
            if i * self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data

class MnistM_DataLoader(BaseDataLoader):
    def name(self):
        return 'MnistM_DataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        root_path = "traindata/"
        
        self.mnistmdataset = dataloaderm.get_training_set(root_path)
        self.mnistmdataloader = dataloaderm.DataLoader(dataset=self.mnistmdataset, num_workers=int(opt.nThreads), batch_size = opt.batchSize/2, shuffle= True)

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.mnistmdataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.mnistmdataloader):
            if i * self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data


class eval_DataLoader(BaseDataLoader):
    def name(self):
        return 'eval_DataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        root_path = "traindata/"
        
        self.evaldataset = dataloaderm_eval.get_training_set(root_path)
        self.evaldataloader = dataloaderm_eval.DataLoader(dataset=self.evaldataset, num_workers=int(opt.nThreads), batch_size = opt.batchSize/2, shuffle= True)

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.evaldataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.evaldataloader):
            if i * self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data

