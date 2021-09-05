#coding:utf8
import warnings
class DefaultConfig(object):
        
    testsize     = 352
    gpu_id        = 0
    
    test_path     = '/raid/Tao/RGBD/Dataset/RGBD1010/TestDataset/'
    model_path    = '/raid/Tao/RGBD/SCL_RGBD_v2/Results/SCLNet_semi_MT_train_var/'
    test_datasets = ['NJU2K','NLPR', 'DES', 'SSD','LFSD','SIP', 'STERE', 'GIT']
    
    
    method        = 'SCLNet'
    sal_map_path  = '/raid/Tao/RGBD/SCL_RGBD_v2/test_maps/'
    
            
def parse(self,kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self,k,v)

        print('user config:')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k,getattr(self,k))

DefaultConfig.parse = parse
opt =DefaultConfig()

#parser = argparse.ArgumentParser()
#parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
#parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
#parser.add_argument('--;', type=int, default=2, help='size of the batches')
#parser.add_argument('--dataset_name', type=str, default='edges2shoes', help='name of the dataset')
#parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
#parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
#parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
#parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
#parser.add_argument('--img_size', type=int, default=128, help='size of each image dimension')
#parser.add_argument('--channels', type=int, default=3, help='number of image channels')
#parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
#parser.add_argument('--sample_interval', type=int, default=200, help='interval betwen image samples')
#parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
#opt = parser.parse_args()
#print(opt)

