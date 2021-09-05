import torch
import torch.nn as nn
import argparse
import os.path as osp
import os
from Code.utils.evaluator import Eval_thread
from Code.utils.dataloader import EvalDataset
import scipy.io as scio 
# from concurrent.futures import ThreadPoolExecutor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(cfg):
    
    root_dir = cfg.root_dir
    gt_dir   = cfg.gt_dir
    
    if cfg.save_dir is not None:
        output_dir = cfg.save_dir
    else:
        output_dir = root_dir
        
        
    method_names  = cfg.methods
    dataset_names = cfg.datasets
        
    
#    if cfg.methods is None:
#        method_names = os.listdir(pred_dir)
#    else:
#        method_names = cfg.methods.split(' ')
#    if cfg.datasets is None:
#        dataset_names = os.listdir(gt_dir)
#    else:
#        dataset_names = cfg.datasets.split(' ')
    
    threads = []
    
    for method in method_names:
        
        test_res = []
        
        for dataset in dataset_names:
            loader = EvalDataset(osp.join(root_dir, method, dataset), osp.join(gt_dir, dataset,'GT'))
            thread = Eval_thread(loader, method, dataset, output_dir, cfg.cuda)
            threads.append(thread)

            ##
            print(['Evaluating----------',dataset,'----------'])
            mae,s,max_f,max_e= thread.run()    ## only compute MAE and s_measure
            
            print(['MAE:',mae,'----- Smeansure:',s,'----- max_f:',max_f,'----- max_e:',max_e])
            
            test_res.append([mae,s,max_f,max_e])
            scio.savemat('res.mat', {'test_res':test_res})  
            
            
            
            
            
#            
#    for thread in threads:
#        print(thread.run())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    gt_path       = './Data/TestDataset/'
    sal_path      = './Predict_maps/'
    test_datasets = ['NJU2K','NLPR', 'DES', 'SSD','SIP', 'STERE'] 

    
    parser.add_argument('--methods',  type=str,  default=['SPNet'])
    parser.add_argument('--datasets', type=str,  default=test_datasets)
    parser.add_argument('--gt_dir',   type=str,  default=gt_path)
    parser.add_argument('--root_dir', type=str,  default=sal_path)
    parser.add_argument('--save_dir', type=str,  default=None)
    parser.add_argument('--cuda',     type=bool, default=True)
    cfg = parser.parse_args()
    main(cfg)
