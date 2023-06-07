import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import traceback
import copy
import logging
import math
import random
import time
import os
import datetime

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from model.yolov5.models.common import DetectMultiBackend
from model.yolov5.utils.general import (LOGGER, Profile, check_amp,
                                        check_dataset, check_file,
                                        check_img_size, check_yaml, colorstr,
                                        non_max_suppression, scale_coords,
                                        xywh2xyxy)

from model.yolov5.utils.loggers import Loggers
from model.yolov5.utils.loss import ComputeLoss
from model.yolov5.utils.metrics import (ConfusionMatrix, ap_per_class, box_iou,
                                        yolov5_ap_per_class)

import fedml
from fedml.core import ClientTrainer
from model.yolov5.utils.loss import ComputeLoss
from model.yolov5.utils.general import Profile, non_max_suppression, xywh2xyxy, scale_coords
from model.yolov5.utils.metrics import ConfusionMatrix, yolov5_ap_per_class, ap_per_class, box_iou
from fedml.core.mlops.mlops_profiler_event import  MLOpsProfilerEvent
from model.yolov5 import val as validate # imported to use original yolov5 validation function!!!

from model.yolov5.utils.loggers import Loggers
from model.yolov5.utils.loss import ComputeLoss
from model.yolov5.utils.metrics import (ConfusionMatrix, ap_per_class, box_iou,
                                        yolov5_ap_per_class)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

use_shoaib_code=True
if use_shoaib_code:      

    import sys
    sys.path.append('Yolov5_DeepSORT_PseudoLabels/')
    sys.path.append('Yolov5_DeepSORT_PseudoLabels/yolov5')
    sys.path.append('Yolov5_DeepSORT_PseudoLabels/deep_sort')
    
    import shutil
    
    from model.yolov5 import val_pseudos as pseudos  # imported to use modified yolov5 validation function!!!
    from Yolov5_DeepSORT_PseudoLabels import trackv2_from_file as recover
    from Yolov5_DeepSORT_PseudoLabels.merge_forward_backward_v2 import merge 

    from data.data_loader import create_dataloader
    import yaml
    
    def use_new_data(args,model,compute_loss,train_data,test_data):
        
        
        # -------------------------------- Shoaib Code --------------------------------------- #
        args.flag_new_data = 0
        args.curr_step=0
        args.old_train_data = train_data
        args.old_test_data  = test_data
        genTest = False # args.generate_validation_pseudos
        
        merged_test_dataloader, new_test_dataloader_gt = [], []
        
        round_above_min = args.round_idx > args.new_data_min_rounds
        clinet_in_list = args.rank in args.psuedo_generate_clients or args.rank in args.psuedo_recovery_on_clients
        
        if args.use_new_data and round_above_min  and clinet_in_list: 
            try:
                args.org_data_train = train_data.dataset.img_files
                args.new_data_train = check_dataset(args.new_data_conf)['train']
                
                if genTest: 
                    args.org_data_test = test_data.dataset.img_files
                    args.new_data_test = check_dataset(args.new_data_conf)['val']
                        
                    _log_it(args,f"Loading {args.new_data_num_images_test } images from new test dataset at [{args.new_data}]")
                    new_test_dataloader_gt  = get_new_data(args.new_data_test, args, numimgs = args.new_data_num_images_test, shuffle=False)
                    
                # TODO: Should we use pseudo labels or GT ?
                if args.use_new_data_pseudos and ((args.rank in args.psuedo_generate_clients) or (args.rank in args.psuedo_recovery_on_clients)):
                    
                    if (args.client_map50_all[args.client_id]<args.min_map_for_pseudo_generation and args.round_idx>0)or(args.round_idx==0 and args.weights=="none"):
                        _log_it(args,f"mAP too low. More training required before generating pseudo labels.")
                        return train_data, merged_test_dataloader, new_test_dataloader_gt #FIXME: Return
                    
                    else:
                        # Remove old labels
                        # if os.path.isdir(args.save_dir/'labels'):   
                        _dir = args.save_dir/'train'/'labels'/f'Trainer_{args.client_id}--Round_{args.round_idx}'  
                        if os.path.isdir(_dir):     
                            _log_it(args,f"Removing old label files if present at {_dir}")
                            shutil.rmtree(_dir)
                            
                        _dir = args.save_dir/'test'/'labels'/f'Trainer_{args.client_id}--Round_{args.round_idx}'  
                        if os.path.isdir(_dir):     
                            _log_it(args,f"Removing old label files if present at {_dir}")
                            shutil.rmtree(_dir)
                        
                        
                        # Load new dataset equal to args.new_data_num_images_train from args.new_data
                        _log_it(args,f"Loading {args.new_data_num_images_train} images from new train dataset at [{args.new_data}]")
                        try:
                            new_train_dataloader = get_new_data(args.new_data_train,args, numimgs = args.new_data_num_images_train, shuffle=False)
                        except Exception as e:
                            print(traceback.format_exc())
                            
                        
                        args.save_dir_train = args.save_dir/'train'
                        if genTest: args.save_dir_test  = args.save_dir/'test'
                            
                        # ------------------ Generate only high confidence pseudo labels without confidence values -----------------
                        # if not args.use_new_data_recover and args.rank in args.psuedo_generate_clients and not (args.rank in args.psuedo_recovery_on_clients):
                        if not args.use_new_data_recover :
                            # Generate HIGH Confidence Pseudo Labels without confidence values for new dataset
                            _f_train =      pseudo_labels(  data            =   check_dataset(args.new_data_conf),
                                                            model           =   model,
                                                            dataloader      =   new_train_dataloader,
                                                            compute_loss    =   compute_loss, 
                                                            args            =   args,
                                                            conf_thresh     =   0.5,
                                                            save_conf       =   False,
                                                            save_dir        =   args.save_dir_train
                                                            )
                            if _f_train==[]: 
                                return train_data, merged_test_dataloader, new_test_dataloader_gt   #FIXME: Return original dataloader
                            else:
                                args.pseudo_label_path_train = os.path.split(_f_train)[0]    
                                # Merge un-recovered pseudo labels of new data with original data
                                # print(args.pseudo_label_path)
                                new_data_path_train = args.pseudo_label_path_train


                                if genTest: 
                                    _f_test =   pseudo_labels(  data            =   check_dataset(args.new_data_conf),
                                                                model           =   model,
                                                                dataloader      =   new_test_dataloader_gt,
                                                                compute_loss    =   compute_loss, 
                                                                args            =   args,
                                                                conf_thresh     =   0.5,
                                                                save_conf       =   False,
                                                                save_dir        =   args.save_dir_test
                                                                )
                                    
                                    args.pseudo_label_path_test = os.path.split(_f_test)[0]    
                                    # Merge un-recovered pseudo labels of new data with original data
                                    # print(args.pseudo_label_path)
                                    new_data_path_test = args.pseudo_label_path_test
                                    
                                data_type="pseudo labels"



                        # ----------------- Generate Pseudo Labels and perform missing box recovery --------------------
                        if args.use_new_data_recover: #TODO: Should we apply recovery  on the generated pseudo labels 
                        # if args.use_new_data_recover and (args.rank in args.psuedo_recovery_on_clients): #TODO: Should we apply recovery  on the generated pseudo labels 

                            # Train Dataset
                            # Generate Pseudo Labels for new dataset
                            args.path_low_train =   pseudo_labels(  data            =   check_dataset(args.new_data_conf),
                                                                    model           =   model,
                                                                    dataloader      =   new_train_dataloader,
                                                                    compute_loss    =   None, 
                                                                    args            =   args,
                                                                    conf_thresh     =   args.conf_thresh_low,
                                                                    save_dir        =   args.save_dir_train
                                                                    )
                            if args.path_low_train==[]: 
                                return train_data, merged_test_dataloader, new_test_dataloader_gt   #FIXME: Return original dataloader
                            # Run Forward and Backward Bounding Box Recovery
                            args.new_pseudos_recovered_train = recover_labels( args , args.path_low_train )
                            # Merge recovered pseudos of new data with original data
                            new_data_path_train = args.new_pseudos_recovered_train
                            data_type="recovered pseudo labels"


                        
                            # Test Dataset
                            if genTest: 
                                # Generate Pseudo Labels for new dataset
                                args.path_low_test =    pseudo_labels(  data            =   check_dataset(args.new_data_conf),
                                                                        model           =   model,
                                                                        dataloader      =   new_test_dataloader_gt,
                                                                        compute_loss    =   compute_loss, 
                                                                        args            =   args,
                                                                        conf_thresh     =   args.conf_thresh_low,
                                                                        save_dir        =   args.save_dir_test
                                                                        )
                                # Run Forward and Backward Bounding Box Recovery
                                args.new_pseudos_recovered_test = recover_labels( args , args.path_low_test )
                                # Merge recovered pseudos of new data with original data
                                new_data_path_test = args.new_pseudos_recovered_test
                            data_type="recovered pseudo labels"



                else:# Don't use pseudo labels
                    if (args.rank in args.psuedo_generate_clients) and (args.rank in args.psuedo_generate_clients):
                        _log_it(args,f"Client not in the list to generate pseudo labels or recover them.")
                        
                    _log_it(args,f"Removing old label files if present at {args.save_dir/'labels'}")
                    
                    # Use ground truth data as new training data
                    new_data_path_train = get_X_GTs_train(args)
                    _log_it(args,f"Reading {args.new_data_num_images_train} GT for new train dataset at {args.new_data_train} and save at {new_data_path_train}")
                    
                    if genTest:  
                        new_data_path_test = get_X_GTs_test(args)
                        _log_it(args,f"Reading {args.new_data_num_images_test} GT for new test dataset at {args.new_data_test} and save at {new_data_path_test}")
                        
                    data_type="ground truth"
                
                #merge
                # train data = train data + pseudo labels    
                mixed_train_data_path, numimgs = merge_training_lists(args, org_files = args.org_data_train, new_data_path = new_data_path_train)        
                _log_it(args,f"Creating the dataloader using original data (ground truth) and new training data ({data_type}) from {new_data_path_train}.")
                train_data = get_new_data(mixed_train_data_path,args,numimgs=numimgs)
                
                if genTest: 
                    mixed_test_data_path , numimgs = merge_training_lists(args, org_files = args.org_data_test , new_data_path = new_data_path_test )        
                    _log_it(args,f"Creating the dataloader using original data (ground truth) and new testing data ({data_type}) from {new_data_path_test}.")
                    merged_test_dataloader  = get_new_data(mixed_test_data_path ,args,numimgs=numimgs) #FIXME:
            
                
            except Exception as e:
                _log_it(args,f"Trouble loading the pseudo labels.\n\tError: {e}")
        else:
            _log_it(args,f"Continue with original data only.")
            return train_data, merged_test_dataloader, new_test_dataloader_gt
        
        args.flag_new_data = 1
        args.new_train_data = train_data
        if genTest: args.merged_test_data = merged_test_dataloader
        if genTest: args.new_test_data_GT = new_test_dataloader_gt
        return train_data, merged_test_dataloader, new_test_dataloader_gt
            
    def _log_it(args,msg):
        _logging = logging.getLogger("client_logger")
        args.curr_step+=1
        __msg =   f"Step {args.curr_step}:  -- Round {args.round_idx}:  -- Client {args.client_id} -- str(msg)"
        args.curr_step+=1
        _logging.info(__msg)
        logging.info(__msg)
        msg =   "\n\t"                                                                +\
                colorstr( "bright_green"  , "bold" , f"Step {args.curr_step}: "      )+\
                colorstr( "bright_blue"   , "bold" , f"Round {args.round_idx}: "     )+\
                colorstr( "bright_cyan"   , "bold" , f"Client {args.client_id}\n\t"  )+\
                colorstr( args.color_str  , "bold" , str(msg)                        )+\
                "\n"
        LOGGER.info(msg)
        logging.info(msg)
        # args.logging.info(msg)
        
    def get_X_GTs_train(args):
        import random
        tmp_gt_files_train = args.tmp_gt_files_train.split('.')[0]+f"_train_client{args.client_id}.txt"
        _f = open(tmp_gt_files_train, 'w')
        data=check_dataset(args.new_data_conf)['train']
        [_f.write(x) for x in random.sample(population=open(data,'r').readlines(), k=args.new_data_num_images_train)]
        _f.close()
        return tmp_gt_files_train
    
    def get_X_GTs_test(args):
        import random
        tmp_gt_files_test = args.tmp_gt_files_test.split('.')[0]+f"_test_client{args.client_id}.txt"
        __f = open(tmp_gt_files_test, 'w')
        data=check_dataset(args.new_data_conf)['val']
        [__f.write(x) for x in random.sample(population=open(args.new_data_test,'r').readlines(), k=args.new_data_num_images_test)]
        __f.close()
        return tmp_gt_files_test
        
    def partition_data(data_path,total_num=[],shuffle=True):
        if os.path.isfile(data_path):
            with open(data_path) as f:
                data = sorted(f.readlines())
            n_data = len(data)
        else:
            n_data = len(os.listdir(data_path))
        if total_num==[]:
            total_num = n_data
        if shuffle:
            idxs = np.random.permutation(total_num)
        else:
            idxs    = np.random.permutation(n_data-total_num)
            _start  = idxs[np.random.permutation(total_num)[0]]
            _idxs   = [] 
            for i in range(_start,_start+total_num):
                _idxs.append(i)
            idxs    = np.array(_idxs)
            print(f"\tTaking indices from {_start} to {_start+total_num}")
        batch_idxs = np.array_split(idxs, 1)
        net_dataidx_map = {i: batch_idxs[i] for i in range(1)}
        return net_dataidx_map


    def copy_images(args,target):
        new_files=[]
        merged_dir = os.path.abspath(os.path.split(args.path_low)[0])
        target = os.path.abspath(target)
        for _file in os.listdir(merged_dir):
            if _file.endswith('.jpg'):
                shutil.copyfile(os.path.join(merged_dir,_file), os.path.join( target, os.path.basename(_file)) )
                

    def move_images(args,srcPath,target):
        new_files=[]
        merged_dir = os.path.abspath(os.path.split(srcPath)[0])
        target = os.path.abspath(target)
        for _file in os.listdir(merged_dir):
            if _file.endswith('.jpg'):
                shutil.move(os.path.join(merged_dir,_file), os.path.join( target, os.path.basename(_file)) )
                
                
    def merge_training_lists(args, org_files = [], new_data_path = []):
        
        if org_files==[]:
            org_files = args.org_data, new_data_path = args.new_pseudos_recovered
        
        isfile = os.path.isfile(new_data_path)
        
        if isfile:
            with open(new_data_path) as f:
                new_files = f.readlines()
        else:
            # new_files = os.listdir(new_data_path)
            new_files=[]
            for _file in os.listdir(new_data_path):
                if _file.endswith('.jpg'):
                    new_files.append(os.path.join(os.path.abspath(new_data_path),_file))
        
        _log_it(args,f"Merging {len(org_files)} images from {args.data_conf} with {len(new_files)} images from {new_data_path}.")
        tmp_merge_file = args.tmp_merge_file.split('.')[0]+f"_client{args.client_id}.txt"
        with open(tmp_merge_file , 'w') as outfile:
            for line in org_files:
                outfile.write(line+'\n')
            for line in new_files:
                outfile.write(line+'\n')
                # outtext = os.path.join(new_data_path,line)+'\n'
                # outfile.write(os.path.abspath(outtext))
                
        numimgs = len(org_files)+len(new_files)
        return tmp_merge_file, numimgs

    def get_new_data(data_path,args,numimgs=0,shuffle=True):
        # data_path            = args.new_data
        imgsz_test           = args.new_dataloader_args[0]
        total_batch_size     = args.new_dataloader_args[1]
        gs                   = args.new_dataloader_args[2]
        args                 = args.new_dataloader_args[3]
        hyp                  = args.new_dataloader_args[4]
        rect                 = args.new_dataloader_args[5]
        rank                 = -1
        pad                  = args.new_dataloader_args[7]
        workers              = args.new_dataloader_args[8]
        total_num            = args.new_data_num_images_train if numimgs==0 else numimgs
        net_dataidx_map_test = partition_data(data_path,total_num=total_num,shuffle=shuffle)
        testloader           = create_dataloader(
                                                data_path,
                                                imgsz_test,
                                                total_batch_size,
                                                gs,
                                                args,  # testloader
                                                hyp=hyp,
                                                rect=rect,
                                                rank=rank,
                                                pad=pad,
                                                net_dataidx_map=net_dataidx_map_test[0],
                                                workers=workers,
                                                )[0]
        return testloader


    def modify_dataset(dataloader,new_path):
        
        count_missing_file = 0
        # Replace GT with Pseudos
        for i, _label_file in enumerate(dataloader.dataset.label_files):
            
            # Path for new label file
            new_label_file = os.path.join ( os.path.realpath(new_path), os.path.basename(_label_file) )
            
            # Make file if there are no pseudos for this file
            if not os.path.isfile(new_label_file): 
                open(new_label_file, 'w')
                count_missing_file+=1
                
            # Read labels from pseudo file
            _label_old    = dataloader.dataset.labels[i]
            _label_pseudo = np.array([x.split() for x in open(new_label_file, 'r').readlines()],dtype='float32')
            
            # Replace labels in dataset
            dataloader.dataset.labels[i] = _label_pseudo
            
            # Replace label file address in dataset
            dataloader.dataset.label_files[i] = new_label_file
            
        return dataloader

    def pseudo_labels(data,model,dataloader,compute_loss,args,conf_thresh=0.5,save_conf=True, save_dir=[]):
        """
        Generate pseudo labels
        """
        
        epoch_no        =   args.round_idx
        half            =   False
        single_cls      =   False
        plots           =   False
        host_id         =   args.client_id
        batch_size      =   args.batch_size
        imgsz           =   args.img_size[0]
        
        if save_dir==[]: save_dir=args.save_dir
        
        conf = 'low' if conf_thresh<0.5 else 'high'   # if 'low', images will also be copied with the generated labels
        
        _log_it(args,f"Generating {conf} confidence labels at {conf_thresh} threshold.")
        
        results, maps, _t, _file =  pseudos.run(data            = data          ,
                                                batch_size      = batch_size    ,
                                                imgsz           = imgsz         ,
                                                half            = half          ,
                                                model           = model         ,
                                                single_cls      = single_cls    ,
                                                dataloader      = dataloader    ,
                                                save_dir        = save_dir      ,
                                                plots           = plots         ,
                                                compute_loss    = compute_loss  ,
                                                
                                                save_txt        = True          ,
                                                save_conf       = save_conf     ,
                                                epoch_no        = epoch_no      ,
                                                host_id         = host_id       ,
                                                conf_thres      = conf_thresh   ,   
                                                confidence      = conf                                     
                                                )
        
        if _file ==[]:
            _log_it(args,f"No labels generated due to poor predictions. Model needs more training.")
            
        return _file

    def recover_labels(args,low_conf_pred_path):

        
        from pathlib import Path            
        class opt_recovery(object):

            agnostic_nms        = False
            augment             = False
            classes             = None
            config_deepsort     = args.new_deepsort_config
            device              = args.device.index
            dnn                 = False
            evaluate            = False
            fourcc              = 'mp4v'
            half                = False
            imgsz               = args.new_dataloader_args[0]
            max_hc_boxes        = 1000
            name                = 'Recover'
            project             = Path('runs/track')
            save_img            = False
            save_vid            = False
            show_vid            = False
            visualize           = False
            
            deep_sort_model     = "resnet50_MSMT17"
            yolo_model          = []
            low_conf_thres      = args.conf_thresh_low
            high_conf_thres     = args.conf_thresh_high
            iou_thres           = args.iou_thres
            save_txt            = True
            exist_ok            = True
            reverse             = False
            
            epoch_no            =   0
            source              =   os.path.split(low_conf_pred_path)[0]
            host_id             =   args.client_id
            
            # source = save_dir / 'labels' 
            # source = source / f'Trainer_{host_id}--epoch_{epoch_no}'
            # source = source / f'low_0.01'
            # source = source / f'high_0.5'
            # source.mkdir(parents=True, exist_ok=True)
            source = str(source)
            output = source
            # txt_path = str(Path(opt.output)) + '/' + os.path.basename(path)[:-4]+'.txt'
        try:
            # Recover in Forward
            opt_recovery.output = os.path.split( opt_recovery.source )[0] +'/Recover-FW'
            opt_recovery.reverse= False
            with torch.no_grad():
                _log_it(args,f"Performing pseudo label recovery in Forward direction at {opt_recovery.output} on device {opt_recovery.device}.")
                recover.detect(opt_recovery,args)     
        except Exception as e:    
            print(f"Error in FW Recovery - {e}")

        try:
            # Recover in Backward
            opt_recovery.output = os.path.split( opt_recovery.source )[0] +'/Recover-BW'
            opt_recovery.reverse= True
            with torch.no_grad():
                _log_it(args,f"Performing pseudo label recovery in Backward direction at {opt_recovery.output} on device {opt_recovery.device}.")
                recover.detect(opt_recovery,args)
        except Exception as e:    
            print(f"Error in BW Recovery - {e}")
        
        try:
            # Merge
            class opt_merge(object):
                forward     = os.path.split( opt_recovery.source )[0]+'/Recover-FW'
                backward    = os.path.split( opt_recovery.source )[0]+'/Recover-BW'
                merged      = os.path.split( opt_recovery.source )[0]+'/Recover-Merged'
            _log_it(args,f"Merge the results from Yolo's pseudo labels with the recovered labels from forward and backward recovery.")
            merge(opt_merge)
        except Exception as e:    
            print(f"Error in Merging - {e}")
        
        _log_it(args,f"Move the images to the Recover-Merged directory for next training.")
        move_images(args,low_conf_pred_path,opt_merge.merged)
            
        return opt_merge.merged
    
def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where(
            (iou >= iouv[i]) & correct_class
        )  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


class YOLOv5Trainer(ClientTrainer):
    def __init__(self, model, args=None):
        super(YOLOv5Trainer, self).__init__(model, args)
        self.hyp = args.hyp
        self.args = args
        self.round_loss = []
        self.round_idx = 0
        self.ValOn = self.args.data_desc[self.args.val_on]
        self._best_model=[]
        self._best_model_score=[]
        self._best_model_epochNo = 0
        
        self.last_model_state = None
        
        args.client_map50_all = dict()
        args.client_map_all   = dict()
        args.client_map50_all[args.rank]=0
        args.client_map_all[args.rank]=0
        self.logging = logging.getLogger("client_logger")
        
        hyp = self.hyp if self.hyp else self.args.hyp
        epochs = args.epochs  # number of epochs

        # % =============== Adding parameter groups in Optimizer ==========================
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
        if args.client_optimizer == "adam":
            self.optimizer = optim.Adam(pg0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999))  # adjust beta1 to momentum
        else:
            self.optimizer = optim.SGD(pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)

        self.optimizer.add_param_group({"params": pg1, "weight_decay": hyp["weight_decay"]})  # add pg1 with weight_decay
        self.optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
        logging.info("Optimizer groups: %g .bias, %g conv.weight, %g other" % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2



        # % ============================= Freeze Layers ====================================
        freeze = []  # parameter names to freeze (full or partial)
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print("freezing %s" % k)
                v.requires_grad = False



        # % =============== Define Learning Rate scheduler based on total epochs ==================
        total_epochs = epochs * args.comm_round
        # if not args.weights == "none": total_epochs += args.last_epochs
        if not args.weights == "none": total_epochs += 300
        lf = (lambda x: ((1 + math.cos(x * math.pi / total_epochs)) / 2) * (1 - hyp["lrf"]) + hyp["lrf"] )  # cosine
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        # scheduler = lr_scheduler.ConstantLR(self.self.optimizer, lr_lambda=lf)
        if not args.weights == "none":
            for _i_ in tqdm(range(300), desc='Running for learning rate decrease', leave=True):
                self.scheduler.step()




    def get_model_params_gpu(self):
        return self.model.state_dict()
    
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, test_data, device, args):
        try:
            self.round_idx = args.round_idx
            args = self.args
            
            # =============== Logging information ==========================
            host_id = int(list(args.client_id_list)[1])
            args.client_id = host_id
            logging.info("Start training on Trainer {}".format(host_id))
            logging.info(f"Hyperparameters: {self.hyp}, Args: {self.args}")
            LOGGER.info(colorstr('hyperparameters: ')+ ', '.join(f'{k}={v}' for k, v in self.hyp.items()))  ############
            logging.info(colorstr('hyperparameters: ')+ ', '.join(f'{k}={v}' for k, v in self.hyp.items()))  ############
            model = self.model
            
            # =============== Model Loss and Move model to device ======================
            model.to(device)
            model.train()
            compute_loss = ComputeLoss(model)
            
            

            # %% ======================= Validation Part - Before Training ====================================

            # LOGGER.info(colorstr("bright_green","bold", f"\n\n\n\tValidating on client {args.client_id} before starting training.\n"))
            
            # logging.info("Start val on Trainer before training {}".format(host_id))

            # test_data_list_before = [test_data              , args.test_dataloader_merged           , args.test_dataloader_new          ]
            # test_data_desc_before = ["Before_Training_"     ,"Before_Training_Merged_TestD_"        ,"Before_Training_New_TestD_"       ]
            
            # for val_data,data_desc in zip(test_data_list_before, test_data_desc_before):
                    
            #     if val_data==[]:
            #         pass
            #     else:
            #         if use_shoaib_code: 
            #             LOGGER.info(colorstr("bright_green","bold", f"\tValidating on {len(val_data.dataset.labels)} images {data_desc} from {os.path.split(val_data.dataset.label_files[0])[0]}.\n"))
                        
            #         logging.info(f"Start val on Trainer {host_id} using {val_data}")
            #         self._val(  data=check_dataset(args.data_conf),
            #                     batch_size=args.batch_size,
            #                     imgsz=args.img_size[0],
            #                     half=False,
            #                     model=model,
            #                     single_cls=False,
            #                     dataloader=val_data,
            #                     save_dir=self.args.save_dir,
            #                     plots=False,
            #                     compute_loss=compute_loss, 
            #                     args = args,
            #                     phase= data_desc
            #                     )

            # %% =================== Validation Part - Before Starting Training ====================================
            
            if args.round_idx==0:
                logging.info(colorstr("bright_green","bold", f"\n\tValidating before starting training.\n"))
                LOGGER.info(colorstr("bright_green","bold", f"\n\tValidating before starting training.\n"))
                test_data_list = [test_data         , args.test_dataloader_new      , args.test_dataloader_merged   ]
                test_data_desc = args.data_desc
                
                args.mAPs=dict()
                args.mAPs[args.client_id]=dict()
                for val_data,data_desc in zip(test_data_list, test_data_desc):
                    if val_data==[]:
                        pass
                    else:
                        args.mAPs[args.client_id][data_desc]=dict()
                        if use_shoaib_code: 
                            logging.info(colorstr("bright_green","bold", f"\tValidating on {len(val_data.dataset.labels)} images {data_desc} from {os.path.split(val_data.dataset.label_files[0])[0]}.\n"))
                            LOGGER.info(colorstr("bright_green","bold", f"\tValidating on {len(val_data.dataset.labels)} images {data_desc} from {os.path.split(val_data.dataset.label_files[0])[0]}.\n"))
                            
                        logging.info(f"Start val on Trainer {host_id} using {val_data.dataset.path}")
                        self._val(  data=check_dataset(args.data_conf),
                                    batch_size=args.batch_size,
                                    imgsz=args.img_size[0],
                                    half=False,
                                    model=model,
                                    single_cls=False,
                                    dataloader=val_data,
                                    save_dir=self.args.save_dir,
                                    plots=False,
                                    compute_loss=compute_loss, 
                                    args = args,
                                    phase= data_desc
                                    )
                        
                    
                self._best_model         = copy.deepcopy(model)
                self._best_model_score   = args.mAPs[args.client_id][self.ValOn]
                
                args.curr_step=0
                _log_it(args,"Skip training for round 0")

            else: # After Round 0
                    
                if not args.use_update_from_server:
                    LOGGER.info(colorstr("bright_green","bold", f"\n\tServer is NOT updating the client model weights. Using the model from last epoch.\n"))
                    logging.info(colorstr("bright_green","bold", f"\n\tServer is NOT updating the client model weights. Using the model from last epoch.\n"))
                    self.set_model_params(self.last_model_state)
                else:
                    LOGGER.info(colorstr("bright_green","bold", f"\tR:{self.round_idx} - Server is updating the client model.\n"))
                    logging.info(colorstr("bright_green","bold", f"\tR:{self.round_idx} - Server is updating the client model.\n"))
                
                # %% ======================== Shoaib's part =======================================
                args.client_data_size_org_train = len(train_data.dataset.labels)
                args.client_data_size_org_test = len(test_data.dataset.labels)
                args.logging = logging
                args.flag_new_data = 0
                if use_shoaib_code: # TODO: 
                    try:
                        train_data, test_data_with_pseudo, new_test_data_gt = use_new_data(args,model,compute_loss,train_data, test_data)
                    except Exception as e:
                        print(f"Some error \n\t{e}")
                        args.flag_new_data = 0
                args.client_data_size_mod_train = len(train_data.dataset.labels)
                # if args.round_idx==1 and args.rank==args.psuedo_gen_on_clients[0]: fedml.core.mlops.mlops_profiler_event.MLOpsProfilerEvent.log_to_wandb(args.__dict__)
                
                
                
                
            
                # %%
                # =================================================================================
                # ============================== Training Part ====================================
                # =================================================================================
                mloss = torch.zeros(3, device=device)  # mean losses
                for epoch in range(args.epochs):
                    epoch_loss = []
                    model.train()
                    t = time.time()
                    batch_loss = []
                    logging.info("\tTrainer_ID: {0}, Epoch: {1}".format(host_id, epoch))
                    
                    if use_shoaib_code: 
                        logging.info(colorstr("bright_green","bold", f"\n\tR:{self.round_idx} - Training on {args.client_data_size_mod_train} images. Original training data is {args.client_data_size_org_train} for client {args.client_id}.\n"))
                        LOGGER.info(colorstr("bright_green","bold", f"\n\tR:{self.round_idx} - Training on {args.client_data_size_mod_train} images. Original training data is {args.client_data_size_org_train} for client {args.client_id}.\n"))
                    
                    logging.info(("%10s" * 8) % ("Epoch", "batch", "gpu_mem", "box", "obj", "cls", "targets", "img_size"))
                    total_batches = train_data.batch_sampler.sampler.sampler.data_source.batch_shapes.shape[0]
                    for (batch_idx, batch) in enumerate(train_data):
                        # ============= Read images and move to GPU ===================
                        imgs, targets, paths, _ = batch
                        imgs = imgs.to(device, non_blocking=True).float() / 256.0 - 0.5


                        # ============= Clear the previous gradient values ============
                        self.optimizer.zero_grad()
                        
                        
                        # ========================== Forward ==========================
                        # with torch.cuda.amp.autocast(amp):
                        pred = model(imgs)  # forward
                        loss, loss_items = compute_loss(pred, targets.to(device).float())  # loss scaled by batch_size


                        # ========================== Backward ==========================
                        loss.backward()
                        
                        
                        # ======================= Update weights =======================
                        self.optimizer.step()
                        batch_loss.append(loss.item())


                        # ====================== Calculate Losses ======================
                        mloss = (mloss * batch_idx + loss_items) / (batch_idx + 1)  # update mean losses
                        mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)  # (GB)
                        s = ("%10s" * 3 + "%10.4g" * 5) % ("%g/%g" % (epoch, args.epochs - 1), "%g/%g" % (batch_idx, total_batches - 1),mem,*mloss,targets.shape[0],imgs.shape[-1])
                        print(s)
                        
                    args.RoundxEpoch = ( (args.round_idx-1) * (args.epochs) ) + (epoch) + 1
                    
                
                    # self.last_model_state = copy.deepcopy(model)    
                    self.last_model_state = self.get_model_params_gpu()    
                        
                    # ============= Update learning rate =============
                    logging.info(s)
                    self.scheduler.step()





                    # %% ============================== Results Logging ==================================
                    epoch_loss.append(copy.deepcopy(mloss.cpu().numpy()))
                    logging.info(f"Trainer {host_id} epoch {epoch} box: {mloss[0]} obj: {mloss[1]} cls: {mloss[2]} total: {mloss.sum()} time: {(time.time() - t)}")

                    logging.info("#" * 20)

                    try: logging.info(f"Trainer {host_id} epoch {epoch} time: {(time.time() - t)}s batch_num: {batch_idx} speed: {(time.time() - t)/batch_idx} s/batch")
                    except: pass
                    logging.info("#" * 200)
                    
                    MLOpsProfilerEvent.log_to_wandb({
                                                    # f"client_{host_id}_round_idx":          self.round_idx,
                                                    # f"client_{host_id}_train_box_loss":     np.float(mloss[0]),
                                                    # f"client_{host_id}_train_obj_loss":     np.float(mloss[1]),
                                                    # f"client_{host_id}_train_cls_loss":     np.float(mloss[2]),
                                                    # f"client_{host_id}_train_total_loss":   np.float(mloss.sum()),
                                                    f"train_box_loss":      np.float(mloss[0]),
                                                    f"train_obj_loss":      np.float(mloss[1]),
                                                    f"train_cls_loss":      np.float(mloss[2]),
                                                    f"train_total_loss":    np.float(mloss.sum()),
                                                    f"Original Number of Training Images":       args.client_data_size_org_train,
                                                    f"Modified Number of Training Images":       args.client_data_size_mod_train,
                                                    f"Learning Rate":       self.scheduler.get_last_lr()[0],
                                                    f"Round_No":            args.round_idx,
                                                    f"Epoch_No":            epoch,
                                                    f"Round x Epoch":       args.RoundxEpoch,
                                                    })



                    # %% ============================== Saving Weights ===================================
                    if (epoch + 1) % self.args.checkpoint_interval == 0:
                        model_path = (self.args.save_dir/ "weights"/ f"model_client_{host_id}_round_{self.round_idx}_epoch_{epoch}.pt")
                        logging.info(f"Trainer {host_id} epoch {epoch} saving model to {model_path}")
                    
                        # Old saving method     # torch.save(model.state_dict(), model_path)
                        # Modified saving method
                        ckpt = {'epoch': epoch,
                                'model': copy.deepcopy(model).half(),
                                'self.optimizer': self.optimizer.state_dict()}
                        torch.save(ckpt, model_path)
                        del ckpt





                    # ============================== Logging Results ===================================
                    epoch_loss = np.array(epoch_loss)
                    logging.info(f"Epoch loss: {epoch_loss}")

                    fedml.mlops.log({
                                    f"train_box_loss":      np.float(epoch_loss[-1, 0]),
                                    f"train_obj_loss":      np.float(epoch_loss[-1, 1]),
                                    f"train_cls_loss":      np.float(epoch_loss[-1, 2]),
                                    f"train_total_loss":    np.float(epoch_loss[-1, :].sum()),
                                    f"Round_No":            args.round_idx,
                                    f"Round x Epoch":       args.RoundxEpoch,
                                    })

                    self.round_loss.append(epoch_loss[-1, :])
                    if self.round_idx == args.comm_round:
                        self.round_loss = np.array(self.round_loss)
                        logging.info(f"Trainer {host_id} round {self.round_idx} finished, round loss: {self.round_loss}")


                    # %% =================== Validation Part - After Training ====================================
                    if args.ValClientsOnServer:
                        LOGGER.info(colorstr("bright_red","bold", f"\tR/E: {self.round_idx}/{epoch} - Client model will be evaluated only on Server.\n"))
                        logging.info(colorstr("bright_red","bold", f"\tR/E: {self.round_idx}/{epoch} - Client model will be evaluated only on Server.\n"))
                    
                    else:
                        test_data_list = [test_data         , args.test_dataloader_new      , args.test_dataloader_merged   ]
                        test_data_desc = args.data_desc
                        args.mAPs=dict()
                        args.mAPs[args.client_id]=dict()
                        for val_data,data_desc in zip(test_data_list, test_data_desc):
                            if val_data==[]:
                                pass
                            else:
                                args.mAPs[args.client_id][data_desc]=dict()
                                if use_shoaib_code: 
                                    logging.info(colorstr("bright_green","bold", f"\tR/E: {self.round_idx}/{epoch} - Validating on {len(val_data.dataset.labels)} images {data_desc} from {os.path.split(val_data.dataset.label_files[0])[0]}.\n"))
                                    LOGGER.info(colorstr("bright_green","bold", f"\tR/E: {self.round_idx}/{epoch} - Validating on {len(val_data.dataset.labels)} images {data_desc} from {os.path.split(val_data.dataset.label_files[0])[0]}.\n"))
                                    
                                logging.info(f"Start val on Trainer {host_id} using {val_data.dataset.path}")
                                self._val(  data=check_dataset(args.data_conf),
                                            batch_size=args.batch_size,
                                            imgsz=args.img_size[0],
                                            half=False,
                                            model=model,
                                            single_cls=False,
                                            dataloader=val_data,
                                            save_dir=self.args.save_dir,
                                            plots=False,
                                            compute_loss=compute_loss, 
                                            args = args,
                                            phase= data_desc,
                                            )
                                            # conf_thres=0.20,
                        # if args.keep_client_model_history:   
                                    
                        if args.mAPs[args.client_id][self.ValOn][2] > self._best_model_score[2]:
                            self._best_model         = copy.deepcopy(model)
                            self._best_model_score   = args.mAPs[args.client_id][self.ValOn]
                            self._best_model_epochNo = args.RoundxEpoch
                            
                            msg = colorstr( "bright_yellow"  , "bold" , f"\n\t The best weights are given by EpochxRound {args.RoundxEpoch} with mAP: {self._best_model_score[2]}\n" )
                            print(msg)
                            model_path = self.args.save_dir / "weights" / f"model_client_{host_id}_best.pt"
                            logging.info(f"Trainer {host_id} saving model to {model_path}")

                            ckpt = {'model': copy.deepcopy(self._best_model).half(),
                                    'self.optimizer': self.optimizer.state_dict()}
                            torch.save(ckpt, model_path)
                            del ckpt
                        
                    
                
            
            
            if args.keep_client_model_history and not args.ValClientsOnServer:            
                self.set_model_params(self._best_model.state_dict())
                args.mAPs[args.client_id][self.ValOn] = self._best_model_score            

                # %%
                MLOpsProfilerEvent.log_to_wandb({
                                                # f"client_{device.index}_{names[_idx]}_class_ap@50":_ap50 ),
                                                f"Best_Model_in_Epoch": self._best_model_epochNo,
                                                f"Best_Model_in_Epoch_Model_Score": self._best_model_score[2],
                                                f"Round_No": args.round_idx,
                                                f"Round x Epoch": args.RoundxEpoch,
                                                })
                
                
                
                
                
                
            # %% ============================== Saving Weights ===================================
            logging.info("End training on Trainer {}".format(host_id))
            
            model_path = self.args.save_dir / "weights" / f"model_client_{host_id}_round_{self.round_idx}_end.pt"
            logging.info(f"Trainer {host_id} saving model to {model_path}")

            ckpt = {'model': copy.deepcopy(model).half(),
                    'self.optimizer': self.optimizer.state_dict()}
            torch.save(ckpt, model_path)
            del ckpt
            
            
            # self.last_model_state = copy.deepcopy(model)
            self.last_model_state = self.get_model_params_gpu()   
        
        
        except Exception as e:
            print(traceback.format_exc())

        # %% ========================= End Trainer Function ===================================
        return









    def _val(self, 
             data, 
             batch_size, 
             imgsz, 
             half, 
             model, 
             single_cls, 
             dataloader, 
             save_dir, 
             plots, 
             compute_loss, 
             args,
             conf_thres=0.001,
             phase=''):
        
        host_id = int(list(args.client_id_list)[1])
        results, maps, _, ap50 = validate.run(data = data,
                                    batch_size = args.batch_size,#128,
                                    imgsz = imgsz,
                                    half = half,
                                    model = model,
                                    single_cls = single_cls,
                                    dataloader = dataloader,
                                    save_dir = save_dir,
                                    plots = plots,
                                    compute_loss = compute_loss,
                                    conf_thres = conf_thres,)
        args.mAPs[args.client_id][phase] = results
        names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
        for _idx, _ap50 in enumerate(ap50):
            MLOpsProfilerEvent.log_to_wandb({
                                            # f"client_{device.index}_{names[_idx]}_class_ap@50":_ap50 ),
                                            f"{phase}{names[_idx]}_class_ap@50":_ap50 ,
                                            f"Round_No": args.round_idx,       
                                            f"Round x Epoch":       args.RoundxEpoch,
                                            })
        
        
        if args.data_desc[args.val_on] == phase: args.client_map50_all[host_id]  = np.float(results[2])
        if args.data_desc[args.val_on] == phase: args.client_map_all[host_id]    = np.float(results[3])
        
        MLOpsProfilerEvent.log_to_wandb(
                {
                    # f"client_{host_id}_mean_precision":     np.float(results[0]),
                    # f"client_{host_id}_mean_recall":        np.float(results[1]),
                    # f"client_{host_id}_map@50":             np.float(results[2]),
                    # f"client_{host_id}_map":                np.float(results[3]),
                      
                    # f"client_{host_id}_test_box_loss":      np.float(results[4]),
                    # f"client_{host_id}_test_obj_loss":      np.float(results[5]),
                    # f"client_{host_id}_test_cls_loss":      np.float(results[6]),
                    
                    # f"{phase}client_{host_id}_training_data_org":  args.client_data_size_org_train,
                    # f"{phase}client_{host_id}_training_data_mod":  args.client_data_size_mod_train,
                    # f"{phase}client_{host_id}_testing_data_org":   args.client_data_size_org_test,
                    
                    
                    # f"mean_precision":      np.float(results[0]),
                    # f"mean_recall":         np.float(results[1]),
                    # f"map@50_all_classes":  np.float(results[2]),
                    # f"map_all_classes":     np.float(results[3]),
                    
                    f"{phase}mean_precision":      np.float(results[0]),
                    f"{phase}mean_recall":         np.float(results[1]),
                    f"{phase}map@50_all_classes":  np.float(results[2]),
                    f"{phase}map_all_classes":     np.float(results[3]),
                    
                    f"{phase}test_box_loss":       np.float(results[4]),
                    f"{phase}test_obj_loss":       np.float(results[5]),
                    f"{phase}test_cls_loss":       np.float(results[6]),
 
                    # f"{phase}training_data_org":   args.client_data_size_org_train,
                    # f"{phase}training_data_mod":   args.client_data_size_mod_train,
                    # f"{phase}testing_data_org":    args.client_data_size_org_test,
                    
                    f"{phase}Round_No": args.round_idx,
                    f"Round_No": args.round_idx,
                    f"Round x Epoch":       args.RoundxEpoch,
                }
            )
        
        for i,cls in enumerate(check_dataset(args.data_conf)['names']):
            MLOpsProfilerEvent.log_to_wandb({
                                            # f"client_{host_id}_map_{cls}":  maps[i],
                                            # f"map_{cls}":                   maps[i],
                                            f"{phase}map_{cls}":            maps[i],
                                            f"Round_No":                    args.round_idx,
                                            f"Round x Epoch":       args.RoundxEpoch,
                                            })
        
        logging.info(f"mAPs of all class in a list {maps}")

