##General imports
import csv
import os
import time
from datetime import datetime
import shutil
import copy
import numpy as np
from functools import partial
import platform
from sklearn.metrics import mean_squared_error, mean_absolute_error

##Torch imports
import torch.nn.functional as F
import torch
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.nn import DataParallel
import torch_geometric.transforms as T
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

##Matdeeplearn imports
from matdeeplearn import models
from matdeeplearn.process import process_cl, process_nm, process
import matdeeplearn.training as training
from matdeeplearn.models.utils import model_summary

from pytorch_metric_learning.losses import NTXentLoss


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target))
    
    
################################################################################
#  Training functions
################################################################################
##Pretrain step, runs model in train mode
def train_pretrain(model, linear_nm, linear_ve, optimizer, optimizer_nm, optimizer_ve, loader_cl1, loader_cl2, loader_nm, criterion_nm, criterion_ve, criterion_cl, rank):
    
    model.train()
    linear_nm.train()
    linear_ve.train()
    
    loss_all = 0
    acc_node_accum = 0
    acc_ve_accum = 0
    count = 0    
    count_node = 0
    count_ve = 0
    for data_cl1, data_cl2, data_nm in zip(loader_cl1, loader_cl2, loader_nm):
        data_cl1 = data_cl1.to(rank)
        data_cl2 = data_cl2.to(rank)
        data_nm = data_nm.to(rank)
        
        optimizer.zero_grad()
        optimizer_nm.zero_grad()
        optimizer_ve.zero_grad()
        
        ### Masking
        target = data_nm.y
        target = torch.argmax(target, 1)
        
        target_ve = data_nm.y1
        target_ve = torch.argmax(target_ve, 1)
        
        node_pre1, node_cl1, node_cl2 = model(data_nm, data_cl1, data_cl2)
        
        pred_node = linear_nm(node_pre1[data_nm.atom_index])
        pred_ve = linear_ve((node_pre1[data_nm.atom_index]))
        
        # node masking loss
        loss_nm = criterion_nm(pred_node, target) 
        acc_node = compute_accuracy(pred_node, target)
        acc_node_accum += acc_node     
        count_node = count_node + pred_node.size(0)
        
        # valence electron loss
        loss_ve = criterion_ve(pred_ve, target_ve)
        acc_ve = compute_accuracy(pred_ve, target_ve)
        acc_ve_accum += acc_ve     
        #count_ve = count_ve + pred_ve.size(0)
        
        ### Contrastive Learning
        g1_list = []
        g2_list = []
        pre_index = data_cl1.ptr[0]
        for index in data_cl1.ptr:
            if index == 0:
                continue
            g1_list.append(node_cl1[pre_index: index, :])
            g2_list.append(node_cl2[pre_index: index, :])
            pre_index = index
        
        z_g1 = []
        z_g2 = []
        for i in range(len(g1_list)):
            z_g1.append(torch.mean(g1_list[i], dim=0, keepdim=True))
            z_g2.append(torch.mean(g2_list[i], dim=0, keepdim=True))
        
        z_g1 = torch.cat(z_g1, dim=0)
        z_g2 = torch.cat(z_g2, dim=0)
        y = torch.ones(z_g1.size(0)).to(z_g1.device)
        #print(z_g1.shape, z_g2.shape, y.shape)
        
        loss_cl = criterion_cl(z_g1, z_g2, y)
        
        
        loss = 0.4*loss_nm + 0.3*loss_ve + 0.3*loss_cl    # consider
        #loss = 0.5*loss_nm + 0.5*loss_cl
        #loss = 0.75*loss_nm + 0.25*loss_cl
        #loss = 0.25*loss_nm + 0.75*loss_cl
        #loss = loss_nm + 0.2*loss_cl
        
        loss.backward()
      
        
        #print("backward time: ", time-time()-train_start_time)
        optimizer.step()
        optimizer_nm.step()
        optimizer_ve.step()
        
        count = count + node_pre1.size(0)
        loss_all += loss.detach() * node_pre1.size(0)
          
    train_error = loss_all/count
    train_acc_atom = acc_node_accum/count_node
    train_acc_ve = acc_ve_accum/count_node
    
    #print(train_error, train_acc_atom)
    return train_error, train_acc_atom, train_acc_ve


##Evaluation step, runs model in eval mode
def evaluate_pretrain(loader_cl1, loader_cl2, loader_nm, model, linear_nm, linear_ve, criterion_nm, criterion_ve, criterion_cl, rank, out=False):

    model.eval()
    linear_nm.eval()
    linear_ve.eval()
    
    loss_all = 0
    acc_node_accum = 0
    acc_ve_accum = 0
    count = 0
    count_node = 0
    count_ve = 0
    for data_cl1, data_cl2, data_nm in zip(loader_cl1, loader_cl2, loader_nm):
        data_cl1 = data_cl1.to(rank)
        data_cl2 = data_cl2.to(rank)
        data_nm = data_nm.to(rank)
        
        with torch.no_grad():
        
            target = data_nm.y
            target = torch.argmax(target, 1) 
            
            target_ve = data_nm.y1
            target_ve = torch.argmax(target_ve, 1) 
        
            node_pre1, node_cl1, node_cl2 = model(data_nm, data_cl1, data_cl2)
            
            ## Masking node
            pred_node = linear_nm(node_pre1[data_nm.atom_index])
            pred_ve = linear_ve((node_pre1[data_nm.atom_index]))
            
            
            loss_nm = criterion_nm(pred_node, target) 
            
            acc_node = compute_accuracy(pred_node, target)
            acc_node_accum += acc_node
            count_node = count_node +  pred_node.size(0)
            
            
            # valence electron loss
            loss_ve = criterion_ve(pred_ve, target_ve)
            acc_ve = compute_accuracy(pred_ve, target_ve)
            acc_ve_accum += acc_ve     
            

            ### Contrastive Learning
            g1_list = []
            g2_list = []
            pre_index = data_cl1.ptr[0]
            for index in data_cl1.ptr:
                if index == 0:
                    continue
                g1_list.append(node_cl1[pre_index: index, :])
                g2_list.append(node_cl2[pre_index: index, :])
                pre_index = index
            
            z_g1 = []
            z_g2 = []
            for i in range(len(g1_list)):
                z_g1.append(torch.mean(g1_list[i], dim=0, keepdim=True))
                z_g2.append(torch.mean(g2_list[i], dim=0, keepdim=True))
            
            z_g1 = torch.cat(z_g1, dim=0)
            z_g2 = torch.cat(z_g2, dim=0)
            y = torch.ones(z_g1.size(0)).to(z_g1.device)
            #print(z_g1.shape, z_g2.shape, y.shape)
        
            loss_cl = criterion_cl(z_g1, z_g2, y)
            
            ## total loss 
            loss = 0.4*loss_nm + 0.3*loss_ve + 0.3*loss_cl   # consider
            #loss = 0.75*loss_nm + 0.25*loss_cl
            #loss = 0.25*loss_nm + 0.75*loss_cl
            #loss = loss_nm + 0.2*loss_cl
            
            loss_all += loss * node_pre1.size(0)
            count = count + node_pre1.size(0)

    val_error = loss_all / count
    val_acc_atom = acc_node_accum/count_node
    val_acc_ve = acc_ve_accum/count_node

    return val_error, val_acc_atom, val_acc_ve

##Train step, runs model in train mode
def train_finetune(model, optimizer, loader, loss_method, rank):
    model.train()
    loss_all = 0
    count = 0
    for data in loader:
        data = data.to(rank)
        optimizer.zero_grad()
        
        output = model(data)
        ## nihang: struct embedding
        #output, _ = model(data)
        
        # print(data.y.shape, output.shape)
        loss = getattr(F, loss_method)(output, data.y)
        loss.backward()
        loss_all += loss.detach() * output.size(0)

        optimizer.step()
        count = count + output.size(0)

    loss_all = loss_all / count
    return loss_all


##Evaluation step, runs model in eval mode
def evaluate_finetune(loader, model, loss_method, rank, out=False):

    model.eval()
    loss_all = 0
    count = 0
    for data in loader:
        data = data.to(rank)
        with torch.no_grad():
            output = model(data)
            ## nihang: struct embedding
            #output, struct_emb = model(data)
            
            #print(struct_emb.shape)
            # loss = getattr(F, loss_method)(output, data.y)
            loss = mean_absolute_error(output.cpu().numpy(), data.y.cpu().numpy())
            loss_all += loss * output.size(0)
            if out == True:  # save out
                if count == 0:
                    ids = [item for sublist in data.structure_id for item in sublist]
                    ids = [item for sublist in ids for item in sublist]
                    predict = output.data.cpu().numpy()
                    target = data.y.cpu().numpy()
                    
                    ## nihang: struct embedding
                    #struct_feature = struct_emb.cpu().numpy()
                else:
                    ids_temp = [
                        item for sublist in data.structure_id for item in sublist
                    ]
                    ids_temp = [item for sublist in ids_temp for item in sublist]
                    ids = ids + ids_temp
                    
                    predict = np.concatenate(
                        (predict, output.data.cpu().numpy()), axis=0
                    )
                    target = np.concatenate((target, data.y.cpu().numpy()), axis=0)
                    
                    ## nihang: struct embedding
                    #struct_feature = np.concatenate((struct_feature, struct_emb.cpu().numpy()), axis=0)

            count = count + output.size(0)

    loss_all = loss_all / count

    if out == True:
        test_out = np.column_stack((ids, target, predict))
        ## nihang: struct embedding
        #test_out = np.column_stack((ids, target, predict, struct_feature))
        return loss_all, test_out
    elif out == False:
        return loss_all
        
##Model trainer
def trainer(
    rank,
    world_size,
    model,
    linear_nm,
    linear_ve,
    optimizer,
    optimizer_nm,
    optimizer_ve,
    scheduler,
    loss_nm,
    loss_ve,
    loss_cl,
    train_loader_cl1,
    train_loader_cl2,
    train_loader_nm,
    val_loader_cl1,
    val_loader_cl2,
    val_loader_nm,
    epochs,
    verbosity,
    filename = "my_model_temp.pth",
):

    train_error = val_error = test_error = epoch_time = float("NaN")
    train_start = time.time()
    best_val_error = 1e10
    model_best = model
    best_val_error = np.inf   ## nihang modified. Set inf for the initial valuate error
    
    num_epoch = []
    num_train_error = []
    num_val_error = []
    num_train_acc_node = []
    num_train_acc_ve = []
    num_val_acc_node = []
    num_val_acc_ve = []
    
    ##Start training over epochs loop
    for epoch in range(1, epochs + 1):
        
        lr = scheduler.optimizer.param_groups[0]["lr"]
        #if rank not in ("cpu", "cuda"):
        #    train_sampler.set_epoch(epoch)
        
        ##Train model
        train_error, train_acc_atom, train_acc_ve = train_pretrain(model, linear_nm, linear_ve, optimizer, optimizer_nm, optimizer_ve, train_loader_cl1, train_loader_cl2, train_loader_nm, loss_nm, loss_ve, loss_cl, rank=rank)
        
        #if rank not in ("cpu", "cuda"):
        #    torch.distributed.reduce(train_error, dst=0)
        #    train_error = train_error / world_size

        ##Get validation performance
        if rank not in ("cpu", "cuda"):
            dist.barrier()
            

        val_error, val_acc_atom, val_acc_ve = evaluate_pretrain(val_loader_cl1, val_loader_cl2, val_loader_nm, model.module, linear_nm, linear_ve, loss_nm, loss_ve, loss_cl, rank=rank, out=False)
        #print('valuate time: ', time.time()-testtest_time)
        num_val_error.append(val_error)
        num_val_acc_node.append(val_acc_atom)
        num_val_acc_ve.append(val_acc_ve)
        
        '''
        if val_loader != None and rank in (0, "cpu", "cuda"):
            if rank not in ("cpu", "cuda"):
                val_error = evaluate(
                    val_loader, model.module, loss, rank=rank, out=False
                )
            else:
                val_error = evaluate(val_loader, model, loss, rank=rank, out=False)
        '''
        
        ##Train loop timings
        epoch_time = time.time() - train_start
        train_start = time.time()

        ##scheduler on train error
        scheduler.step(train_error)

        num_epoch.append(epoch)
        num_train_error.append(train_error)       
        num_train_acc_node.append(train_acc_atom)
        num_train_acc_ve.append(train_acc_ve)
        
        ##Print performance
        if epoch % int(verbosity) == 0:
            if rank in (0, "cpu", "cuda"):
                print(
                    "Epoch: {:04d}, Learning Rate: {:.5f}, Training Error: {:.4f}, Validation Error: {:.4f},Train Acc of Atom: {:.4f}, Val Acc of Atom: {:.4f}, Train Acc of VE: {:.4f}, Val Acc of VE: {:.4f}, Time per epoch (s): {:.3f}".format(
                        epoch, lr, train_error, val_error, train_acc_atom, val_acc_atom, train_acc_ve, val_acc_ve, epoch_time
                    )
                )
          
            ##remember the best val error and save model and checkpoint        
            if val_loader_cl1 != None and rank in (0, "cpu", "cuda"):
                if val_error == float("NaN") or val_error < best_val_error:
                    if rank not in ("cpu", "cuda"):
                        model_best = copy.deepcopy(model.module)
                        torch.save(
                            {
                                "state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "full_model": model,
                            },
                            filename,
                        )
                    else:
                        model_best = copy.deepcopy(model)
                        torch.save(
                            {
                                "state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "full_model": model,
                            },
                            filename,
                        )
                best_val_error = min(val_error, best_val_error)
      

            '''      
            ## nihang modify    
            if rank not in ("cpu", "cuda"):
                model_best = copy.deepcopy(model.module)
                torch.save(
                                {
                                    "state_dict": model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "scheduler_state_dict": scheduler.state_dict(),
                                    "full_model": model,
                                },
                                filename,
                            )
            else:
                model_best = copy.deepcopy(model)
                torch.save(
                            {
                                "state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "full_model": model,
                            },
                            filename,
                        )
            '''
                        
    if rank not in ("cpu", "cuda"):
        dist.barrier()

    return model_best, num_epoch, num_train_error, num_val_error, num_train_acc_node, num_val_acc_node, num_train_acc_ve, num_val_acc_ve

##Model trainer
def trainer_finetune(
    rank,
    world_size,
    model,
    optimizer,
    scheduler,
    loss,
    train_loader,
    val_loader,
    train_sampler,
    epochs,
    verbosity,
    filename = "my_model_ft_temp.pth",
):

    train_error = val_error = test_error = epoch_time = float("NaN")
    train_start = time.time()
    best_val_error = 1e10
    model_best = model
    
    num_epoch = []
    num_train_error = []
    num_val_error = []
    
    ##Start training over epochs loop
    for epoch in range(1, epochs + 1):
        
        lr = scheduler.optimizer.param_groups[0]["lr"]
        if rank not in ("cpu", "cuda"):
            train_sampler.set_epoch(epoch)
        ##Train model
        train_error = train_finetune(model, optimizer, train_loader, loss, rank=rank)
        if rank not in ("cpu", "cuda"):
            torch.distributed.reduce(train_error, dst=0)
            train_error = train_error / world_size

        ##Get validation performance
        if rank not in ("cpu", "cuda"):
            dist.barrier()
        if val_loader != None and rank in (0, "cpu", "cuda"):
            if rank not in ("cpu", "cuda"):
                val_error = evaluate_finetune(
                    val_loader, model.module, loss, rank=rank, out=False
                )
            else:
                val_error = evaluate_finetune(val_loader, model, loss, rank=rank, out=False)

        ##Train loop timings
        epoch_time = time.time() - train_start
        train_start = time.time()

        ##remember the best val error and save model and checkpoint        
        if val_loader != None and rank in (0, "cpu", "cuda"):
            if val_error == float("NaN") or val_error < best_val_error:
                if rank not in ("cpu", "cuda"):
                    model_best = copy.deepcopy(model.module)
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "full_model": model,
                        },
                        filename,
                    )
                else:
                    model_best = copy.deepcopy(model)
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "full_model": model,
                        },
                        filename,
                    )
            best_val_error = min(val_error, best_val_error)
        elif val_loader == None and rank in (0, "cpu", "cuda"):
            if rank not in ("cpu", "cuda"):
                model_best = copy.deepcopy(model.module)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    filename,
                )
            else:
                model_best = copy.deepcopy(model)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    filename,
                )

        ##scheduler on train error
        scheduler.step(train_error)

        num_epoch.append(epoch)
        num_train_error.append(train_error)
        num_val_error.append(val_error)
        
        ##Print performance
        if epoch % verbosity == 0:
            if rank in (0, "cpu", "cuda"):
                print(
                    "Epoch: {:04d}, Learning Rate: {:.6f}, Training Error: {:.5f}, Val Error: {:.5f}, Time per epoch (s): {:.5f}".format(
                        epoch, lr, train_error, val_error, epoch_time
                    )
                )

    if rank not in ("cpu", "cuda"):
        dist.barrier()

    return model_best, num_epoch, num_train_error, num_val_error
        
        
##Write results to csv file
def write_results(output, filename):
    shape = output.shape
    with open(filename, "w") as f:
        csvwriter = csv.writer(f)
        for i in range(0, len(output)):
            if i == 0:
                csvwriter.writerow(
                    ["ids"]
                    + ["target"] * int((shape[1] - 1) / 2)
                    + ["prediction"] * int((shape[1] - 1) / 2)
                )
            elif i > 0:
                csvwriter.writerow(output[i - 1, :])


##Pytorch ddp setup
def ddp_setup(rank, world_size):
    if rank in ("cpu", "cuda"):
        return
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    if platform.system() == 'Windows':
        dist.init_process_group("gloo", rank=rank, world_size=world_size)    
    else:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True


##Pytorch model setup
def model_setup(
    rank,
    model_name,
    model_params,
    dataset,
    load_model=False,
    model_path=None,
    print_model=True,
    input_model_file="",
):
    model = getattr(models, model_name)(
        data=dataset, **(model_params if model_params is not None else {})
    ).to(rank)
    
    #if load_model == "True":
    #    assert os.path.exists(model_path), "Saved model not found"
    #    if str(rank) in ("cpu"):
    #        saved = torch.load(model_path, map_location=torch.device("cpu"))
    #    else:
    #        saved = torch.load(model_path)
    #    model.load_state_dict(saved["model_state_dict"])
        # optimizer.load_state_dict(saved['optimizer_state_dict'])
    
    if input_model_file != "":   # if no input_model_file, we will do finetune
        print("Loading pretrained model---------------------------")
        model.from_pretrained(input_model_file)
        print("Load Success!")
        
    # DDP
    if rank not in ("cpu", "cuda"):
        model = DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True
        )
        # model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
    if print_model == True and rank in (0, "cpu", "cuda"):
        model_summary(model)
    
    #print("return modellllll")    
    return model


##Pytorch loader setup
def loader_setup(
    batch_size,
    dataset,
    rank,
    seed,
    world_size=0,
    num_workers=0,
):

    ##DDP
    if rank not in ("cpu", "cuda"):
        train_sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank
        )
    elif rank in ("cpu", "cuda"):
        train_sampler = None

    ##Load data
    train_loader = None
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
   
    return (
        train_loader,
        train_sampler
    )

def loader_setup_cv(index, batch_size, dataset, rank, world_size=0, num_workers=0):
    ##Split datasets
    train_dataset = [x for i, x in enumerate(dataset) if i != index]
    train_dataset = torch.utils.data.ConcatDataset(train_dataset)
    test_dataset = dataset[index]

    ##DDP
    if rank not in ("cpu", "cuda"):
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
    elif rank in ("cpu", "cuda"):
        train_sampler = None

    train_loader = val_loader = test_loader = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    if rank in (0, "cpu", "cuda"):
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, test_loader, train_sampler, train_dataset, test_dataset
    
    
################################################################################
#  Trainers
################################################################################

###Regular training with train, val, test split
def train_regular(
    rank,
    world_size,
    data_path,
    job_parameters=None,
    training_parameters=None,
    model_parameters=None,
): 
    
    ##DDP
    ddp_setup(rank, world_size)
    ##some issues with DDP learning rate
    
    if rank not in ("cpu", "cuda"):
        model_parameters["lr"] = model_parameters["lr"] * world_size

    ##Get dataset
    dataset_cl1 = process_cl.get_dataset('data_cl1', data_path, training_parameters["target_index"], False)
    dataset_cl2 = process_cl.get_dataset('data_cl2', data_path, training_parameters["target_index"], False)    
    dataset_nm = process_nm.get_dataset(data_path, training_parameters["target_index"], False)
    
    ##Split datasets: for pretraining stage, there is no need to add test set.
    train_cl1, val_cl1, test_cl1 = process.split_data(
        dataset_cl1, training_parameters["train_ratio"], training_parameters["val_ratio"],training_parameters["test_ratio"], job_parameters["seed"]
    )
    train_cl2, val_cl2, test_cl2 = process.split_data(
        dataset_cl2, training_parameters["train_ratio"], training_parameters["val_ratio"],training_parameters["test_ratio"], job_parameters["seed"]
    )
    train_nm, val_nm, test_nm = process.split_data(
        dataset_nm, training_parameters["train_ratio"], training_parameters["val_ratio"],training_parameters["test_ratio"], job_parameters["seed"]
    )

    
    if rank not in ("cpu", "cuda"):
        dist.barrier()

    ##Set up loader
    (
        train_loader_cl1,
        train_sampler_cl1,
    ) = loader_setup(
        model_parameters["batch_size"],
        train_cl1,
        rank,
        job_parameters["seed"],
        world_size,
    )
    
    (
        train_loader_cl2,
        train_sampler_cl2
    ) = loader_setup(
        model_parameters["batch_size"],
        train_cl2,
        rank,
        job_parameters["seed"],
        world_size,
    )
    
    (
        train_loader_nm,
        train_sampler_nm
    ) = loader_setup(
        model_parameters["batch_size"],
        train_nm,
        rank,
        job_parameters["seed"],
        world_size,
    )
    
    #### set up val loader
    (
        val_loader_cl1,
        val_sampler_cl1,
    ) = loader_setup(
        model_parameters["batch_size"],
        val_cl1,
        rank,
        job_parameters["seed"],
        world_size,
    )
    
    (
        val_loader_cl2,
        val_sampler_cl2,
    ) = loader_setup(
        model_parameters["batch_size"],
        val_cl2,
        rank,
        job_parameters["seed"],
        world_size,
    )
    
    (
        val_loader_nm,
        val_sampler_nm,
    ) = loader_setup(
        model_parameters["batch_size"],
        val_nm,
        rank,
        job_parameters["seed"],
        world_size,
    )
    
    ##Set up model
    model = model_setup(
        rank,
        model_parameters["model"],
        model_parameters,
        dataset_cl1,
        job_parameters["load_model"],
        job_parameters["model_path"],
        model_parameters.get("print_model", True),
    )
    linear_nm = torch.nn.Linear(115, 115).to(rank)
    linear_ve = torch.nn.Linear(115, 17).to(rank)
    
    ##Set-up optimizer & scheduler
    optimizer = getattr(torch.optim, model_parameters["optimizer"])(
        model.parameters(),
        lr=model_parameters["lr"],
        **model_parameters["optimizer_args"]
    )
    optimizer_nm = torch.optim.Adam(linear_nm.parameters(), lr=0.0005, weight_decay=0)
    optimizer_ve = torch.optim.Adam(linear_ve.parameters(), lr=0.0005, weight_decay=0)
    
    scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
        optimizer, **model_parameters["scheduler_args"]
    )
    
    criterion_nm = torch.nn.CrossEntropyLoss()
    criterion_ve = torch.nn.CrossEntropyLoss()
    
    #criterion_cl = NTXentLoss(temperature=0.07)
    #criterion_cl = torch.nn.CosineSimilarity(dim=1)
    #criterion_cl = torch.nn.BCEWithLogitsLoss()
    criterion_cl = torch.nn.CosineEmbeddingLoss()
    
    ##Start training
    job_name = str(job_parameters["job_name"])
    model, num_epoch, num_train_error, num_val_error, num_train_acc_node, num_val_acc_node, num_train_acc_ve, num_val_acc_ve = trainer(
        rank,
        world_size,
        model,
        linear_nm,
        linear_ve,
        optimizer,
        optimizer_nm,
        optimizer_ve,
        scheduler,
        criterion_nm,
        criterion_ve,
        criterion_cl,
        train_loader_cl1,
        train_loader_cl2,
        train_loader_nm,
        val_loader_cl1,
        val_loader_cl2,
        val_loader_nm,
        model_parameters["epochs"],
        training_parameters["verbosity"],
        job_name + "_tmp_pretrain.pth",
    )
    
    
    num_epoch, num_train_error, num_val_error = torch.Tensor(num_epoch).numpy(), torch.Tensor(num_train_error).numpy(), torch.Tensor(num_val_error).numpy()
    loss_matri = np.vstack((num_epoch, num_train_error, num_val_error, num_train_acc_node, num_val_acc_node, num_train_acc_ve, num_val_acc_ve))
    np.save(f"./{job_name}_loss.npy", loss_matri)
    
    if rank in (0, "cpu", "cuda"):

        ##Save model
        if job_parameters["save_model"] == "True":

            if rank not in ("cpu", "cuda"):
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    job_parameters["model_path"],
                )
            else:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    job_parameters["model_path"],
                )

        if rank not in ("cpu", "cuda"):
            dist.destroy_process_group()


###Regular training with train, val, test split
def train_regular_finetune(
    rank,
    world_size,
    data_path,
    input_model_file,
    job_parameters=None,
    training_parameters=None,
    model_parameters=None,
):
    # input_model_file = 'my_model.pth'
    ##DDP
    ddp_setup(rank, world_size)
    ##some issues with DDP learning rate
    if rank not in ("cpu", "cuda"):
        model_parameters["lr"] = model_parameters["lr"] * world_size

    ##Get dataset
    train_dataset = process.get_dataset(os.path.join(data_path, 'train'), training_parameters["target_index"], False)
    val_dataset = process.get_dataset(os.path.join(data_path, 'validation'), training_parameters["target_index"], False)
    test_dataset = process.get_dataset(os.path.join(data_path, 'test'), training_parameters["target_index"], False)

    if rank not in ("cpu", "cuda"):
        dist.barrier()

    ##Set up train loader
    (
        train_loader,
        train_sampler
    ) = loader_setup(
        model_parameters["batch_size"],
        train_dataset,
        rank,
        job_parameters["seed"],
        world_size,
    )

    ##Set up validation loader
    (
        val_loader,
        val_sampler
    ) = loader_setup(
        model_parameters["batch_size"],
        val_dataset,
        rank,
        job_parameters["seed"],
        world_size,
    )
    
    ##Set up test loader
    (
        test_loader,
        test_sampler
    ) = loader_setup(
        model_parameters["batch_size"],
        test_dataset,
        rank,
        job_parameters["seed"],
        world_size,
    )
    
    ##Set up model
    model = model_setup(
        rank,
        model_parameters["model"],
        model_parameters,
        train_dataset,
        job_parameters["load_model"],
        job_parameters["model_path"],
        model_parameters.get("print_model", True),
        input_model_file,
    )
    
     
    ##Set-up optimizer & scheduler
    optimizer = getattr(torch.optim, model_parameters["optimizer"])(
        model.parameters(),
        lr=model_parameters["lr"],
        **model_parameters["optimizer_args"]
    )
       
    scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
        optimizer, **model_parameters["scheduler_args"]
    )


    ##Start training
    job_name = str(job_parameters["job_name"])
    model, num_epoch, num_train_error, num_val_error = trainer_finetune(
        rank,
        world_size,
        model,
        optimizer,
        scheduler,
        training_parameters["loss"],
        train_loader,
        val_loader,
        train_sampler,
        model_parameters["epochs"],
        training_parameters["verbosity"],
        job_name + "_tmp_fintune.pth",
    )
    
    num_epoch, num_train_error, num_val_error = torch.Tensor(num_epoch).numpy(), torch.Tensor(num_train_error).numpy(), torch.Tensor(num_val_error).numpy()
    loss_matri = np.vstack((num_epoch, num_train_error, num_val_error))
    print("shape of finetune loss matrix: ", np.shape(loss_matri))
    np.save(f"./{job_name}_loss.npy", loss_matri)

    if rank in (0, "cpu", "cuda"):

        train_error = val_error = test_error = float("NaN")

        ##workaround to get training output in DDP mode
        ##outputs are slightly different, could be due to dropout or batchnorm?
        train_loader = DataLoader(
            train_dataset,
            batch_size=model_parameters["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        ##Get train error in eval mode
        train_error, train_out = evaluate_finetune(
            train_loader, model, training_parameters["loss"], rank, out=True
        )
        print("Train Error: {:.5f}".format(train_error))

        ##Get val error
        if val_loader != None:
            val_error, val_out = evaluate_finetune(
                val_loader, model, training_parameters["loss"], rank, out=True
            )
            print("Val Error: {:.5f}".format(val_error))

        ##Get test error
        if test_loader != None:
            test_error, test_out = evaluate_finetune(
                test_loader, model, training_parameters["loss"], rank, out=True
            )
            print("Test Error: {:.5f}".format(test_error))

        ##Save model
        if job_parameters["save_model"] == "True":

            if rank not in ("cpu", "cuda"):
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    job_parameters["model_path"],
                )
            else:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    job_parameters["model_path"],
                )

        ##Write outputs
        if job_parameters["write_output"] == "True":

            write_results(
                train_out, str(job_parameters["job_name"]) + "_train_outputs.csv"
            )
            if val_loader != None:
                write_results(
                    val_out, str(job_parameters["job_name"]) + "_val_outputs.csv"
                )
            if test_loader != None:
                write_results(
                    test_out, str(job_parameters["job_name"]) + "_test_outputs.csv"
                )

        if rank not in ("cpu", "cuda"):
            dist.destroy_process_group()

        ##Write out model performance to file
        #error_values = np.array((train_error.cpu(), val_error.cpu(), test_error.cpu()))
        error_values = np.array((train_error, val_error, test_error))
        if job_parameters.get("write_error") == "True":
            np.savetxt(
                job_parameters["job_name"] + "_error.csv",
                error_values[np.newaxis, ...],
                delimiter=",",
            )

        return error_values


###n-fold cross validation
def train_cv_finetune(
    rank,
    world_size,
    data_path,
    input_model_file,
    job_parameters=None,
    training_parameters=None,
    model_parameters=None,
):

    job_parameters["load_model"] = "False"
    job_parameters["save_model"] = "False"
    job_parameters["model_path"] = None
    
    ##DDP
    ddp_setup(rank, world_size)
    ##some issues with DDP learning rate
    if rank not in ("cpu", "cuda"):
        model_parameters["lr"] = model_parameters["lr"] * world_size
    
    #print('SESESE:', data_path, training_parameters["target_index"])
    ##Get dataset
    dataset = process.get_dataset(data_path, training_parameters["target_index"], False)

    ##Split datasets
    cv_dataset = process.split_data_cv(
        dataset, num_folds=job_parameters["cv_folds"], seed=job_parameters["seed"]
    )

    cv_error = 0
    cv_list = []
    for index in range(0, len(cv_dataset)):
        #print("Index: ", index)

        ##Set up model
        if index == 0:
            model = model_setup(
                rank,
                model_parameters["model"],
                model_parameters,
                dataset,
                job_parameters["load_model"],
                job_parameters["model_path"],
                print_model=True,
                input_model_file=input_model_file,
            )
        else:
            model = model_setup(
                rank,
                model_parameters["model"],
                model_parameters,
                dataset,
                job_parameters["load_model"],
                job_parameters["model_path"],
                print_model=False,
                input_model_file=input_model_file,
            )
            
        print("Get Model")
        ##Set-up optimizer & scheduler
        optimizer = getattr(torch.optim, model_parameters["optimizer"])(
            model.parameters(),
            lr=model_parameters["lr"],
            **model_parameters["optimizer_args"]
        )
        scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
            optimizer, **model_parameters["scheduler_args"]
        )

        ##Set up loader
        train_loader, test_loader, train_sampler, train_dataset, _ = loader_setup_cv(
            index, model_parameters["batch_size"], cv_dataset, rank, world_size
        )

        ##Start training (nihang modified
        model, _, _, _ = trainer_finetune(
            rank,
            world_size,
            model,
            optimizer,
            scheduler,
            training_parameters["loss"],
            train_loader,
            None,   # use test as validation to get the best model.
            train_sampler,
            model_parameters["epochs"],
            training_parameters["verbosity"],
            str(job_parameters["job_name"]) + "_tmp_cv.pth",
        )
      

        #if rank not in ("cpu", "cuda"):
        #    dist.barrier()

        if rank in (0, "cpu", "cuda"):
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=model_parameters["batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            ##Get train error
            train_error, train_out = evaluate_finetune(
                train_loader, model, training_parameters["loss"], rank, out=True
            )
            print("Train Error: {:.5f}".format(train_error))

            ##Get test error
            test_error, test_out = evaluate_finetune(
                test_loader, model, training_parameters["loss"], rank, out=True
            )
            print("Test Error: {:.5f}".format(test_error))

            cv_error = cv_error + test_error

            if index == 0:
                total_rows = test_out
            else:
                total_rows = np.vstack((total_rows, test_out))

        cv_list.append(cv_error)
        cv_error = 0
        
    ## nihang
    print('folds error: ', cv_list)
    cv_error_mean = np.mean(cv_list)
    cv_error_std = np.std(cv_list)
    print("CV Error Mean: {:.5f}".format(cv_error_mean))
    print("CV Error std: {:.5f}".format(cv_error_std))
        

    if rank not in ("cpu", "cuda"):
        dist.destroy_process_group()

    return cv_error


###Predict using a saved movel
def predict(dataset, loss, job_parameters=None):

    rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##Loads predict dataset in one go, care needed for large datasets)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    ##Load saved model
    assert os.path.exists(job_parameters["model_path"]), "Saved model not found"
    if str(rank) == "cpu":
        saved = torch.load(
            job_parameters["model_path"], map_location=torch.device("cpu")
        )
    else:
        saved = torch.load(
            job_parameters["model_path"], map_location=torch.device("cuda")
        )
    model = saved["full_model"]
    model = model.to(rank)
    model_summary(model)

    ##Get predictions
    time_start = time.time()
    test_error, test_out = evaluate_finetune(loader, model, loss, rank, out=True)
    elapsed_time = time.time() - time_start

    print("Evaluation time (s): {:.5f}".format(elapsed_time))

    ##Write output
    if job_parameters["write_output"] == "True":
        write_results(
            test_out, str(job_parameters["job_name"]) + "_predicted_outputs.csv"
        )

    return test_error

