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

################################################################################
#  Training functions
################################################################################

##Pretrain step, runs model in train mode
def train_pretrain(model, model_linear, optimizer, optimizer_linear, loader_cl1, loader_cl2, loader_nm, criterion_nm, criterion_cl, rank):
    #print("SSSSSSstart train step-------")
    model.train()
    model_linear.train()
    
    loss_all = 0
    acc_node_accum = 0
    count = 0    
    count_node = 0
    for data_cl1, data_cl2, data_nm in zip(loader_cl1, loader_cl2, loader_nm):
        data_cl1 = data_cl1.to(rank)
        data_cl2 = data_cl2.to(rank)
        data_nm = data_nm.to(rank)

        #optimizer.zero_grad()
        
        target = data_nm.y
        target = torch.argmax(target, 1)
        node_pre1, node_pre2 = model(data_nm, data_nm)
        #print(data_nm)
        print(node_pre1.shape, node_pre2.shape)
        
        pred_node = model_linear(node_pre1[data_nm.atom_index])
        loss_nm = criterion_nm(pred_node, target) 
        
        node_cl1, node_cl2 = model(data_cl1, data_cl2)
        #_, graph_cl2 = model(data_cl2)
        #loss_cl = criterion_cl(graph_cl1, graph_cl2)
               
        optimizer.zero_grad()
        optimizer_linear.zero_grad()
        
        #loss = 0.5*loss_nm + 0.5*loss_cl
        loss_nm.backward()
        
        #print("backward time: ", time-time()-train_start_time)
        optimizer.step()
        optimizer_linear.step()
      
        count = count + output.size(0)
        #loss_all += loss.detach().cpu() * output.size(0)
        loss_all += loss.detach()* output.size(0)
          
    train_error = loss_all/count
    train_acc_atom = acc_node_accum/count_node
    #print(train_error, train_acc_atom)
    return train_error, train_acc_atom

'''
##Evaluation step, runs model in eval mode
def evaluate(loader, model, loss_method, rank, out=False):
    model.eval()
    loss_all = 0
    count = 0
    for data in loader:
        data = data.to(rank)
        with torch.no_grad():
            output = model(data)
            loss = getattr(F, loss_method)(output, data.y)
            loss_all += loss * output.size(0)
            if out == True:
                if count == 0:
                    ids = [item for sublist in data.structure_id for item in sublist]
                    ids = [item for sublist in ids for item in sublist]
                    predict = output.data.cpu().numpy()
                    target = data.y.cpu().numpy()
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
            count = count + output.size(0)

    loss_all = loss_all / count

    if out == True:
        test_out = np.column_stack((ids, target, predict))
        return loss_all, test_out
    elif out == False:
        return loss_all
'''

##Model trainer
def trainer(
    rank,
    world_size,
    model,
    model_linear,
    optimizer,
    optimizer_linear,
    scheduler,
    loss_nm,
    loss_cl,
    train_loader_cl1,
    train_loader_cl2,
    train_loader_nm,
    val_loader_cl1,
    val_loader_cl2,
    val_loader_nm,
    train_sampler_cl1,
    train_sampler_cl2,
    train_sampler_nm,
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
    num_val_acc_node = []
    
    ##Start training over epochs loop
    for epoch in range(1, epochs + 1):
        print("SSSSSSstart train step1111-------")
        lr = scheduler.optimizer.param_groups[0]["lr"]
        #if rank not in ("cpu", "cuda"):
        #    train_sampler.set_epoch(epoch)
        ##Train model
        train_error, train_acc_atom = train_pretrain(model, model_linear, optimizer, optimizer_linear, train_loader_cl1, train_loader_cl2, train_loader_nm, loss_nm, loss_cl, rank=rank)
        
        '''
        if rank not in ("cpu", "cuda"):
            torch.distributed.reduce(train_error, dst=0)
            train_error = train_error / world_size

        ##Get validation performance
        if rank not in ("cpu", "cuda"):
            dist.barrier()
        if val_loader != None and rank in (0, "cpu", "cuda"):
            if rank not in ("cpu", "cuda"):
                val_error = evaluate(
                    val_loader, model.module, loss, rank=rank, out=False
                )
            else:
                val_error = evaluate(val_loader, model, loss, rank=rank, out=False)

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

    return model_best
    '''


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

    # DDP
    if rank not in ("cpu", "cuda"):
        model = DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True
        )
        # model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
    if print_model == True and rank in (0, "cpu", "cuda"):
        model_summary(model)
        
    #print("FFFFFffinish model load-------")
    return model


##Pytorch loader setup
def data_loader(
    train_ratio,
    val_ratio,
    test_ratio,
    batch_size,
    dataset_cl1,
    dataset_cl2,
    dataset_nm,
    rank,
    seed,
    world_size=0,
    num_workers=0,
):
    ##Split datasets
    train_cl1, val_cl1, test_cl1 = process.split_data(
        dataset_cl1, train_ratio, val_ratio, test_ratio, seed
    )
    train_cl2, val_cl2, test_cl2 = process.split_data(
        dataset_cl2, train_ratio, val_ratio, test_ratio, seed
    )
    
    train_nm, val_nm, test_nm = process.split_data(
        dataset_nm, train_ratio, val_ratio, test_ratio, seed
    )

    ##DDP
    if rank not in ("cpu", "cuda"):
        train_sampler_cl1 = DistributedSampler(
            train_cl1, num_replicas=world_size, rank=rank
        )
        
        train_sampler_cl2 = DistributedSampler(
            train_cl2, num_replicas=world_size, rank=rank
        )
        
        train_sampler_nm = DistributedSampler(
            train_nm, num_replicas=world_size, rank=rank
        )
    elif rank in ("cpu", "cuda"):
        train_sampler_cl1 = None
        train_sampler_cl2 = None
        train_sampler_nm = None
    
    ##Load data
    train_loader_cl1 = val_loader_cl1 = test_loader_cl1 = None
    train_loader_cl2 = val_loader_cl2 = test_loader_cl2 = None
    train_loader_nm = val_loader_nm = test_loader_nm = None
    
    train_loader_cl1 = DataLoader(
        train_cl1,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler_cl1,
    )
    
    train_loader_cl2 = DataLoader(
        train_cl2,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler_cl2,
    )
    
    train_loader_nm = DataLoader(
        train_nm,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler_nm,
    )
    
    # may scale down batch size if memory is an issue
    if rank in (0, "cpu", "cuda"):
        if len(val_cl1) > 0 and len(val_cl2) > 0 and len(val_nm) > 0: 
            val_loader_cl1 = DataLoader(
                val_cl1,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            
            val_loader_cl2 = DataLoader(
                val_cl2,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            
            val_loader_nm = DataLoader(
                val_nm,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        if len(test_cl1) > 0 and len(test_cl2) > 0 and len(test_nm) > 0:
            test_loader_cl1 = DataLoader(
                test_cl1,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            test_loader_cl2 = DataLoader(
                test_cl2,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            test_loader_nm = DataLoader(
                test_nm,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
    return (
        train_loader_cl1,
        train_loader_cl2,
        train_loader_nm,
        val_loader_cl1,
        val_loader_cl2,
        val_loader_nm,
        test_loader_cl1,
        test_loader_cl2,
        test_loader_nm,
        #train_sampler,
        train_cl1,
        train_cl2,
        train_nm,
        val_cl1,
        val_cl2,
        val_nm,
        test_cl1,
        test_cl2,
        test_nm,
    )



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
    
    
    #print(dataset_cl1, dataset_cl1[0], len(dataset_cl1))
    #print(dataset_cl2, dataset_cl2[0], len(dataset_cl2))
    #print(dataset_nm, dataset_nm[0], len(dataset_nm))
    

    if rank not in ("cpu", "cuda"):
        dist.barrier()

    ##Set up loader
    (
        train_loader_cl1,
        train_loader_cl2,
        train_loader_nm,
        val_loader_cl1,
        val_loader_cl2,
        val_loader_nm,
        test_loader_cl1,
        test_loader_cl2,
        test_loader_nm,
        #train_sampler,
        train_cl1,
        train_cl2,
        train_nm,
        val_cl1,
        val_cl2,
        val_nm,
        test_cl1,
        test_cl2,
        test_nm,
    ) = data_loader(
        training_parameters["train_ratio"],
        training_parameters["val_ratio"],
        training_parameters["test_ratio"],
        model_parameters["batch_size"],
        dataset_cl1,
        dataset_cl2,
        dataset_nm,
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

    model_linear = torch.nn.Linear(115, 115).to(rank)
    
    ##Set-up optimizer & scheduler
    optimizer = getattr(torch.optim, model_parameters["optimizer"])(
        model.parameters(),
        lr=model_parameters["lr"],
        **model_parameters["optimizer_args"]
    )
    optimizer_linear = torch.optim.Adam(model_linear.parameters(), lr=0.0005, weight_decay=0)
    
    scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
        optimizer, **model_parameters["scheduler_args"]
    )
    
    criterion_nm = torch.nn.CrossEntropyLoss()
    #criterion_cl = NTXentLoss(temperature=0.07)
    #criterion_cl = torch.nn.CosineSimilarity(dim=1)
    criterion_cl = torch.nn.BCEWithLogitsLoss()
    ##Start training
    #model, num_epoch, num_train_error, num_val_error, num_train_acc_node, num_val_acc_node = 
    trainer(
        rank,
        world_size,
        model,
        model_linear,
        optimizer,
        optimizer_linear,
        scheduler,
        criterion_nm,
        criterion_cl,
        train_loader_cl1,
        train_loader_cl2,
        train_loader_nm,
        val_loader_cl1,
        val_loader_cl2,
        val_loader_nm,
        None,
        None,
        None,
        model_parameters["epochs"],
        training_parameters["verbosity"],
        "my_model_temp.pth",
    )
    
    '''
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
        train_error, train_out = evaluate(
            train_loader, model, training_parameters["loss"], rank, out=True
        )
        print("Train Error: {:.5f}".format(train_error))

        ##Get val error
        if val_loader != None:
            val_error, val_out = evaluate(
                val_loader, model, training_parameters["loss"], rank, out=True
            )
            print("Val Error: {:.5f}".format(val_error))

        ##Get test error
        if test_loader != None:
            test_error, test_out = evaluate(
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
        error_values = np.array((train_error.cpu(), val_error.cpu(), test_error.cpu()))
        if job_parameters.get("write_error") == "True":
            np.savetxt(
                job_parameters["job_name"] + "_errorvalues.csv",
                error_values[np.newaxis, ...],
                delimiter=",",
            )

        return error_values
    '''


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
    test_error, test_out = evaluate(loader, model, loss, rank, out=True)
    elapsed_time = time.time() - time_start

    print("Evaluation time (s): {:.5f}".format(elapsed_time))

    ##Write output
    if job_parameters["write_output"] == "True":
        write_results(
            test_out, str(job_parameters["job_name"]) + "_predicted_outputs.csv"
        )

    return test_error

