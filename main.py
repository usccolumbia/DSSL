import os
import argparse
import time
import csv
import sys
import json
import random
import numpy as np
import pprint
import yaml

import torch
import torch.multiprocessing as mp

import ray
from ray import tune

from matdeeplearn import models
from matdeeplearn import process, process_cl, process_nm, process_nm_ve, process_nm_as
from matdeeplearn import training

################################################################################
#
################################################################################
#  MatDeepLearn code
################################################################################
#
################################################################################
def main():
    start_time = time.time()
    print("Starting...")
    print(
        "GPU is available:",
        torch.cuda.is_available(),
        ", Quantity: ",
        torch.cuda.device_count(),
    )

    parser = argparse.ArgumentParser(description="MatDeepLearn inputs")
    ###Job arguments
    parser.add_argument(
        "--config_path",
        default="config.yml",
        type=str,
        help="Location of config file (default: config.json)",
    )
    parser.add_argument(
        "--run_mode",
        default=None,
        type=str,
        help="run modes: Training, Predict, Repeat, CV, Hyperparameter, Ensemble, Analysis",
    )
    parser.add_argument(
        "--job_name",
        default=None,
        type=str,
        help="name of your job and output files/folders",
    )
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        help="DEEP_GATGNN_demo, FINETUNE_DEEP_GATGNN_demo, CGCNN_demo, MPNN_demo, SchNet_demo, MEGNet_demo, GCN_net_demo, SOAP_demo, SM_demo",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="seed for data split, 0=random",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="path of the model .pth file",
    )
    parser.add_argument(
        "--save_model",
        default=None,
        type=str,
        help="Save model",
    )
    parser.add_argument(
        "--load_model",
        default=None,
        type=str,
        help="Load model",
    )
    parser.add_argument(
        "--write_output",
        default=None,
        type=str,
        help="Write outputs to csv",
    )
    parser.add_argument(
        "--parallel",
        default=None,
        type=str,
        help="Use parallel mode (ddp) if available",
    )
    parser.add_argument(
        "--reprocess",
        default=None,
        type=str,
        help="Reprocess data since last run",
    )
    ###Processing arguments
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        help="Location of data containing structures (json or any other valid format) and accompanying files",
    )
    parser.add_argument("--format", default=None, type=str, help="format of input data")
    ###Training arguments
    parser.add_argument("--train_ratio", default=None, type=float, help="train ratio")
    parser.add_argument(
        "--val_ratio", default=None, type=float, help="validation ratio"
    )
    parser.add_argument("--test_ratio", default=None, type=float, help="test ratio")
    parser.add_argument(
        "--verbosity", default=None, type=int, help="prints errors every x epochs"
    )
    parser.add_argument(
        "--target_index",
        default=None,
        type=int,
        help="which column to use as target property in the target file",
    )
    ###Model arguments
    parser.add_argument(
        "--epochs",
        default=5,
        type=int,
        help="number of total epochs to run",
    )
    parser.add_argument("--batch_size", default=None, type=int, help="batch size")
    parser.add_argument("--lr", default=None, type=float, help="learning rate")
    parser.add_argument(
        "--gc_count",
        default=None,
        type=int,
        help="number of gc layers",
    )
    parser.add_argument(
        "--dropout_rate",
        default=None,
        type=float,
        help="dropout rate",
    )

    parser.add_argument(
        "--input_model_file",
        default='processed',
        type=str,
        help="Pretrained model file",
    )
    
    parser.add_argument(
        "--processed_path",
        default='processed',
        type=str,
        help="processed data file",
    )

    parser.add_argument(
        "--microp",
        default=None,
        type=str,
        help="None or AS or VE",
    )
    
    parser.add_argument("--local_rank", type=int, default=0)
    ##Get arguments from command line
    args = parser.parse_args(sys.argv[1:])

    ##Open provided config file
    assert os.path.exists(args.config_path), (
        "Config file not found in " + args.config_path
    )
    with open(args.config_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    ##Update config values from command line
    if args.run_mode != None:
        config["Job"]["run_mode"] = args.run_mode
    run_mode = config["Job"].get("run_mode")
    config["Job"] = config["Job"].get(run_mode)
    if config["Job"] == None:
        print("Invalid run mode")
        sys.exit()

    if args.job_name != None:
        config["Job"]["job_name"] = args.job_name
    if args.model != None:
        config["Job"]["model"] = args.model
    if args.seed != None:
        config["Job"]["seed"] = args.seed
    if args.model_path != None:
        config["Job"]["model_path"] = args.model_path
    if args.load_model != None:
        config["Job"]["load_model"] = args.load_model
    if args.save_model != None:
        config["Job"]["save_model"] = args.save_model
    if args.write_output != None:
        config["Job"]["write_output"] = args.write_output
    if args.parallel != None:
        config["Job"]["parallel"] = args.parallel
    if args.reprocess != None:
        config["Job"]["reprocess"] = args.reprocess

    if args.data_path != None:
        config["Processing"]["data_path"] = args.data_path
    if args.format != None:
        config["Processing"]["data_format"] = args.format

    if args.train_ratio != None:
        config["Training"]["train_ratio"] = args.train_ratio
    if args.val_ratio != None:
        config["Training"]["val_ratio"] = args.val_ratio
    if args.test_ratio != None:
        config["Training"]["test_ratio"] = args.test_ratio
    if args.verbosity != None:
        config["Training"]["verbosity"] = args.verbosity
    if args.target_index != None:
        config["Training"]["target_index"] = args.target_index

    for key in config["Models"]:
        if args.epochs != None:
            config["Models"][key]["epochs"] = args.epochs
        if args.batch_size != None:
            config["Models"][key]["batch_size"] = args.batch_size
        if args.lr != None:
            config["Models"][key]["lr"] = args.lr
        if args.gc_count != None:
            config["Models"][key]["gc_count"] = args.gc_count
        if args.dropout_rate != None:
            config["Models"][key]["dropout_rate"] = args.dropout_rate

    config["Models"] = config["Models"].get(config["Job"]["model"])
      
    if config["Job"]["seed"] == 0:
        config["Job"]["seed"] = np.random.randint(1, 1e6)

    ##Print and write settings for job
    print("Settings: ")
    pprint.pprint(config)
    with open(str(config["Job"]["job_name"]) + str(args.gc_count) + "_settings.txt", "w") as log_file:
        pprint.pprint(config, log_file)

    ################################################################################
    #  Begin data processing
    ################################################################################

    if run_mode == "Training":
        
        process_start_time = time.time()

        dataset_cl1 = process_cl.get_dataset(
            "data_cl1",
            config["Processing"]["data_path"],
            config["Training"]["target_index"],
            config["Job"]["reprocess"],
            config["Processing"],
        )
        
        dataset_cl2 = process_cl.get_dataset(
            "data_cl2",
            config["Processing"]["data_path"],
            config["Training"]["target_index"],
            config["Job"]["reprocess"],
            config["Processing"],

        )
        
        if not args.microp:
            dataset_nm = process_nm.get_dataset(
                config["Processing"]["data_path"],
                config["Training"]["target_index"],
                config["Job"]["reprocess"],
                config["Processing"],
            )
        elif args.microp == "VE":
            dataset_nm = process_nm_ve.get_dataset(
                config["Processing"]["data_path"],
                config["Training"]["target_index"],
                config["Job"]["reprocess"],
                config["Processing"],
            )
        elif args.microp == "AS":
            dataset_nm = process_nm_as.get_dataset(
                config["Processing"]["data_path"],
                config["Training"]["target_index"],
                config["Job"]["reprocess"],
                config["Processing"],
            )
        else:
            print("No valid microproperty selected, try again.")
            exit()

        #print("Dataset used:", dataset_cl1, dataset_cl1, dataset_nm)
        #print(dataset_cl1[0], dataset_cl2[0], dataset_nm[0])

        print("--- %s seconds for processing ---" % (time.time() - process_start_time))
    
    elif run_mode == "Finetune":
        
        process_start_time = time.time()
        
        train_dataset = process.get_dataset(
                     os.path.join(config["Processing"]["data_path"], 'train'), 
                     config["Training"]["target_index"],
                     config["Job"]["reprocess"],
                     config["Processing"],)
        val_dataset = process.get_dataset(
                     os.path.join( config["Processing"]["data_path"], 'validation'), 
                     config["Training"]["target_index"],
                     config["Job"]["reprocess"],
                     config["Processing"],)
        test_dataset = process.get_dataset(
                     os.path.join( config["Processing"]["data_path"], 'test'), 
                     config["Training"]["target_index"],
                     config["Job"]["reprocess"],
                     config["Processing"],)

        #print("Dataset used:", train_dataset)
        print("Dataset example: ", train_dataset[0])

        print("--- %s seconds for processing ---" % (time.time() - process_start_time))
        
    elif run_mode == "CV" or "Predict":
        
        process_start_time = time.time()
        
        dataset = process.get_dataset(
                     os.path.join(config["Processing"]["data_path"]), 
                     config["Training"]["target_index"],
                     config["Job"]["reprocess"],
                     config["Processing"],)
       
        #print("Dataset used:", train_dataset)
        print("Dataset example: ", dataset[0])

        print("--- %s seconds for processing ---" % (time.time() - process_start_time))
    
    ################################################################################
    #  Training begins
    ################################################################################
    
    ## Pretraining
    if run_mode == "Training":
   
        print("Starting  pretraining------------")
        print(
            "running for "
            + str(args.epochs)
            + " epochs"
            + " on "
            + str(config["Job"]["model"])
            + " model"
        )
        world_size = torch.cuda.device_count()

        ## dssl
        if not args.microp:
            if world_size == 0:
                print("Running on CPU - this will be slow")
                training.train_dssl(
                    "cpu",
                    world_size,
                    config["Processing"]["data_path"],
                    config["Job"],
                    config["Training"],
                    config["Models"],
                )

            elif world_size > 0:
                if config["Job"]["parallel"] == "True":
                    print("Running on", world_size, "GPUs")
                    mp.spawn(
                        training.train_dssl,
                        args=(
                            world_size,
                            config["Processing"]["data_path"],
                            config["Job"],
                            config["Training"],
                            config["Models"],
                        ),
                        nprocs=world_size,
                        join=True,
                    )
                if config["Job"]["parallel"] == "False":
                    print("Running on one GPU")
                    training.train_dssl(
                        "cuda",
                        world_size,
                        config["Processing"]["data_path"],
                        config["Job"],
                        config["Training"],
                        config["Models"],
                    )
        ## dssl + micro
        else:
            if world_size == 0:
                print("Running on CPU - this will be slow")
                training.train_dssl_micro(
                    "cpu",
                    world_size,
                    config["Processing"]["data_path"],
                    config["Job"],
                    config["Training"],
                    config["Models"],
                    args.microp,
                )

            elif world_size > 0:
                if config["Job"]["parallel"] == "True":
                    print("Running on", world_size, "GPUs")
                    mp.spawn(
                        training.train_dssl_micro,
                        args=(
                            world_size,
                            config["Processing"]["data_path"],
                            config["Job"],
                            config["Training"],
                            config["Models"],
                            args.microp,
                        ),
                        nprocs=world_size,
                        join=True,
                    )
                if config["Job"]["parallel"] == "False":
                    print("Running on one GPU")
                    training.train_dssl_micro(
                        "cuda",
                        world_size,
                        config["Processing"]["data_path"],
                        config["Job"],
                        config["Training"],
                        config["Models"],
                        args.microp,
                    )
                
    ## Finetune from a pretrained model
    elif run_mode == "Finetune":
        
        print("Starting Finetune")
        
        world_size = torch.cuda.device_count()
        if world_size == 0:
            print("Running on CPU - this will be slow")
            training.train_regular_finetune(
                "cpu",
                world_size,
                config["Processing"]["data_path"],
                args.input_model_file,
                config["Job"],
                config["Training"],
                config["Models"],
            )

        elif world_size > 0:
            if config["Job"]["parallel"] == "True":
                print("Running on", world_size, "GPUs")
                mp.spawn(
                    training.train_regular_finetune,
                    args=(
                        world_size,
                        config["Processing"]["data_path"],
                        args.input_model_file,
                        config["Job"],
                        config["Training"],
                        config["Models"],
                    ),
                    nprocs=world_size,
                    join=True,
                )
            if config["Job"]["parallel"] == "False":
                print("Running on one GPU")
                training.train_regular_finetune(
                    "cuda",
                    world_size,
                    config["Processing"]["data_path"],
                    args.input_model_file,
                    config["Job"],
                    config["Training"],
                    config["Models"],
                )  
    
    ##Running n fold cross validation
    elif run_mode == "CV":

        print("Starting cross validation")
        print(
            "running for "
            + str(config["Models"]["epochs"])
            + " epochs"
            + " on "
            + str(config["Job"]["model"])
            + " model"
        )
        world_size = torch.cuda.device_count()
        if world_size == 0:
            print("Running on CPU - this will be slow")
            training.train_cv_finetune(
                "cpu",
                world_size,
                config["Processing"]["data_path"],
                args.input_model_file,
                config["Job"],
                config["Training"],
                config["Models"],
            )

        elif world_size > 0:
            if config["Job"]["parallel"] == "True":
                print("Running on", world_size, "GPUs")
                mp.spawn(
                    training.train_cv_finetune,
                    args=(
                        world_size,
                        config["Processing"]["data_path"],
                        args.input_model_file,
                        config["Job"],
                        config["Training"],
                        config["Models"],
                    ),
                    nprocs=world_size,
                    join=True,
                )
            if config["Job"]["parallel"] == "False":
                print("Running on one GPU")
                training.train_cv_finetune(
                    "cuda",
                    world_size,
                    config["Processing"]["data_path"],
                    args.input_model_file,
                    config["Job"],
                    config["Training"],
                    config["Models"],
                )
                            
    ## Predicting from a trained model  ************not modified now
    elif run_mode == "Predict":

        print("Starting prediction from trained model")
        train_error = training.predict(
            dataset, config["Training"]["loss"], config["Job"]
        )
        print("Test Error: {:.5f}".format(train_error))

    else:
        print("No valid mode selected, try again")

    print("--- %s total seconds elapsed ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
