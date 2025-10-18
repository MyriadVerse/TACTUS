from src.train import load_checkpoint, inference_on_tables
import torch
import pandas as pd
import numpy as np
import glob
import pickle
import argparse

def extractVectors(dfs, dataFolder, augment, sample, table_order, run_id, singleCol=False, epoch=0):
    if singleCol:
        model_path = "dir_to_model/%s/model_%s_%s_%s_ep@%d_%dsingleCol.pt" % (dataFolder, augment, sample, table_order,run_id)
    else:
        model_path = "dir_to_model/%s/model_%s_%s_%s_ep@%d_%d.pt" % (dataFolder, augment, sample, table_order, epoch, run_id)
    ckpt = torch.load(model_path, map_location=torch.device('cuda'))
    model, trainset = load_checkpoint(ckpt)
    return inference_on_tables(dfs, model, trainset, batch_size=512), model

def get_df(dataFolder):
    dataFiles = glob.glob(dataFolder+"/*.csv")
    dataDFs = {}
    for file in dataFiles:
        df = pd.read_csv(file,lineterminator='\n')
        filename = file.split("/")[-1]
        dataDFs[filename] = df
    return dataDFs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="santos") 
    parser.add_argument("--single_column", dest="single_column", action="store_true")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--table_order", type=str, default='column')
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--augment_op", type=str, default='sample_table')
    parser.add_argument("--sample_meth", type=str, default='priority_sample')

    hp = parser.parse_args()

    dataFolder = hp.benchmark
    isSingleCol = hp.single_column
    ao = hp.augment_op
    sm = hp.sample_meth
    run_id = hp.run_id
    table_order = hp.table_order
    epoch = hp.n_epochs

    if dataFolder == 'santos':
        DATAPATH = "dir_to_benchmark"
        dataDir = ['query', 'datalake']
    elif dataFolder == 'santosLarge':
        DATAPATH = 'dir_to_benchmark'
        dataDir = ['query', 'datalake']
    elif dataFolder == 'tus':
        DATAPATH = 'dir_to_benchmark'
        dataDir = ['query', 'benchmark']
    elif dataFolder == 'tusLarge':
        DATAPATH = 'dir_to_benchmark'
        dataDir = ['query', 'benchmark']
    elif dataFolder == 'wiki':
        DATAPATH = 'dir_to_benchmark'
        dataDir = ['query', 'datalake']
    elif dataFolder == 'wdc':
        DATAPATH = 'dir_to_benchmark'
        dataDir = ['query', 'benchmark']

    for dir in dataDir:
        DATAFOLDER = DATAPATH+dir
        dfs = get_df(DATAFOLDER)

        dataEmbeds = []
        dfs_totalCount = len(dfs)
        dfs_count = 0

        cl_features, model = extractVectors(list(dfs.values()), dataFolder, ao, sm, table_order, run_id, singleCol=isSingleCol, epoch=epoch)
        for i, file in enumerate(dfs):
            dfs_count += 1
            cl_features_file = np.array(cl_features[i])

            with torch.no_grad():
                z = torch.tensor(cl_features_file, dtype=torch.float32).to(model.device) 
                table_indices = torch.zeros(z.size(0), dtype=torch.long, device=model.device)
                
                if model.pool == 'attn':
                    table_embedding = model._attention_pool(z, table_indices)
                elif model.pool == 'multihead_attn':
                    table_embedding = model._multihead_attn_pool(z, table_indices)
                else:
                    raise ValueError("Invalid pooling type.")
                table_embedding = model.table_projector(table_embedding)
            dataEmbeds.append((file, table_embedding.squeeze(0)))

        if dir == 'santos-query':
            saveDir = 'query'
        elif dir == 'benchmark':
            saveDir = 'datalake'
        else: saveDir = dir

        if isSingleCol:
            output_path = "dir_to_embedding/%s/vectors/cl_%s_%s_%s_%s_ep@%d_%d_singleCol.pkl" % (dataFolder, saveDir, ao, sm, table_order, epoch, run_id)
        else:
            output_path = "dir_to_embedding/%s/vectors/cl_%s_%s_%s_%s_ep@%d_%d.pkl" % (dataFolder, saveDir, ao, sm, table_order, epoch, run_id)
        if hp.save_model:
            pickle.dump(dataEmbeds, open(output_path, "wb"))
