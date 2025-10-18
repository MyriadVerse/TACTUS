import pickle
import argparse
import time
import numpy as np
from tqdm import tqdm
from src.search import TACTSearcher
from src.utils import calcMetrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default='santos')
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--single_column", dest="single_column", action="store_true")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--scal", type=float, default=1.00)
    parser.add_argument("--augment_op", type=str, default='sample_table')
    parser.add_argument("--sample_meth", type=str, default='priority_sample')
    parser.add_argument("--pooling", type=str, default='multihead_attn')

    hp = parser.parse_args()

    singleCol = hp.single_column
    epoch = hp.n_epochs
    pooling = hp.pooling
    table_id = hp.run_id
    dataFolder = hp.benchmark
    sampAug = hp.augment_op + "_" + hp.sample_meth
    K = hp.K

    table_path = "dir_to_embedding/"+dataFolder+"/vectors/cl_datalake_"+sampAug+"_column_ep@"+str(epoch)+'_'+str(table_id)+".pkl"
    query_path = "dir_to_embedding/"+dataFolder+"/vectors/cl_query_"+sampAug+"_column_ep@"+str(epoch)+'_'+str(table_id)+".pkl"
    index_path = "dir_to_index/"+dataFolder+"/indexes/hnsw_open_data_"+str(table_id)+"_"+str(hp.scal)+".bin"

    searcher = TACTSearcher(table_path, index_path, hp.scal)
    queries = pickle.load(open(query_path,"rb"))

    returnedResults = {}
    saveResults = {}
    avgNumResults = []
    query_times = []

    for q in tqdm(queries):
        query_start_time = time.time()
        res, scoreLength = searcher.topk(q,K)
        avgNumResults.append(scoreLength)
        saveResults[q[0]] = res
        returnedResults[q[0]] = [r[1] for r in res]
        query_times.append(time.time() - query_start_time)

    print("Average number of Results: ", sum(avgNumResults)/len(avgNumResults))
    print("Average QUERY TIME: %s seconds " % (sum(query_times)/len(query_times)))
    print("10th percentile: ", np.percentile(query_times, 10), " 90th percentile: ", np.percentile(query_times, 90))

    if hp.benchmark == 'santosLarge' or hp.benchmark == 'wdc':
        print("No groundtruth for %s benchmark" % (hp.benchmark))
    else:
        if 'santos' in hp.benchmark:
            k_range = 2
            groundTruth = "dir_to_gt"
        elif hp.benchmark == 'wiki':
            k_range = 8
            groundTruth = "dir_to_gt"
        else: 
            k_range = 10
            if hp.benchmark == 'tus':
                groundTruth = 'dir_to_gt'
            elif hp.benchmark == 'tusLarge':
                groundTruth = 'dir_to_gt'

        calcMetrics(K, k_range, returnedResults, gtPath=groundTruth)

    with open(f'dir_to_results/{hp.benchmark}_{pooling}_results.pkl', 'wb') as f:
        pickle.dump(saveResults, f)
