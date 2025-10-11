import pickle
import pickle5 as p
import mlflow


def loadDictionaryFromPickleFile(dictionaryPath):
    filePointer=open(dictionaryPath, 'rb')
    dictionary = p.load(filePointer)
    filePointer.close()
    return dictionary

def saveDictionaryAsPickleFile(dictionary, dictionaryPath):
    filePointer=open(dictionaryPath, 'wb')
    pickle.dump(dictionary,filePointer, protocol=pickle.HIGHEST_PROTOCOL)
    filePointer.close()


def calcMetrics(max_k, k_range, resultFile, gtPath=None, resPath=None, record=True):
    groundtruth = loadDictionaryFromPickleFile(gtPath)

    precision_array = []
    recall_array = []
    ideal_recalls = []
    for k in range(1, max_k+1):
        true_positive = 0
        false_positive = 0
        false_negative = 0
        rec = 0
        ideal_recall = []
        for table in resultFile:
            if table.split("____",1)[0] != "t_28dc8f7610402ea7": 
                if table in groundtruth:
                    groundtruth_set = set(groundtruth[table])
                    groundtruth_set = {x.split(".")[0] for x in groundtruth_set}
                    result_set = resultFile[table][:k]
                    result_set = [x.split(".")[0] for x in result_set]
                    find_intersection = set(result_set).intersection(groundtruth_set)
                    tp = len(find_intersection)
                    fp = k - tp
                    fn = len(groundtruth_set) - tp
                    if len(groundtruth_set)>=k: 
                        true_positive += tp
                        false_positive += fp
                        false_negative += fn
                    rec += tp / (tp+fn)
                    ideal_recall.append(k/len(groundtruth[table]))
        precision = true_positive / (true_positive + false_positive)
        recall = rec/len(resultFile)
        precision_array.append(precision)
        recall_array.append(recall)
        if k % 10 == 0:
            ideal_recalls.append(sum(ideal_recall)/len(ideal_recall))
    used_k = [k_range]
    if max_k > k_range:
        for i in range(k_range * 2, max_k+1, k_range):
            used_k.append(i)
    print("--------------------------")
    for k in used_k:
        print("Precision at k = ",k,"=", precision_array[k-1])
        print("Recall at k = ",k,"=", recall_array[k-1])
        print("IDEAL RECALL at k = ",k,"=", ideal_recalls[k//10-1])
        print("--------------------------")
    
    map_sum = 0
    for k in range(0, max_k):
        map_sum += precision_array[k]
    mean_avg_pr = map_sum/max_k
    print("The mean average precision is:", mean_avg_pr)

    if record: 
        mlflow.log_metric("mean_avg_precision", mean_avg_pr)
        mlflow.log_metric("prec_k", precision_array[max_k-1])
        mlflow.log_metric("recall_k", recall_array[max_k-1])

    return mean_avg_pr, precision_array[max_k-1], recall_array[max_k-1] 