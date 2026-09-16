import numpy as np
import torch
from operator import truediv

def evaluate_accuracy(data_iter, net, loss, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            test_l_sum, test_num = 0, 0
            #X = X.permute(0, 3, 1, 2)
            X = X.to(device)
            y = y.to(device)
            net.eval() 
            y_hat = net(X)
            l = loss(y_hat, y.long())
            acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            test_l_sum += l
            test_num += 1
            net.train() 
            n += y.shape[0]
    return [acc_sum / n, test_l_sum] # / test_num]


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc



import numpy as np

def record_output(oa_ae, aa_ae, kappa_ae, element_acc_ae, precision_ae, f1_ae, path):
    f = open(path, 'w')
    
    # 1. 记录每次实验的原始数据
    f.write('OAs for each iteration are: ' + str(oa_ae) + '\n')
    f.write('AAs for each iteration are: ' + str(aa_ae) + '\n')
    f.write('Kappas for each iteration are: ' + str(kappa_ae) + '\n') ## ADDED ##
    f.write('Precisions for each iteration are: ' + str(precision_ae) + '\n')
    f.write('F1-Scores for each iteration are: ' + str(f1_ae) + '\n')
    f.write('\n')

    # 2. 记录均值和标准差 (Mean ± Std)
    f.write('mean_OA ± std_OA is: ' + str(np.mean(oa_ae)) + ' ± ' + str(np.std(oa_ae)) + '\n')
    f.write('mean_AA ± std_AA is: ' + str(np.mean(aa_ae)) + ' ± ' + str(np.std(aa_ae)) + '\n')
    f.write('mean_Kappa ± std_Kappa is: ' + str(np.mean(kappa_ae)) + ' ± ' + str(np.std(kappa_ae)) + '\n') ## ADDED ##
    f.write('mean_Precision ± std_Precision is: ' + str(np.mean(precision_ae)) + ' ± ' + str(np.std(precision_ae)) + '\n') 
    f.write('mean_F1 ± std_F1 is: ' + str(np.mean(f1_ae)) + ' ± ' + str(np.std(f1_ae)) + '\n')       
    f.write('\n')

    # 3. 记录每一类的精度 (Per-Class Accuracy)
    element_mean = np.mean(element_acc_ae, axis=0)
    element_std = np.std(element_acc_ae, axis=0)
    
    f.write('Mean of all elements in confusion matrix: ' + str(element_mean) + '\n')
    f.write('Standard deviation of all elements in confusion matrix: ' + str(element_std) + '\n\n')

    # 4. 生成最终汇总行 (方便复制到 Excel)
    element_mean = list(element_mean)
    ## MODIFIED ## - 将 Kappa 均值添加到汇总列表中
    element_mean.extend([
        np.mean(oa_ae), 
        np.mean(aa_ae), 
        np.mean(kappa_ae), # <-- 添加 Kappa
        np.mean(precision_ae), 
        np.mean(f1_ae)         
    ])
    
    element_std = list(element_std)
    ## MODIFIED ## - 将 Kappa 标准差添加到汇总列表中
    element_std.extend([
        np.std(oa_ae), 
        np.std(aa_ae), 
        np.std(kappa_ae), # <-- 添加 Kappa
        np.std(precision_ae), 
        np.std(f1_ae)         
    ])

    f.write('All values without std: ' + str(element_mean) + '\n')
    f.write('All values with std: \n')
    
    sentence11 = ""
    for i, x in enumerate(element_mean):
        sentence11 += str(element_mean[i]) + " ± " + str(element_std[i]) + "  "
    
    f.write(sentence11 + '\n')
    f.close()
