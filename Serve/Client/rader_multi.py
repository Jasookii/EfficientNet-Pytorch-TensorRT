from subprocess import PIPE, Popen
import subprocess
import os, sys
import time

signal_idx = [0, 1, 2, 3, 4, 5, 6, 7]
signal_label = ['LFM', 'Barker', 'costas', 'Frank', 'P1', 'P2', 'P3', 'P4']

def inference_cmd(img_path):
    cwd = './'
    cmd = "curl -s http://localhost:8081/hi -d {}".format(img_path)

    process = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=PIPE, stderr=None)
    output = process.communicate()[0]
   
    pred_label = int(bytes.decode(output))
    return pred_label

def infer_dir(src_dir):
    signal_labels = [0, 1, 2, 3, 4, 5, 6, 7]
    signal_dirs = ['signal_1', 'signal_2', 'signal_3', 'signal_4', 'signal_5', 'signal_6', 'signal_7', 'signal_8']
    total_num = 0
    acc_num = 0
    
    for idx, signal_dir in enumerate(signal_dirs):
        print('='*12, signal_dir, '='*12)
        root_path = os.path.join(src_dir, signal_dir)
        for root, dirs, files in os.walk(root_path):
            # One Class Prediction
            true_label = signal_labels[idx]
            current_total_num = len(files)
            current_acc_num = 0
            
            for i, file in enumerate(files):
                img_path = os.path.join(root_path, file)
                pred_label = inference_cmd(img_path)
                print(img_path, signal_label[pred_label])
                if pred_label == true_label:
                    current_acc_num = current_acc_num + 1
                if (i!=1 and i%20==0) or (i==len(file)-1):
                    print('Num:', i+1, '  Acc Num:', current_acc_num, '  Acc1:', current_acc_num/(i+1))
            
            total_num = total_num + current_total_num
            acc_num = acc_num + current_acc_num
    print('Total Num:', total_num, '  Acc Num:', acc_num, '  Acc1:{:.5f}'.format(acc_num/total_num))

dir_path = "/home/nvidia/EfficientNet/Show/signal500_stft50_matrix_split/test/" 
infer_dir(dir_path)





