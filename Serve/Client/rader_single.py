import argparse
import os,sys
import time
import subprocess
from subprocess import PIPE, Popen

signal_idx = [0, 1, 2, 3, 4, 5, 6, 7]
signal_label = ['LFM', 'Barker', 'costas', 'Frank', 'P1', 'P2', 'P3', 'P4']

def inference_cmd(img_path):
    cmd = "curl -s http://localhost:8081/hi -d {}".format(img_path)

    process = subprocess.Popen(cmd, shell=True, stdout=PIPE, stderr=None)
    output = process.communicate()[0]
    
    pred_idx = int(bytes.decode(output))
    return pred_idx

parser = argparse.ArgumentParser(description="地址接收")
parser.add_argument("img_path", type=str, help="传入地址")
args = parser.parse_args()

idx = inference_cmd(args.img_path)
label = signal_label[idx]
print(label)

# python rader_single.py img_path
