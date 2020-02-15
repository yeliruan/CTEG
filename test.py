#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

def handle():
    path = r'C:\Users\13314\Desktop\test\concept_mem_adv_log.txt'
    path1 = r'C:\Users\13314\Desktop\test\reward.txt'

    writer1 = open(path1,'w',encoding='utf8')
    writer1.write('epoch\tstep\taverage reward\tadversarial loss\tmle loss')
    with open(path,'r',encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if(line.find('epoch')!=-1):
                writer1.write('\n')
                strs = line.split("  ")
                writer1.write(strs[0]+'\t')

                writer1.write(strs[3]+'\t')

            else:
                strs = line.split(":")
                writer1.write(strs[1]+'\t')

if __name__ == '__main__':
    handle()
    exit(0)
