#!/bin/bash

python train.py Dataset/vmatrices --source 1 2 --target 1 2 --save checkpoint_s1_2.pt
python train.py Dataset/vmatrices --source 1 2 3 --target 1 2 3 --save checkpoint_s1_2_3.pt
python train.py Dataset/vmatrices --source 1 2 3 4 --target 1 2 3 4 --save checkpoint_s1_2_3_4.pt
python train.py Dataset/vmatrices --source 1 2 3 4 5 --target 1 2 3 4 5 --save checkpoint_s1_2_3_4_5.pt
python train.py Dataset/vmatrices --source 1 2 3 4 5 6 --target 1 2 3 4 5 6 --save checkpoint_s1_2_3_4_5_6.pt
python train.py Dataset/vmatrices --source 1 2 3 4 5 6 7 --target 1 2 3 4 5 6 7 --save checkpoint_s1_2_3_4_5_6_7.pt
python train.py Dataset/vmatrices --source 1 2 3 4 5 6 7 8 --target 1 2 3 4 5 6 7 8 --save checkpoint_s1_2_3_4_5_6_7_8.pt
python train.py Dataset/vmatrices --source 1 2 3 4 5 6 7 8 9 --target 1 2 3 4 5 6 7 8 9 --save checkpoint_s1_2_3_4_5_6_7_8_9.pt





# python train.py Dataset/vmatrices --source 1 --target 2 --save checkpoint_s1.pt
# python train.py Dataset/vmatrices --source 2 --target 2 --save checkpoint_s2.pt
# python train.py Dataset/vmatrices --source 2 --target 2 --save checkpoint_s2.pt
# python train.py Dataset/vmatrices --source 3 --target 3 --save checkpoint_s3.pt
# python train.py Dataset/vmatrices --source 4 --target 4 --save checkpoint_s4.pt
# python train.py Dataset/vmatrices --source 5 --target 5 --save checkpoint_s5.pt
# python train.py Dataset/vmatrices --source 6 --target 6 --save checkpoint_s6.pt
# python train.py Dataset/vmatrices --source 7 --target 7 --save checkpoint_s7.pt
# python train.py Dataset/vmatrices --source 8 --target 8 --save checkpoint_s8.pt
# python train.py Dataset/vmatrices --source 9 --target 9 --save checkpoint_s9.pt
