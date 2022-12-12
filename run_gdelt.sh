# sh config
####### GDELT #######
CUDA_VISIBLE_DEVICES=3 python3 run_e2e_train.py --loss_lambda 0.3 --exp pure_de_eb1_gdelt --ablation 3 --seed 10000

CUDA_VISIBLE_DEVICES=2 python3 run_e2e_train.py --loss_lambda 0.3 --exp pure_de_eb2_gdelt --ablation 3 --seed 50000

CUDA_VISIBLE_DEVICES=3 python3 run_e2e_train.py --loss_lambda 0.3 --exp pure_ut_eb1_gdelt_5_0.5 --ablation 3 --seed 10000 --tkg_type UTEE --num_train_epochs 5 --warm_up 0.5

CUDA_VISIBLE_DEVICES=2 python3 run_e2e_train.py --loss_lambda 0.3 --exp pure_ut_eb2_gdelt_5_0.5 --ablation 3 --seed 50000 --tkg_type UTEE --num_train_epochs 5 --warm_up 0.5

#test e2e on hit1 data
CUDA_VISIBLE_DEVICES=0 python3 test_e2e.py --data_dir "/home/wiss/zhang/ruotong/TKGC/FTKE_Bert/data/GDELT_hit1_test/pretrain_data" --model_save_dir "/home/wiss/zhang/ruotong/TKGC/FTKE_Bert/model/ckpts_e2e_2/2022-04-04 19:49:19.272029" --model_index 2 --kg_model_chkpnt "../model/ckpts_finetuning_2/768_baseline/DE_SimplE_bs256_ne20_lr0.0005_nr200_seprop0.68_1.chkpnt"
#test de ckpt on gdelt test data
CUDA_VISIBLE_DEVICES=3 python3 test_e2e.py --model_save_dir "/home/wiss/zhang/ruotong/TKGC/FTKE_Bert/model/ckpts_e2e_2/2022-04-04 19:49:19.272029" --model_index 2 --kg_model_chkpnt "../model/ckpts_finetuning_2/768_baseline/DE_SimplE_bs256_ne20_lr0.0005_nr200_seprop0.68_1.chkpnt"
#test ecola-de ckpt on gdelt test data
CUDA_VISIBLE_DEVICES=3 python3 test_e2e.py --model_save_dir "/home/wiss/zhang/ruotong/TKGC/FTKE_Bert/model/ckpts_e2e_2/2022-04-04 19:49:19.272029" --model_index 2  

#gdelt-s
CUDA_VISIBLE_DEVICES=0 python3 run_e2e_train.py --loss_lambda 0.3 --exp pure_de_gdelt_s --ablation 3 --dataset 'GDELT-s' --data_dir '../data/GDELT_s/pretrain_data' --entity_dic_file '../data/GDELT_s/entities.txt' --relation_dic_file '../data/GDELT_s/relations.txt' --num_train_epoch 3
#plunder
CUDA_VISIBLE_DEVICES=2 python3 run_e2e_train.py --loss_lambda 0.3 --exp pure_dy_eb1_gdelt_p4_2 --ablation 3 --seed 10000 --tkg_type DyERNIE --num_train_epochs 4
#plunder
CUDA_VISIBLE_DEVICES=2 python3 run_e2e_train.py --loss_lambda 0.3 --exp pure_dy_eb2_gdelt_p4_2 --ablation 3 --seed 50000 --tkg_type DyERNIE --num_train_epochs 4

####### WIKI #######
#DE
CUDA_VISIBLE_DEVICES=2 python3 run_e2e_train.py --loss_lambda 0.001 --learning_rate 0.00001 --exp pure_de_eb1_wiki_ckpt --ablation 3 --seed 10000 --tkg_type DE --dataset 'Wiki' --data_dir '../data/yago_filtered' --entity_dic_file '../data/yago_filtered/ent2id_with_mask.txt' --relation_dic_file '../data/yago_filtered/rel2id_with_mask.txt' --num_train_epoch 10 --kg_model_chkpnt "/home/wiss/zhang/ruotong/TKGC/FTKE_Bert/model/yago_model/ckpts_finetuning/DE_SimplE_2022-03-03 13:00:50.138083/DE_SimplE_bs256_ne20_lr0.0005_nr100_seprop0.68_10.chkpnt"

CUDA_VISIBLE_DEVICES=1 python3 run_e2e_train.py --loss_lambda 0.001 --learning_rate 0.00001 --exp pure_de_eb2_wiki_ckpt --ablation 3 --seed 50000 --tkg_type DE --dataset 'Wiki' --data_dir '../data/yago_filtered' --entity_dic_file '../data/yago_filtered/ent2id_with_mask.txt' --relation_dic_file '../data/yago_filtered/rel2id_with_mask.txt' --num_train_epoch 10 --kg_model_chkpnt "/home/wiss/zhang/ruotong/TKGC/FTKE_Bert/model/yago_model/ckpts_finetuning/DE_SimplE_2022-03-03 13:00:50.138083/DE_SimplE_bs256_ne20_lr0.0005_nr100_seprop0.68_10.chkpnt"

#utee
CUDA_VISIBLE_DEVICES=1 python3 run_e2e_train.py --loss_lambda 0.3 --learning_rate 0.0002 --exp pure_ut_eb1_wiki --ablation 3 --seed 10000 --tkg_type UTEE --dataset 'Wiki' --data_dir '../data/yago_filtered' --entity_dic_file '../data/yago_filtered/ent2id_with_mask.txt' --relation_dic_file '../data/yago_filtered/rel2id_with_mask.txt' --num_train_epoch 20

CUDA_VISIBLE_DEVICES=1 python3 run_e2e_train.py --loss_lambda 0.3 --learning_rate 0.0002 --exp pure_ut_eb2_wiki --ablation 3 --seed 50000 --tkg_type UTEE --dataset 'Wiki' --data_dir '../data/yago_filtered' --entity_dic_file '../data/yago_filtered/ent2id_with_mask.txt' --relation_dic_file '../data/yago_filtered/rel2id_with_mask.txt' --num_train_epoch 20

#dyernie

CUDA_VISIBLE_DEVICES=0 python3 run_e2e_train.py --loss_lambda 0.3 --learning_rate 0.0002 --exp pure_dy_eb1_wiki --ablation 3 --seed 10000 --tkg_type DyERNIE --dataset 'Wiki' --data_dir '../data/yago_filtered' --entity_dic_file '../data/yago_filtered/ent2id_with_mask.txt' --relation_dic_file '../data/yago_filtered/rel2id_with_mask.txt' --num_train_epoch 25

CUDA_VISIBLE_DEVICES=0 python3 run_e2e_train.py --loss_lambda 0.3 --learning_rate 0.0002 --exp pure_dy_eb2_wiki --ablation 3 --seed 50000 --tkg_type DyERNIE --dataset 'Wiki' --data_dir '../data/yago_filtered' --entity_dic_file '../data/yago_filtered/ent2id_with_mask.txt' --relation_dic_file '../data/yago_filtered/rel2id_with_mask.txt' --num_train_epoch 25


####### DUEE ######
#de
CUDA_VISIBLE_DEVICES=0 python3 run_e2e_train.py --loss_lambda 0.3 --exp pure_de_eb1_duee --ablation 3 --seed 10000 --warm_up 0.3 --warm_up 0.2 --learning_rate 0.0005 --train_batch_size 8 --num_train_epoch 1000 --dataset 'DuEE' --data_dir "../data/DuEE/DuEE_e2e_backup_3/pretrain_data/" --entity_dic_file "../data/DuEE/DuEE_e2e/entities.txt" --relation_dic_file "../data/DuEE/DuEE_e2e/relations.txt"

CUDA_VISIBLE_DEVICES=2 python3 run_e2e_train.py --loss_lambda 0.3 --exp pure_de_eb2_duee --ablation 3 --seed 50000 --warm_up 0.3 --warm_up 0.2 --learning_rate 0.0005 --train_batch_size 8 --num_train_epoch 1000 --dataset 'DuEE' --data_dir "../data/DuEE/DuEE_e2e_backup_2/pretrain_data/" --entity_dic_file "../data/DuEE/DuEE_e2e/entities.txt" --relation_dic_file "../data/DuEE/DuEE_e2e/relations.txt"

#utee
CUDA_VISIBLE_DEVICES=3 python3 run_e2e_train.py --loss_lambda 0.3 --exp pure_ut_eb1_duee --ablation 3 --seed 10000 --train_batch_size 8 --num_train_epoch 1000 --dataset 'DuEE' --data_dir "../data/DuEE/DuEE_e2e_backup_1/pretrain_data/" --entity_dic_file "../data/DuEE/DuEE_e2e/entities.txt" --relation_dic_file "../data/DuEE/DuEE_e2e/relations.txt"

CUDA_VISIBLE_DEVICES=3 python3 run_e2e_train.py --loss_lambda 0.3 --exp pure_ut_eb1_duee --ablation 3 --seed 50000 --train_batch_size 8 --num_train_epoch 1000 --dataset 'DuEE' --data_dir "../data/DuEE/DuEE_e2e/pretrain_data/" --entity_dic_file "../data/DuEE/DuEE_e2e/entities.txt" --relation_dic_file "../data/DuEE/DuEE_e2e/relations.txt"

#dyernie
CUDA_VISIBLE_DEVICES=1 python3 run_e2e_train.py --loss_lambda 0.3 --exp pure_dy_eb1_duee --ablation 3 --seed 10000 --train_batch_size 8 --num_train_epoch 1000 --dataset 'DuEE' --data_dir "../data/DuEE/DuEE_e2e_backup_4/pretrain_data/" --entity_dic_file "../data/DuEE/DuEE_e2e/entities.txt" --relation_dic_file "../data/DuEE/DuEE_e2e/relations.txt"

CUDA_VISIBLE_DEVICES=0 python3 run_e2e_train.py --loss_lambda 0.3 --exp pure_dy_eb2_duee --ablation 3 --seed 50000 --train_batch_size 8 --num_train_epoch 1000 --dataset 'DuEE' --data_dir "../data/DuEE/DuEE_e2e_backup_5/pretrain_data/" --entity_dic_file "../data/DuEE/DuEE_e2e/entities.txt" --relation_dic_file "../data/DuEE/DuEE_e2e/relations.txt"
