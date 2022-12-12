import argparse
import subprocess

learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001] #, ] #
warm_ups = [0.05, 0.2, 0.3] 
weight_decays = [0.01, 0.05, 0.2] 
# weight_decays = [0.1] # default setting
train_batch_sizes = [16,128,256,512]
gpus = [0, 1 ,2, 3]
for lr in learning_rates:
    for warm_up in warm_ups:
        for weight_decay in weight_decays:
            # if warm_up == 0.05 and weight_decay == 0.01:
                # continue
            # else:
            for train_batch_size in train_batch_sizes:
                subprocess.run([
                        "python3", "run_e2e_train.py",  "--num_train_epochs", "100", \
                        "--dataset", "DuEE", "--data_dir", f"../data/DuEE/DuEE_e2e_backup_3/pretrain_data/", \
                        "--entity_dic_file", "../data/DuEE/DuEE_e2e/entities.txt", \
                        "--relation_dic_file", "../data/DuEE/DuEE_e2e/relations.txt", \
                        "--loss_lambda","0.3",  "--exp", f"hp_bs{train_batch_size}_lr{lr}_wu{warm_up}_wd{weight_decay}", \
                        "--learning_rate", f"{lr}", \
                        "--train_batch_size", f"{train_batch_size}", \
                        "--gradient_accumulation_steps", f"{train_batch_size // 8}",
                        "--weight_decay", f"{weight_decay}", \
                        "--warm_up", f"{warm_up}", \
                        "--num_train_epochs", "100"
                ])

# def gpu_ok(gpu_usage, mem_usage, max_gpu_usage, max_mem_usage):
#     return gpu_usage <= max_gpu_usage and mem_usage < max_mem_usage


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument("cmd", help="Command to execute", type=str)
#     parser.add_argument("--gpu_usage", default=5, help="Maximum GPU usage on available GPU, default 5%", type=int)
#     parser.add_argument("--mem_usage", default=10, help="Maximum GPU memory usage on available GPU, default 10%", type=int)

#     args = parser.parse_args()

#     smi_command = ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory', '--format=csv,noheader']
#     with subprocess.Popen(smi_command, stdout=subprocess.PIPE) as proc:
#         gpu_info = [[int(y.replace('%', '')) for y in x.decode().strip().split(',')] for x in proc.stdout.readlines()]

#         selected_gpu_id = -1
#         for gpu_id in range(len(gpu_info)):
#             if gpu_ok(gpu_info[gpu_id][0], gpu_info[gpu_id][1], args.gpu_usage, args.mem_usage):
#                 selected_gpu_id = gpu_id
#                 break

#         if selected_gpu_id == -1:
#             exit(selected_gpu_id)

#         print("selected GPU %d" % selected_gpu_id)
#         cmd = 'export CUDA_VISIBLE_DEVICES=%d; %s' % (selected_gpu_id, args.cmd)
#         subprocess.call(cmd, shell=True)