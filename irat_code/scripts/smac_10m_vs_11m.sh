#!/bin/sh
env="StarCraft2"
map="10m_vs_11m"
algos=( "rmappo" "rmappo" "rmappo" "rmappo" )

user="mdrw"
project="10m_vs_11m_new"
pres=( "IDV" "TEAM" "RS" "CHANGE" )
group="RMAPPO"

exp="train_10m_vs_11m"
n_train=1
n_rollout=8
n_eval=1
num_mini_batch=1
episode_length=400
num_env_steps=4000000
ppo_epoch=10
clip=0.2
gain=0.01

change_reward_episode=625

declare -a seeds=( "2" "3" "6" "7" "9" )


n=0
gpunum=2

files=${exp}
mkdir out_logs/${files} &> /dev/null

echo "env is ${env}, map is ${map}, algo is ${algos[0]}"


for seed in "${seeds[@]}"
do
    echo "seed is ${seed}:"

    CUDA_VISIBLE_DEVICES=${n} python train/train_smac.py --env_name ${env} --algorithm_name ${algos[0]} --experiment_name ${exp}_${pres[0]} --map_name ${map} --seed ${seed} --n_training_threads ${n_train} --n_rollout_threads ${n_rollout} --n_eval_rollout_threads ${n_eval} --num_mini_batch ${num_mini_batch} --episode_length ${episode_length} --num_env_steps ${num_env_steps} --ppo_epoch ${ppo_epoch} --use_value_active_masks --use_eval --wandb_project ${project} --user_name ${user} --wandb_group ${pres[0]}_${group} --wandb_exp_name ${env}_${map}_${seed} --clip_param ${clip} --gain ${gain} >& out_logs/${files}/${algos[0]}_${pres[0]}_${env}_${map}_${seed}.txt &
    if [ ${n} != -1 ]; then
    n=$[($n+1) % ${gpunum}]
    fi
    echo "${algos[0]}_${pres[0]}_${env}_${map}_${seed} start"
    #sleep 5
    
    CUDA_VISIBLE_DEVICES=${n} python train/train_smac.py --env_name ${env} --algorithm_name ${algos[1]} --experiment_name ${exp}_${pres[1]} --map_name ${map} --seed ${seed} --n_training_threads ${n_train} --n_rollout_threads ${n_rollout} --n_eval_rollout_threads ${n_eval} --num_mini_batch ${num_mini_batch} --episode_length ${episode_length} --num_env_steps ${num_env_steps} --ppo_epoch ${ppo_epoch} --use_value_active_masks --use_eval --wandb_project ${project} --user_name ${user} --wandb_group ${pres[1]}_${group} --wandb_exp_name ${env}_${map}_${seed} --clip_param ${clip} --gain ${gain} --use_team_reward >& out_logs/${files}/${algos[1]}_${pres[1]}_${env}_${map}_${seed}.txt &
    if [ ${n} != -1 ]; then
    n=$[($n+1) % ${gpunum}]
    fi
    echo "${algos[1]}_${pres[1]}_${env}_${map}_${seed} start"
    #sleep 5

    CUDA_VISIBLE_DEVICES=${n} python train/train_smac.py --env_name ${env} --algorithm_name ${algos[2]} --experiment_name ${exp}_${pres[2]} --map_name ${map} --seed ${seed} --n_training_threads ${n_train} --n_rollout_threads ${n_rollout} --n_eval_rollout_threads ${n_eval} --num_mini_batch ${num_mini_batch} --episode_length ${episode_length} --num_env_steps ${num_env_steps} --ppo_epoch ${ppo_epoch} --use_value_active_masks --use_eval --wandb_project ${project} --user_name ${user} --wandb_group ${pres[2]}_${group} --wandb_exp_name ${env}_${map}_${seed} --clip_param ${clip} --gain ${gain} --reward_shaping >& out_logs/${files}/${algos[2]}_${pres[2]}_${env}_${map}_${seed}.txt &
    if [ ${n} != -1 ]; then
    n=$[($n+1) % ${gpunum}]
    fi
    echo "${algos[2]}_${pres[2]}_${env}_${map}_${seed} start"
    # sleep 5

    CUDA_VISIBLE_DEVICES=${n} python train/train_smac.py --env_name ${env} --algorithm_name ${algos[3]} --experiment_name ${exp}_${pres[3]} --map_name ${map} --seed ${seed} --n_training_threads ${n_train} --n_rollout_threads ${n_rollout} --n_eval_rollout_threads ${n_eval} --num_mini_batch ${num_mini_batch} --episode_length ${episode_length} --num_env_steps ${num_env_steps} --ppo_epoch ${ppo_epoch} --use_value_active_masks --use_eval --wandb_project ${project} --user_name ${user} --wandb_group ${pres[3]}_${group} --wandb_exp_name ${env}_${map}_${seed} --clip_param ${clip} --gain ${gain} --change_reward --change_reward_episode ${change_reward_episode} >& out_logs/${files}/${algos[3]}_${pres[3]}_${env}_${map}_${seed}.txt &
    if [ ${n} != -1 ]; then
    n=$[($n+1) % ${gpunum}]
    fi
    echo "${algos[3]}_${pres[3]}_${env}_${map}_${seed} start"
    sleep 5
    
done
