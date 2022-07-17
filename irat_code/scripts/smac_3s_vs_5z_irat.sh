#!/bin/sh
env="StarCraft2"
map="3s_vs_5z"
algos=( "mappotrsyn" )

user="mdrw"
project="3s_vs_5z_sigma_v0"
pres=( "KLCP_s2r2_v0" )
idv_clip_flag=2
idv_clip_flag_refine=2
group="MAPPOTRSYN"

exp="train_3s_vs_5z_sigma_v0"
# algos=( "idv" "team" "rs" "change" )
n_train=1
n_rollout=8
n_eval=1
num_mini_batch=1
episode_length=400
num_env_steps=3200000
ppo_epoch=15
clip=0.05

idv_clip_ratio=1.0
idv_end_clip_ratio=0.05
idv_clip_episodes=1000
team_clip_ratio=0.05
team_end_clip_ratio=0.05
team_clip_episodes=1000
idv_kl_coef=0.
idv_kl_end_coef=0.5
idv_kl_episodes=1000
team_kl_coef=9.0
team_kl_end_coef=1.0
team_kl_episodes=1000

idv_clip_use_time=0
idv_kl_use_time=0
team_kl_use_time=0

declare -a seeds=( "1" "2" "3" "4" "6" )


n=0
gpunum=2

files=${exp}
mkdir out_logs/${files} &> /dev/null

echo "env is ${env}, map is ${map}, algo is ${algos[0]}"


for seed in "${seeds[@]}"
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=${n} python train/train_smac_trsyn.py --env_name ${env} --algorithm_name ${algos[0]} --experiment_name ${exp}_${pres[0]} --map_name ${map} --seed ${seed} --n_training_threads ${n_train} --n_rollout_threads ${n_rollout} --n_eval_rollout_threads ${n_eval} --num_mini_batch ${num_mini_batch} --episode_length ${episode_length} --num_env_steps ${num_env_steps} --ppo_epoch ${ppo_epoch} --use_value_active_masks --use_eval --wandb_project ${project} --user_name ${user} --wandb_group ${pres[0]}_${group} --wandb_exp_name ${env}_${map}_${seed} --idv_clip_ratio ${idv_clip_ratio} --idv_end_clip_ratio ${idv_end_clip_ratio} --idv_clip_episodes ${idv_clip_episodes} --team_clip_ratio ${team_clip_ratio} --team_end_clip_ratio ${team_end_clip_ratio} --team_clip_episodes ${team_clip_episodes} --idv_kl_coef ${idv_kl_coef} --idv_kl_end_coef ${idv_kl_end_coef} --idv_kl_episodes ${idv_kl_episodes} --team_kl_coef ${team_kl_coef} --team_kl_end_coef ${team_kl_end_coef} --team_kl_episodes ${team_kl_episodes} --idv_use_shared_obs --idv_use_kl_loss --team_use_kl_loss --clip_param ${clip} --use_recurrent_policy --use_stacked_frames --stacked_frames 4 --idv_clip_flag ${idv_clip_flag} --idv_clip_flag_refine ${idv_clip_flag_refine} --idv_clip_use_time ${idv_clip_use_time} --idv_kl_use_time ${idv_kl_use_time} --team_kl_use_time ${team_kl_use_time} >& out_logs/${files}/${algos[0]}_${pres[0]}_${env}_${map}_${seed}.txt &
    if [ ${n} != -1 ]; then
    n=$[($n+1) % ${gpunum}]
    fi
    echo "${algos[0]}_${pres[0]}_${env}_${map}_${seed} start"
    sleep 5
    
    # CUDA_VISIBLE_DEVICES=${n} python train/train_smac.py --env_name ${env} --algorithm_name ${algos[1]} --experiment_name ${exp}_${pres[1]} --map_name ${map} --seed ${seed} --n_training_threads ${n_train} --n_rollout_threads ${n_rollout} --n_eval_rollout_threads ${n_eval} --num_mini_batch ${num_mini_batch} --episode_length ${episode_length} --num_env_steps ${num_env_steps} --ppo_epoch ${ppo_epoch} --use_value_active_masks --use_eval --wandb_project ${project} --user_name ${user} --wandb_group ${pres[1]}_${group} --wandb_exp_name ${env}_${map}_${seed} --use_team_reward >& out_logs/${files}/${algos[1]}_${pres[1]}_${env}_${map}_${seed}.txt &
    # if [ ${n} != -1 ]; then
    # n=$[($n+1) % ${gpunum}]
    # fi
    # echo "${algos[1]}_${pres[1]}_${env}_${map}_${seed} start"
    
done
