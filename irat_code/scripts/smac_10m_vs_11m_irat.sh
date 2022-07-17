#!/bin/sh
env="StarCraft2"
map="10m_vs_11m"
algos=( "rmappotrsyn" )

user="mdrw"
project="10m_vs_11m_new"
pres=( "KLCP_s2r2_v0" )
idv_clip_flag=( 2 )
idv_clip_flag_refine=( 2 )
group="RMAPPOTRSYN"

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

idv_clip_ratio=0.5
idv_end_clip_ratio=0.2
idv_clip_episodes=1250
team_clip_ratio=0.2
team_end_clip_ratio=0.2
team_clip_episodes=1250
idv_kl_coef=0.
idv_kl_end_coef=0.1
idv_kl_episodes=1250
team_kl_coef=30.0
team_kl_end_coef=2.0
team_kl_episodes=1250

idv_clip_use_time=0
idv_kl_use_time=0
team_kl_use_time=0

declare -a seeds=( "2" "3" "6" "7" "9" )

n=0
gpunum=2

files=${exp}
mkdir out_logs/${files} &> /dev/null

echo "env is ${env}, map is ${map}, algo is ${algos[0]}"


for seed in "${seeds[@]}"
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=${n} python train/train_smac_trsyn.py --env_name ${env} --algorithm_name ${algos[0]} --experiment_name ${exp}_${pres[0]} --map_name ${map} --seed ${seed} --n_training_threads ${n_train} --n_rollout_threads ${n_rollout} --n_eval_rollout_threads ${n_eval} --num_mini_batch ${num_mini_batch} --episode_length ${episode_length} --num_env_steps ${num_env_steps} --ppo_epoch ${ppo_epoch} --use_value_active_masks --use_eval --wandb_project ${project} --user_name ${user} --wandb_group ${pres[0]}_${group} --wandb_exp_name ${env}_${map}_${seed} --idv_clip_ratio ${idv_clip_ratio} --idv_end_clip_ratio ${idv_end_clip_ratio} --idv_clip_episodes ${idv_clip_episodes} --team_clip_ratio ${team_clip_ratio} --team_end_clip_ratio ${team_end_clip_ratio} --team_clip_episodes ${team_clip_episodes} --idv_kl_coef ${idv_kl_coef} --idv_kl_end_coef ${idv_kl_end_coef} --idv_kl_episodes ${idv_kl_episodes} --team_kl_coef ${team_kl_coef} --team_kl_end_coef ${team_kl_end_coef} --team_kl_episodes ${team_kl_episodes} --idv_use_shared_obs --idv_use_kl_loss --team_use_kl_loss --clip_param ${clip} --gain ${gain} --idv_clip_flag ${idv_clip_flag[0]} --idv_clip_flag_refine ${idv_clip_flag_refine[0]} --idv_clip_use_time ${idv_clip_use_time} --idv_kl_use_time ${idv_kl_use_time} --team_kl_use_time ${team_kl_use_time} >& out_logs/${files}/${algos[0]}_${pres[0]}_${env}_${map}_${seed}.txt &
    if [ ${n} != -1 ]; then
    n=$[($n+1) % ${gpunum}]
    fi
    echo "${algos[0]}_${pres[0]}_${env}_${map}_${seed} start"
    sleep 5

    # echo "seed is ${seed}:"
    # CUDA_VISIBLE_DEVICES=${n} python train/train_smac_trsyn.py --env_name ${env} --algorithm_name ${algos[0]} --experiment_name ${exp}_${pres[1]} --map_name ${map} --seed ${seed} --n_training_threads ${n_train} --n_rollout_threads ${n_rollout} --n_eval_rollout_threads ${n_eval} --num_mini_batch ${num_mini_batch} --episode_length ${episode_length} --num_env_steps ${num_env_steps} --ppo_epoch ${ppo_epoch} --use_value_active_masks --use_eval --wandb_project ${project} --user_name ${user} --wandb_group ${pres[1]}_${group} --wandb_exp_name ${env}_${map}_${seed} --idv_clip_ratio ${idv_clip_ratio} --idv_end_clip_ratio ${idv_end_clip_ratio} --idv_clip_episodes ${idv_clip_episodes} --team_clip_ratio ${team_clip_ratio} --team_end_clip_ratio ${team_end_clip_ratio} --team_clip_episodes ${team_clip_episodes} --idv_kl_coef ${idv_kl_coef} --idv_kl_end_coef ${idv_kl_end_coef} --idv_kl_episodes ${idv_kl_episodes} --team_kl_coef ${team_kl_coef} --team_kl_end_coef ${team_kl_end_coef} --team_kl_episodes ${team_kl_episodes} --idv_use_shared_obs --idv_use_kl_loss --team_use_kl_loss --clip_param ${clip} --gain ${gain} --idv_clip_flag ${idv_clip_flag[1]} --idv_clip_flag_refine ${idv_clip_flag_refine[1]} --idv_clip_use_time ${idv_clip_use_time} --idv_kl_use_time ${idv_kl_use_time} --team_kl_use_time ${team_kl_use_time} >& out_logs/${files}/${algos[0]}_${pres[1]}_${env}_${map}_${seed}.txt &
    # if [ ${n} != -1 ]; then
    # n=$[($n+1) % ${gpunum}]
    # fi
    # echo "${algos[0]}_${pres[1]}_${env}_${map}_${seed} start"
    # sleep 5
    
done
