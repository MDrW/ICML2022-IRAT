#!/bin/sh
env="MultiWalker"
n_walkers=2
num_agents=2
forward_reward=1.0
fall_reward=-5.0
drop_reward=-50.0
# terminate_on_fall=True
# one_hot=True
# ir_use_pos=True
max_cycles=500
reward_mech="local"


base=( "rmappotrsyn" )
exp="exp_train_multiwalker_base_new_2"
algos=( "klcp_s2r2" )
idv_clip_flag=2
idv_clip_flag_refine=2


n_train=1
n_rollout=8
n_eval=8
mini_batch=1
episode_length=500
steps=6000000
ppo_epoch=10
gain=0.01
lr=5e-4
critic_lr=5e-4
entropy_coef=0.1
entropy_end_coef=0.01
entropy_change_episode=1200
action_scale=1.0


idv_clip_ratio=0.2
idv_end_clip_ratio=0.5
idv_clip_episodes=1500
team_clip_ratio=0.2
team_end_clip_ratio=0.2
team_clip_episodes=1500
idv_kl_coef=0.
idv_kl_end_coef=1.0
idv_kl_episodes=1500
team_kl_coef=1.5
team_kl_end_coef=0.5
team_kl_episodes=1500

idv_clip_use_time=0
idv_kl_use_time=0
team_kl_use_time=0

change_reward_episode=750
change_use_policy=( "team" "idv" )


user="mdrw"
project="MultiWalker_Base_NEW_v2"
pres=( "KLCP_s2r2" )
group="RMAPPO"


declare -a seeds=( "1" "55555" "333" "4444" "13579" ) 

n=-1
gpunum=2

files=${exp}
mkdir out_logs/${files} &> /dev/null

echo "env is ${env}, exp is ${exp}"
for seed in "${seeds[@]}"
do
    echo "seed is ${seed}:"
    
    CUDA_VISIBLE_DEVICES=${n} nohup python train/train_naive_sisl.py --action_use_clip --std_seperated --use_eval --use_valuenorm --use_popart --use_ReLU --env_name ${env} --algorithm_name ${base[0]} --experiment_name ${exp}_${algos[0]} --n_walkers ${n_walkers} --num_agents ${num_agents} --forward_reward ${forward_reward} --fall_reward ${fall_reward} --drop_reward ${drop_reward} --max_cycles ${max_cycles} --reward_mech ${reward_mech} --terminate_on_fall --one_hot --seed ${seed} --n_training_threads ${n_train} --n_rollout_threads ${n_rollout} --n_eval_rollout_threads ${n_eval} --num_mini_batch ${mini_batch} --episode_length ${episode_length} --num_env_steps ${steps} --ppo_epoch ${ppo_epoch} --gain ${gain} --lr ${lr} --critic_lr ${critic_lr} --entropy_coef ${entropy_coef} --entropy_end_coef ${entropy_end_coef} --entropy_change_episode ${entropy_change_episode} --action_scale ${action_scale} --wandb_project ${project} --user_name ${user} --wandb_group ${pres[0]}_${group} --wandb_exp_name ${env}_${seed} --idv_clip_ratio ${idv_clip_ratio} --idv_end_clip_ratio ${idv_end_clip_ratio} --idv_clip_episodes ${idv_clip_episodes} --team_clip_ratio ${team_clip_ratio} --team_end_clip_ratio ${team_end_clip_ratio} --team_clip_episodes ${team_clip_episodes} --idv_kl_coef ${idv_kl_coef} --idv_kl_end_coef ${idv_kl_end_coef} --idv_kl_episodes ${idv_kl_episodes} --team_kl_coef ${team_kl_coef} --team_kl_end_coef ${team_kl_end_coef} --team_kl_episodes ${team_kl_episodes} --idv_use_shared_obs --idv_use_kl_loss --team_use_kl_loss --idv_clip_flag ${idv_clip_flag} --idv_clip_flag_refine ${idv_clip_flag_refine} --idv_clip_use_time ${idv_clip_use_time} --idv_kl_use_time ${idv_kl_use_time} --team_kl_use_time ${team_kl_use_time} --ir_use_pos >& out_logs/${files}/${algos[0]}_${env}_${seed}.txt &
    if [ ${n} != -1 ]; then
    n=$[($n+1) % ${gpunum}]
    fi
    echo "${algos[0]}_${env}_${seed} start"
    

    
    sleep 10
    
    # COUNT=$(ps -ef |grep python |grep -v "grep" |wc -l)
    # while [ ${COUNT} -gt 0 ];
    # do
    # sleep 600
    # echo ${COUNT}
    # COUNT=$(ps -ef |grep python |grep -v "grep" |wc -l)
    # done
done
