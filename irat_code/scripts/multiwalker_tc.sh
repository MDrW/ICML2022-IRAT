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
reward_mech=( "idv" "team" "rs" "change" "local" "local" )


base=( "rmappo" "rmappo" "rmappo" "rmappo" "rmappotrsynrnd" "rmappotrsynrnd" )
exp="exp_train_multiwalker_base_new_2"
algos=( "idv" "team" "rs" "change" "rnd" "rnd_gdt_sgy" )
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
change_reward_episode=750

user="mdrw"
project="MultiWalker_Base_NEW_v2"
pres=( "IDV" "TEAM" "RS" "CHANGE" "RND" "RND_GDT_SGY" )
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
    
    CUDA_VISIBLE_DEVICES=${n} nohup python train/train_naive_sisl.py --action_use_clip --std_seperated --use_eval --use_valuenorm --use_popart --use_ReLU --env_name ${env} --algorithm_name ${base[4]} --experiment_name ${exp}_${algos[4]} --n_walkers ${n_walkers} --num_agents ${num_agents} --forward_reward ${forward_reward} --fall_reward ${fall_reward} --drop_reward ${drop_reward} --max_cycles ${max_cycles} --terminate_on_fall --one_hot --seed ${seed} --n_training_threads ${n_train} --n_rollout_threads ${n_rollout} --n_eval_rollout_threads ${n_eval} --num_mini_batch ${mini_batch} --episode_length ${episode_length} --num_env_steps ${steps} --ppo_epoch ${ppo_epoch} --gain ${gain} --lr ${lr} --critic_lr ${critic_lr} --entropy_coef ${entropy_coef} --entropy_end_coef ${entropy_end_coef} --entropy_change_episode ${entropy_change_episode} --action_scale ${action_scale} --wandb_project ${project} --user_name ${user} --wandb_group ${pres[4]}_${group} --wandb_exp_name ${env}_${seed} --reward_mech ${reward_mech[4]} --ir_use_pos >& out_logs/${files}/${algos[4]}_${env}_${reward_mech[4]}_${seed}.txt &
    #if [ ${n} != -1 ]; then
    #n=$[($n+1) % ${gpunum}]
    #fi
    echo "${algos[4]}_${env}_${reward_mech[4]}_${seed} start"
    #sleep 10
    
    CUDA_VISIBLE_DEVICES=${n} nohup python train/train_naive_sisl.py --action_use_clip --std_seperated --use_eval --use_valuenorm --use_popart --use_ReLU --env_name ${env} --algorithm_name ${base[5]} --experiment_name ${exp}_${algos[5]} --n_walkers ${n_walkers} --num_agents ${num_agents} --forward_reward ${forward_reward} --fall_reward ${fall_reward} --drop_reward ${drop_reward} --max_cycles ${max_cycles} --terminate_on_fall --one_hot --seed ${seed} --n_training_threads ${n_train} --n_rollout_threads ${n_rollout} --n_eval_rollout_threads ${n_eval} --num_mini_batch ${mini_batch} --episode_length ${episode_length} --num_env_steps ${steps} --ppo_epoch ${ppo_epoch} --gain ${gain} --lr ${lr} --critic_lr ${critic_lr} --entropy_coef ${entropy_coef} --entropy_end_coef ${entropy_end_coef} --entropy_change_episode ${entropy_change_episode} --action_scale ${action_scale} --wandb_project ${project} --user_name ${user} --wandb_group ${pres[5]}_${group} --wandb_exp_name ${env}_${seed} --reward_mech ${reward_mech[5]} --gradient_use_surgery --ir_use_pos >& out_logs/${files}/${algos[5]}_${env}_${reward_mech[5]}_${seed}.txt &
    #if [ ${n} != -1 ]; then
    #n=$[($n+1) % ${gpunum}]
    #fi
    echo "${algos[5]}_${env}_${reward_mech[5]}_${seed} start"
    
    sleep 10
    
    # COUNT=$(ps -ef |grep python |grep -v "grep" |wc -l)
    # while [ ${COUNT} -gt 0 ];
    # do
    # sleep 600
    # echo ${COUNT}
    # COUNT=$(ps -ef |grep python |grep -v "grep" |wc -l)
    # done
done
