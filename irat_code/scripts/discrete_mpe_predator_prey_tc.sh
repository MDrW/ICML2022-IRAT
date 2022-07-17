#!/bin/sh
env="MPE"
scenario_name="simple_tag_tr"
num_good_agents=2
num_agents=5
num_adv=5
num_landmarks=2
agent_policy="random"


base=( "rmappotrsynrnd" "rmappotrsynrnd" "rmappotrsynrnd" "rmappotrsynrnd" )
# base=( "rmappotrsynrnd" "rmappotrsynrnd" )
exp="exp_train_discrete_tag_new_v2"
algos=( "rnd" "rnd_adv_surgery" "rnd_gdt_surgery" "rnd_adv_gdt_surgery" )
n_train=1
n_rollout=8
n_eval=8
mini_batch=1
episode_length=25
steps=2000000
ppo_epoch=10
gain=0.01
lr=7e-4
critic_lr=7e-4
entropy_coef=0.1
entropy_end_coef=0.01
entropy_change_episode=8000
change_reward_episode=5000

user="mdrw"
project="Discrete_Tag_New_v2"
pres=( "RND" "RND_ADV_SURGERY" "RND_GDT_SURGERY" "RND_ADV_GDT_SURGERY" )
group="RMAPPO"

declare -a seeds=( "1" "22" "333" "4444" "55555" "12345" "67890" )


n=-1
gpunum=0

files=${exp}
mkdir out_logs/${files} &> /dev/null

echo "env is ${env}, exp is ${exp}"
for seed in "${seeds[@]}"
do
    echo "seed is ${seed}:"
    
    CUDA_VISIBLE_DEVICES=${n} nohup python train/train_mpe_trsyn_rnd.py --use_eval --use_valuenorm --use_popart --use_ReLU --env_name ${env} --algorithm_name ${base[0]} --experiment_name ${exp}_${algos[0]} --scenario_name ${scenario_name} --num_good_agents ${num_good_agents} --num_agents ${num_agents} --num_adversaries ${num_adv} --num_landmarks ${num_landmarks} --agent_policy ${agent_policy} --seed ${seed} --n_training_threads ${n_train} --n_rollout_threads ${n_rollout} --n_eval_rollout_threads ${n_eval} --num_mini_batch ${mini_batch} --episode_length ${episode_length} --num_env_steps ${steps} --ppo_epoch ${ppo_epoch} --gain ${gain} --lr ${lr} --critic_lr ${critic_lr} --entropy_coef ${entropy_coef} --entropy_end_coef ${entropy_end_coef} --entropy_change_episode ${entropy_change_episode} --wandb_project ${project} --user_name ${user} --wandb_group ${pres[0]}_${group} --wandb_exp_name ${env}_${seed} --collaborative --scenario_has_diff_rewards --sparse_reward >& out_logs/${files}/${base[0]}_${algos[0]}_${scenario_name}_${seed}.txt &
    if [ ${n} != -1 ]; then
    n=$[($n+1) % ${gpunum}]
    fi
    echo "${algos[0]}_${env}_${seed} start"
    
    
    CUDA_VISIBLE_DEVICES=${n} nohup python train/train_mpe_trsyn_rnd.py --use_eval --use_valuenorm --use_popart --use_ReLU --env_name ${env} --algorithm_name ${base[2]} --experiment_name ${exp}_${algos[2]} --scenario_name ${scenario_name} --num_good_agents ${num_good_agents} --num_agents ${num_agents} --num_adversaries ${num_adv} --num_landmarks ${num_landmarks} --agent_policy ${agent_policy} --seed ${seed} --n_training_threads ${n_train} --n_rollout_threads ${n_rollout} --n_eval_rollout_threads ${n_eval} --num_mini_batch ${mini_batch} --episode_length ${episode_length} --num_env_steps ${steps} --ppo_epoch ${ppo_epoch} --gain ${gain} --lr ${lr} --critic_lr ${critic_lr} --entropy_coef ${entropy_coef} --entropy_end_coef ${entropy_end_coef} --entropy_change_episode ${entropy_change_episode} --wandb_project ${project} --user_name ${user} --wandb_group ${pres[2]}_${group} --wandb_exp_name ${env}_${seed} --gradient_use_surgery --collaborative --scenario_has_diff_rewards --sparse_reward >& out_logs/${files}/${base[2]}_${algos[2]}_${scenario_name}_${seed}.txt &
    if [ ${n} != -1 ]; then
    n=$[($n+1) % ${gpunum}]
    fi
    echo "${algos[2]}_${env}_${seed} start"
    
    
    sleep 10
    
    # COUNT=$(ps -ef |grep python |grep -v "grep" |wc -l)
    # while [ ${COUNT} -gt 0 ];
    # do
    # sleep 600
    # echo ${COUNT}
    # COUNT=$(ps -ef |grep python |grep -v "grep" |wc -l)
    # done
done
