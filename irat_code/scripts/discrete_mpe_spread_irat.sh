#!/bin/sh
env="MPE"
scenario_name="simple_spread_ctr"
num_good_agents=4
num_agents=4
num_adv=2
num_landmarks=2
game_mode="easy"


# base=( "rmappotrsyn" "rmappotrsyn" "rmappotrsyn" )
base=( "rmappotrsyn" )
exp="exp_train_discrete_spread_v0"
# algos=( "klcp" "klcp_change_team" "klcp_change_idv" )
algos=( "klcp_s2r2_v0" )
idv_clip_flag=2
idv_clip_flag_refine=2

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
entropy_coef=0.15
entropy_end_coef=0.01
entropy_change_episode=9000


idv_clip_ratio=3.0
idv_end_clip_ratio=0.5
idv_clip_episodes=10000
team_clip_ratio=0.2
team_end_clip_ratio=0.2
team_clip_episodes=10000
idv_kl_coef=0.2
idv_kl_end_coef=2.0
idv_kl_episodes=10000
team_kl_coef=1.5
team_kl_end_coef=0.
team_kl_episodes=6000

idv_clip_use_time=0
idv_kl_use_time=0
team_kl_use_time=0

change_reward_episode=5000
change_use_policy=( "team" "idv" )


user="mdrw"
project="Discrete_Spread_Ctr_v0"
# pres=( "KLCP" "KLCP_CHANGE_TEAM" "KLCP_CHANGE_IDV" )
pres=( "KLCP_s2r2_v0" )
group="RMAPPO"


declare -a seeds=( "1" "22" "333" "4444" "55555" "12345" "67890" )

n=-1
gpunum=2

files=${exp}
mkdir out_logs/${files} &> /dev/null

echo "env is ${env}, exp is ${exp}"
for seed in "${seeds[@]}"
do
    echo "seed is ${seed}:"
    
    CUDA_VISIBLE_DEVICES=${n} nohup python train/train_mpe_trsyn.py --use_eval --use_valuenorm --use_popart --use_ReLU --env_name ${env} --algorithm_name ${base[0]} --experiment_name ${exp}_${algos[0]} --scenario_name ${scenario_name} --num_good_agents ${num_good_agents} --num_agents ${num_agents} --num_adversaries ${num_adv} --num_landmarks ${num_landmarks} --game_mode ${game_mode} --seed ${seed} --n_training_threads ${n_train} --n_rollout_threads ${n_rollout} --n_eval_rollout_threads ${n_eval} --num_mini_batch ${mini_batch} --episode_length ${episode_length} --num_env_steps ${steps} --ppo_epoch ${ppo_epoch} --gain ${gain} --lr ${lr} --critic_lr ${critic_lr} --entropy_coef ${entropy_coef} --entropy_end_coef ${entropy_end_coef} --entropy_change_episode ${entropy_change_episode} --wandb_project ${project} --user_name ${user} --wandb_group ${pres[0]}_${group} --wandb_exp_name ${env}_${seed} --idv_clip_ratio ${idv_clip_ratio} --idv_end_clip_ratio ${idv_end_clip_ratio} --idv_clip_episodes ${idv_clip_episodes} --team_clip_ratio ${team_clip_ratio} --team_end_clip_ratio ${team_end_clip_ratio} --team_clip_episodes ${team_clip_episodes} --idv_kl_coef ${idv_kl_coef} --idv_kl_end_coef ${idv_kl_end_coef} --idv_kl_episodes ${idv_kl_episodes} --team_kl_coef ${team_kl_coef} --team_kl_end_coef ${team_kl_end_coef} --team_kl_episodes ${team_kl_episodes} --idv_use_shared_obs --idv_use_kl_loss --team_use_kl_loss --scenario_has_diff_rewards --idv_clip_flag ${idv_clip_flag} --idv_clip_flag_refine ${idv_clip_flag_refine} --idv_clip_use_time ${idv_clip_use_time} --idv_kl_use_time ${idv_kl_use_time} --team_kl_use_time ${team_kl_use_time} >& out_logs/${files}/${base[0]}_${algos[0]}_${scenario_name}_${seed}.txt &
    # if [ ${n} != -1 ]; then
    # n=$[($n+1) % ${gpunum}]
    # fi
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
