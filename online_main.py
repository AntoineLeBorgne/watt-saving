from onguard_service import onguard_service
import datetime
import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from tqdm import tqdm


# Driver code for training
def setup_and_train(n_agents, mode, do_ray_init=True):
    global REWARDS
    global ADDITIONAL_RULES

    env_config = {
        'n_agents': n_agents,
        'nmax_rounds': NMAX_ROUNDS,
        'debug': False,
        'mode': mode,
        'rew_timestep': REWARDS['rew_timestep'][0],
        'rew_offer_received': REWARDS['rew_offer_received'][0],
        'rew_guard_given': REWARDS['rew_guard_given'][0],
        'rew_active_remains': REWARDS['rew_active_remains'][0],
        'additional_rules': ADDITIONAL_RULES[0],
    }

    # Define experiment details
    env_name = "onguard_service"
    tune.register_env(env_name, lambda env_config: onguard_service(env_config))

    # Get environment obs, action spaces and number of agents
    env = onguard_service(env_config)
    obs_space = env.observation_space
    act_space = env.action_space
    n_agents = env.n_agents

    # Create a policy mapping
    def gen_policy():
        return (None, obs_space, act_space, {})

    policy_graphs = {}
    for i in range(n_agents):
        policy_graphs['agent-' + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id):
        # return 'agent-' + str(agent_id)
        return 'agent-' + str(agent_id)

    # Initialize ray and run
    if do_ray_init:
        ray.init()

    # trick to have these parameters in the name of the rllib trial
    # env_config['n_agents'] = tune.grid_search([n_agents])
    # env_config['mode'] = tune.grid_search([mode])

    config = {
        'env': env_name,
        # 'callbacks': MyCallbacks,  # disable callbacks for faster training
        'train_batch_size': TRAIN_BATCH_SIZE,  # 4000 by default
        'normalize_actions': False,  # needed when algo is SAC
        "log_level": "WARN",
        "num_workers": N_WORKERS,
        "num_cpus_for_driver": 1,
        "framework": "torch",
        "num_gpus_per_worker": 1,
        "lr": 1e-4,
        # "gamma": 1.0,
        "multiagent": {
            "policies": policy_graphs,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "env_config": env_config,
    }

    analysis = tune.run(
        name='onguard_service' + '_n_agents=' + str(n_agents) + '_mode=' + str(mode),
        local_dir="/home/antoine/Documents/PycharmProjects/offline-watt-saving/ray_results",
        run_or_experiment=RL_ALGO,
        reuse_actors=False,
        verbose=0,
        # progress_reporter = 'JupyterNotebookReporter',
        checkpoint_at_end=True,
        config=config,
        stop={
            "training_iteration": TRAINING_ITERATIONS,
        },
    )

    return analysis, config


if __name__ == '__main__':

    TRAIN_ONLY = False
    EVALUATION = True
    UNIT_TEST = False
    GAME_THEORY = False

    N_AGENTSS = [8, 10]
    MODES = ['free', 'recommended', 'imposed']
    MODES = ['imposed']
    N_RUNS = 5
    TRAINING_ITERATIONS = 10
    # Values of the rewards
    # (if multiple values per reward, then Ray Tune will try and evaluate both)
    REWARDS = {
        'rew_offer_received': [-0.01],
        'rew_guard_given': [-1.0],
        'rew_active_remains': [-0.9],
        'rew_timestep': [-0.01],
    }
    # Maximum number of offerers per episode (aka low-activity period)
    # (if multiple values per reward, then Ray Tune will try and evaluate both)
    ADDITIONAL_RULES = [False]
    CHEATER_AGENT = False
    CHEATER_COUNTER_MEASURE = False
    AGENT_TURN_RECOMMENDED = True
    AGENT_TURN_IMPOSED = True
    COOP_BLACKLIST = True
    CIRCUIT_BREAKER = False
    DLT_CHECK = True
    ENERGY_DISPERSION = 0.1
    N_TESTS_PER_RUN = 100
    NMAX_ROUNDS = 10
    RL_ALGO = 'PPO'  # A2C', 'A3C', 'PG', 'MARWIL', 'PPO' # Multinotsupported 'DQN'
    N_WORKERS = 1
    CURRENT_EPISODE = -1
    TRAIN_BATCH_SIZE = 1000  # 4000 by default for RLlib PPO
    # RATIO = int(4000/TRAIN_BATCH_SIZE)
    # TRAINING_ITERATIONS = 100 * RATIO # 1 iterations <=> 10000 timesteps TD3, 4000 timesteps for PPO, A2C !

    print('TRAINING_ITERATIONS:', TRAINING_ITERATIONS)

    EVAL_CSV_FILE_RESULTS = "/home/antoine/Documents/PycharmProjects/offline-watt-saving/metric_results_onguard_service.csv"
    CURRENT_ENV_INSTANCE = 0
    N_DONE = 0
    debug = False

    if EVALUATION:
        do_ray_init = True

        columns = [
            'n_agents',
            'mode',
            'n_runs',
            'n_tests',
            'efficiency_mean', 'efficiency_std',
            'safety_mean', 'safety_std',
            'ic_mean', 'ic_std',
            'jain_mean', 'jain_std',
        ]
        results = pd.DataFrame(columns=columns)

        t0 = datetime.datetime.now()

        for n_agents in N_AGENTSS:
            print('n_agents:', n_agents)

            for mode in MODES:
                print('mode:', mode)

                env_config = {
                    'n_agents': n_agents,
                    'nmax_rounds': NMAX_ROUNDS,
                    'debug': False,
                    'mode': mode,
                    'rew_timestep': REWARDS['rew_timestep'][0],
                    'rew_offer_received': REWARDS['rew_offer_received'][0],
                    'rew_guard_given': REWARDS['rew_guard_given'][0],
                    'rew_active_remains': REWARDS['rew_active_remains'][0],
                    'additional_rules': ADDITIONAL_RULES[0],
                }

                # array to store the utility for all scenarios, for all agents across the different tests
                utilities = {}
                jains = {}

                scenarios = ['CCC', 'CDD', 'DCC', 'DDD']
                # scenarios = ['CCC']

                for scenario in scenarios:
                    utilities[scenario] = np.zeros(N_RUNS * n_agents * N_TESTS_PER_RUN).reshape(
                        (n_agents, N_RUNS, N_TESTS_PER_RUN))
                    jains[scenario] = np.zeros(N_RUNS * n_agents * N_TESTS_PER_RUN).reshape(
                        (n_agents, N_RUNS, N_TESTS_PER_RUN))

                for run in range(N_RUNS):

                    print('run:', run)
                    t1 = datetime.datetime.now()
                    analysis, config = setup_and_train(n_agents, mode, do_ray_init)
                    do_ray_init = False
                    t2 = datetime.datetime.now()
                    print('run:,', run, ', training duration:', t2 - t1)
                    break
                    trials = analysis.trials
                    print('trials:', trials)
                    checkpoint = analysis.get_best_checkpoint(trial=trials[0],
                                                              metric="episode_reward_mean",
                                                              mode="max")

                    checkpoints = analysis.get_trial_checkpoints_paths(trial=trials[0],
                                                                       metric="episode_reward_mean")
                    print('listes des checkpoints :', checkpoints)
                    print('checkpoint %s' % str(checkpoint))
                    # print('config:', config)

                    for scenario in scenarios:

                        print('##### SCENARIO %s ######' % scenario)

                        # Restore a RLlib Trainer from the checkpoint.
                        trainer = PPOTrainer(config=config)
                        trainer.restore(checkpoint)
                        print('trainer:', trainer)

                        # print('.get_policy().model.base_model.summary()')
                        # print(trainer.get_policy().model.base_model.summary())

                        # launch the environment
                        def env_creator(_):
                            return onguard_service(env_config=env_config)


                        env = onguard_service(env_config=env_config)
                        env_name = "onguard_service"
                        register_env(env_name, env_creator)

                        for t in tqdm(range(N_TESTS_PER_RUN)):
                            if debug:
                                print('==== Test %d ====' % (t))

                            obs = env.reset()

                            if debug:
                                print('obs:', obs)

                            last_infos = {}

                            for s in range(NMAX_ROUNDS):
                                if debug:
                                    print('--- step %d ---' % s)

                                if scenario == 'CCC':
                                    a0 = trainer.compute_action(obs[0], policy_id="agent-0")
                                elif scenario == 'CDD':
                                    a0 = trainer.compute_action(obs[0], policy_id="agent-0")
                                elif scenario == 'DCC':
                                    a0 = np.array([0] * 1 + [1] * (n_agents - 1) + [0] * (n_agents - 1))
                                elif scenario == 'DDD':
                                    a0 = np.array([0] * 1 + [1] * (n_agents - 1) + [0] * (n_agents - 1))
                                else:
                                    print('ERROR: scenario %s not implemented' % scenario)
                                    break
                                actions = {0: a0}

                                for x in range(1, n_agents):
                                    policy_x = "agent-" + str(x)
                                    if scenario == 'CCC':
                                        ax = trainer.compute_action(obs[x], policy_id=policy_x)
                                    elif scenario == 'CDD':
                                        ax = np.array([0] * 1 + [1] * (n_agents - 1) + [0] * (n_agents - 1))
                                    elif scenario == 'DCC':
                                        ax = trainer.compute_action(obs[x], policy_id=policy_x)
                                    elif scenario == 'DDD':
                                        ax = np.array([0] * 1 + [1] * (n_agents - 1) + [0] * (n_agents - 1))
                                    else:
                                        print('ERROR: scenario %s not implemented' % scenario)
                                        break
                                    actions[x] = ax

                                if debug:
                                    print('actions:', actions)

                                obs, rewards, dones, infos = env.step(actions)

                                last_infos = infos.copy()

                                if debug:
                                    print('obs:', obs)
                                    print('rewards:', rewards)
                                    print('dones:', dones)
                                    print('infos:', infos)

                                for a in range(n_agents):
                                    utilities[scenario][a, run, t] += rewards[a]

                                if dones['__all__'] is True:
                                    break

                            for a in range(n_agents):
                                jains[scenario][a, run, t] = float(last_infos[a])

                        trainer.cleanup()

                # print('utilities[%s]:'%scenario, utilities[scenario])

                # reshape (n_agents, runs*iterations)
                utilities['CCC'] = utilities['CCC'].reshape(n_agents, N_RUNS * N_TESTS_PER_RUN)
                utilities['CDD'] = utilities['CDD'].reshape(n_agents, N_RUNS * N_TESTS_PER_RUN)
                utilities['DCC'] = utilities['DCC'].reshape(n_agents, N_RUNS * N_TESTS_PER_RUN)
                utilities['DDD'] = utilities['DDD'].reshape(n_agents, N_RUNS * N_TESTS_PER_RUN)
                print("utilities['CCC'].shape:", utilities['CCC'].shape)

                social_welfares_t = utilities['CCC'].sum(axis=0)
                social_welfares_median = np.median(social_welfares_t)  # anciennement .mean
                social_welfares_mean = social_welfares_t.mean()
                social_welfares_std = social_welfares_t.std()

                print('social_welfares_t:', social_welfares_t)
                print('social_welfares_mean:', social_welfares_mean)
                print('social_welfares_median:', social_welfares_median)
                print('social_welfares_std:', social_welfares_std)

                social_welfares_opt = REWARDS['rew_timestep'][0] * 2 + REWARDS['rew_offer_received'][0]
                social_welfares_opt *= (n_agents - 1)
                social_welfares_opt += REWARDS['rew_timestep'][0] * 2 + REWARDS['rew_guard_given'][0]
                print('social_welfares_opt:', social_welfares_opt)

                social_welfares_min = REWARDS['rew_timestep'][0] * NMAX_ROUNDS + REWARDS['rew_active_remains'][0]
                social_welfares_min *= n_agents
                print('social_welfares_min:', social_welfares_min)

                efficiency_mean = (social_welfares_mean - social_welfares_min) / (
                        social_welfares_opt - social_welfares_min)
                efficiency_std = social_welfares_std / (social_welfares_opt - social_welfares_min)
                print('efficiency:%.2f+-%.2f' % (efficiency_mean, efficiency_std))

                u_ccc_0 = utilities['CCC'][0]
                u_cdd_0 = utilities['CDD'][0]
                u_dcc_0 = utilities['DCC'][0]
                u_ddd_0 = utilities['DDD'][0]

                # 'rew_offer_received': [-0.01],
                # 'rew_guard_given': [-1.0],
                # 'rew_active_remains': [-0.9],
                # 'rew_timestep': [-0.01],

                R = REWARDS['rew_timestep'][0] * 2 + REWARDS['rew_offer_received'][0]  # -0.03
                T = REWARDS['rew_timestep'][0] * 2 + REWARDS['rew_offer_received'][0]  # -0.03
                S = REWARDS['rew_timestep'][0] * NMAX_ROUNDS + REWARDS['rew_guard_given'][0]  # -1.1
                P = REWARDS['rew_timestep'][0] * 10 + REWARDS['rew_active_remains'][0]  # -0.03

                print('T:', T)
                print('S:', S)
                scale = T - S  # -0.03 + 1.1 = 1.07
                print('scale1:', scale)

                safety = (u_cdd_0 - u_ddd_0) / scale
                ic = (u_ccc_0 - u_dcc_0) / scale

                safety_mean = safety.mean() + 1
                safety_std = safety.std()
                ic_mean = ic.mean()
                ic_std = ic.std()

                print('safety=%0.2f+-%0.2f' % (safety.mean(), safety.mean()))
                print('ic=%0.2f+-%0.2f' % (ic.mean(), ic.std()))

                # reshape (n_agents, runs*iterations)
                jains['CCC'] = jains['CCC'].reshape(n_agents, N_RUNS * N_TESTS_PER_RUN)
                # print("xm: jains['CCC']:", jains['CCC'])
                # print("jains['CCC'].shape:", jains['CCC'].shape)
                jains_ccc_0 = jains['CCC'][0]
                jain_mean = jains_ccc_0.mean()
                jain_std = jains_ccc_0.std()

                print('jain=%0.2f+-%0.2f' % (jain_mean, jain_std))

                result = {
                    'n_agents': n_agents,
                    'mode': mode,
                    'n_runs': N_RUNS,
                    'n_tests': N_TESTS_PER_RUN,
                    'efficiency_mean': efficiency_mean,
                    'efficiency_std': efficiency_std,
                    'safety_mean': safety_mean,
                    'safety_std': safety_std,
                    'ic_mean': ic_mean,
                    'ic_std': ic_std,
                    'jain_mean': jain_mean,
                    'jain_std': jain_std,
                }
                print('result:', result)
                results = results.append(result, ignore_index=True)
                results.to_csv(EVAL_CSV_FILE_RESULTS, header=False, index=False, mode='a')

    """t3 = datetime.datetime.now()
    print('datetime:', t3)
    print('duration:', t3 - t0)"""
