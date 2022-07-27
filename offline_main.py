from transformers import AutoTokenizer
import datetime
import numpy as np
import pandas as pd
from ray.tune.registry import register_env
from onguard_service import onguard_service
from Agent import Agent
from tqdm import tqdm


def env_creator(_):
    return onguard_service(env_config=env_config)


if __name__ == '__main__':
    N_AGENTSS = [10]
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
    N_WORKERS = 2
    CURRENT_EPISODE = -1
    TRAIN_BATCH_SIZE = 1000  # 4000 by default for RLlib PPO

    # RATIO = int(4000/TRAIN_BATCH_SIZE)
    # TRAINING_ITERATIONS = 100 * RATIO # 1 iterations <=> 10000 timesteps TD3, 4000 timesteps for PPO, A2C !

    print('TRAINING_ITERATIONS:', TRAINING_ITERATIONS)
    N_RESETS = 0

    TRAIN_ONLY = False
    EVALUATION = True
    UNIT_TEST = False
    GAME_THEORY = False
    SEP = ":"
    EVAL_CSV_FILE_RESULTS = "/home/antoine/Documents/PycharmProjects/offline-watt-saving" \
                            "/metric_results_onguard_service.csv "
    debug = False
    n_agents = 10
    nb_test = 10
    max_rounds = 20
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

    # 3 agents
    agent0 = Agent(3, 0, tokenizer)
    agent1 = Agent(3, 1, tokenizer)
    agent2 = Agent(3, 2, tokenizer)
    agents = [agent0, agent1, agent2]

    # 4 agents

    agent0 = Agent(4, 0, tokenizer)
    agent1 = Agent(4, 1, tokenizer)
    agent2 = Agent(4, 2, tokenizer)
    agent3 = Agent(4, 3, tokenizer)
    agents = [agent0, agent1, agent2, agent3]

    # 8 agents
    agent0 = Agent(8, 0, tokenizer)
    agent1 = Agent(8, 1, tokenizer)
    agent2 = Agent(8, 2, tokenizer)
    agent3 = Agent(8, 3, tokenizer)
    agent4 = Agent(8, 4, tokenizer)
    agent5 = Agent(8, 5, tokenizer)
    agent6 = Agent(8, 6, tokenizer)
    agent7 = Agent(8, 7, tokenizer)
    agents = [agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7]

    # 10 agents
    agent0 = Agent(10, 0, tokenizer)
    agent1 = Agent(10, 1, tokenizer)
    agent2 = Agent(10, 2, tokenizer)
    agent3 = Agent(10, 3, tokenizer)
    agent4 = Agent(10, 4, tokenizer)
    agent5 = Agent(10, 5, tokenizer)
    agent6 = Agent(10, 6, tokenizer)
    agent7 = Agent(10, 7, tokenizer)
    agent8 = Agent(10, 8, tokenizer)
    agent9 = Agent(10, 9, tokenizer)
    agents = [agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7, agent8, agent9]

    """model0 = AutoModelForCausalLM.from_pretrained("./agent-3-0")
    model1 = AutoModelForCausalLM.from_pretrained("./agent-3-1")
    model2 = AutoModelForCausalLM.from_pretrained("./agent-3-2")"""
    env_config = {
        'n_agents': n_agents,
        'nmax_rounds': max_rounds,
        'debug': False,
        'mode': "imposed",
        'rew_timestep': REWARDS['rew_timestep'][0],
        'rew_offer_received': REWARDS['rew_offer_received'][0],
        'rew_guard_given': REWARDS['rew_guard_given'][0],
        'rew_active_remains': REWARDS['rew_active_remains'][0],
        'additional_rules': ADDITIONAL_RULES[0],
    }
    env = onguard_service(env_config=env_config)
    env_name = "onguard_service"
    register_env(env_name, env_creator)

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

                    print(f'run: {run+1}/{N_RUNS}')

                    for scenario in scenarios:

                        print('##### SCENARIO %s ######' % scenario)

                        env = onguard_service(env_config=env_config)
                        env_name = "onguard_service"
                        register_env(env_name, env_creator)

                        for t in tqdm(range(N_TESTS_PER_RUN)):
                            if debug:
                                print('==== Test %d ====' % t)

                            obs = env.reset()

                            if debug:
                                print('obs:', obs)

                            last_infos = {}

                            for s in range(NMAX_ROUNDS):
                                if debug:
                                    print('--- step %d ---' % s)

                                if scenario == 'CCC' or scenario == 'CDD':
                                    a0 = agent0.predict(obs)
                                elif scenario == 'DCC':
                                    a0 = np.array([0] * 1 + [1] * (n_agents - 1) + [0] * (n_agents - 1))
                                elif scenario == 'DDD':
                                    a0 = np.array([0] * 1 + [1] * (n_agents - 1) + [0] * (n_agents - 1))
                                else:
                                    print('ERROR: scenario %s not implemented' % scenario)
                                    break
                                actions = {0: a0}
                                cpt = 1
                                for agent in agents[1:]:
                                    if scenario == 'CCC' or scenario == 'DCC':
                                        ax = agent.predict(obs)
                                    elif scenario == 'CDD':
                                        ax = np.array([0] * 1 + [1] * (n_agents - 1) + [0] * (n_agents - 1))
                                    elif scenario == 'DDD':
                                        ax = np.array([0] * 1 + [1] * (n_agents - 1) + [0] * (n_agents - 1))
                                    else:
                                        print('ERROR: scenario %s not implemented' % scenario)
                                        break
                                    actions[cpt] = ax
                                    cpt += 1

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

                    # sleep_time = 60*5
                    # print('sleep for % sec' % sleep_time)
                    # time.sleep(sleep_time) # sleep to free GPU resources

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

    t3 = datetime.datetime.now()
    print('datetime:', t3)
    print('duration:', t3 - t0)
