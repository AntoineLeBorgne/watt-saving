import random
import sys
import numpy as np
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv

N_RESETS = 0
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
CURRENT_ENV_INSTANCE = 0
N_DONE = 0


def compute_jain_index(resources):
    denom = 0
    nom = 0
    for r in resources:
        nom += r
        denom += r * r
    nom = nom * nom
    denom = len(resources) * denom
    if denom == 0:
        return 0.0
    return nom / denom


class onguard_service(MultiAgentEnv):

    def __init__(self, env_config, return_agent_actions=False, part=False):

        # print('__init__ env_config:', env_config)

        global CURRENT_ENV_INSTANCE
        global TRAINING_ITERATIONS
        global CHEATER_AGENT
        global CHEATER_COUNTER_MEASURE
        global AGENT_TURN_RECOMMENDED
        global AGENT_TURN_IMPOSED
        global COOP_BLACKLIST
        global CIRCUIT_BREAKER
        global DLT_CHECK
        global ENERGY_DISPERSION

        self.rew_timestep = env_config['rew_timestep']
        self.rew_offer_received = env_config['rew_offer_received']
        self.rew_guard_given = env_config['rew_guard_given']
        self.rew_active_remains = env_config['rew_active_remains']

        self.debug = env_config['debug']
        self.nmax_rounds = env_config['nmax_rounds']

        self.additional_rules = env_config['additional_rules']

        self.mode = env_config['mode']
        if self.mode == 'free':
            AGENT_TURN_RECOMMENDED = False
            AGENT_TURN_IMPOSED = False
        elif self.mode == 'recommended':
            AGENT_TURN_RECOMMENDED = True
            AGENT_TURN_IMPOSED = False
        elif self.mode == 'imposed':
            AGENT_TURN_RECOMMENDED = True
            AGENT_TURN_IMPOSED = True
        else:
            print('unknown rule:', self.mode)
            self.mode = 'unknown'

        self.n_low_activity_periods = -1
        self.n_steps = -1
        self.n_steps_episode = -1
        self.n_rounds = -1
        self.n_agents = int(env_config['n_agents'])

        # attribute to check if an agent has left the  game or not
        self.agents_done = [False] * self.n_agents
        self.agents_done_next_turn = [False] * self.n_agents

        # attribute to store the return of the first episode (sw)
        self.agents_return = [0.0] * self.n_agents

        self.efficiency = 0.0
        self.safety = 0.0
        self.safety_norm = 0.0
        self.incentive_compatibility = 0.0
        self.incentive_compatibility_norm = 0.0

        self.sw = 0.0

        if CHEATER_AGENT:
            self.cheater_agent = self.n_agents - 1
        else:
            self.cheater_agent = - 1

        self.offers_performed = np.zeros([self.n_agents, self.n_agents])

        # set counters for low-activity periods
        self.periods = [0] * int(env_config['n_agents'])
        self.periods_off = [0] * int(env_config['n_agents'])
        self.periods_on = [0] * int(env_config['n_agents'])

        self.offers_tot_ok = [0] * int(env_config['n_agents'])
        self.demands_tot_ok = [0] * int(env_config['n_agents'])
        self.offers_per_epi_ok = 0
        self.demands_per_epi_ok = 0
        self.at_least_1_offer_ok = [False] * int(env_config['n_agents'])
        self.at_least_1_demand_ok = [False] * int(env_config['n_agents'])
        self.ratios_offer_demand = [0.0] * int(self.n_agents)

        if AGENT_TURN_IMPOSED and COOP_BLACKLIST:
            self.cooperating_agents = np.array([False] * int(self.n_agents))
            self.blacklisted_agents = np.array([0] * int(self.n_agents))
            self.blacklisted_day_agents = np.array([0] * int(self.n_agents))

        # set arrays for storing the kwh estimation
        self.kwh_base = np.full((self.n_agents), 1.0)
        self.kwh_mean = self.kwh_base.copy()

        self.onguard_periods = np.zeros([self.n_agents])
        self.kwh_tot_burnt = np.zeros([self.n_agents])
        self.kwh_tot_saved = np.zeros([self.n_agents])
        self.kwh_tot_s_b_o = np.zeros([self.n_agents])

        self.config = env_config

        # the N-1 next boolean host the (on/off) demands from B agents to A
        # the N-1 next boolean host the (on/off) offfers from B agent to Achea
        # the next boolean indicates if the agent should be on-guard
        # the last boolean indicates if the agent is blacklisted
        size_obs = 2 * (self.n_agents - 1) + 2 * self.n_agents
        self.observation_space = gym.spaces.MultiDiscrete([2] * size_obs)  # for each station, mno-id + state
        # self.observation_space = gym.spaces.MultiBinary(size_obs) # for each station, mno-id + state

        # action space of an agent:
        # * 1 boolean to power-off the agent's network
        # * N-1 boolean to demands other agents' network,
        # * N-1 boolean to offer its network to other agents,
        sub_actions = 1 + 2 * (self.n_agents - 1)
        self.action_space = gym.spaces.MultiDiscrete([2] * sub_actions)
        # self.action_space = gym.spaces.MultiBinary(sub_actions)

        # state is made of two sub-arrays
        # * 1 sub-array for storing the resources
        # * 1 sub-array for storing the current demands and offers
        self.state = self.get_init_state()
        self.init_state = self.state.copy()
        self.previous_state = self.state.copy()

        self.agent_turn = self.select_agent_turn()
        if self.debug:
            print('state', self.state.tolist())

    def select_agent_turn(self):
        if AGENT_TURN_IMPOSED and COOP_BLACKLIST and self.n_low_activity_periods > 0:
            min_value = sys.maxsize
            argmin = -1
            for a in range(self.n_agents):
                if self.blacklisted_agents[a] == 1:
                    if self.debug:
                        print('agent %d is blacklisted' % a)
                    # skip the blacklisted agent
                    if self.blacklisted_agents.sum() == (self.n_agents - 1):
                        if self.debug:
                            print('select_agent_turn(): only one agent remains !!!')
                    elif self.blacklisted_agents.sum() == self.n_agents:
                        if self.debug:
                            print('select_agent_turn(): zero agent remains !!!')
                else:
                    if self.kwh_tot_s_b_o[a] < min_value:
                        min_value = self.kwh_tot_s_b_o[a]
                        argmin = a
            if argmin == -1:
                if self.debug:
                    print('select_agent_turn(d%d): argmin=-1 should rarely happen' % self.n_low_activity_periods)
                agent_turn = self.kwh_tot_s_b_o.argmin()
            elif min_value == sys.maxsize:
                print(
                    'select_agent_turn(d%d): min_value = sys.maxsize MUST not happen!!!' % self.n_low_activity_periods)
            else:
                agent_turn = argmin
        else:
            agent_turn = self.kwh_tot_s_b_o.argmin()
        if self.debug:
            print('======================================')
            print('get_agent_turn() - n_low_activity_periods:%d' % self.n_low_activity_periods)
            print('get_agent_turn() - kwh_tot_s_b_o:', self.kwh_tot_s_b_o)
            print('get_agent_turn() - agent_turn:%d' % agent_turn)
        return agent_turn

    def get_init_state(self):

        # the first row hosts the state of each MNO's set of ressources
        # the N next rows host the (on/off) demands from A agent to B agents
        # the N next rows host the (on/off) offfers from A agent to B agents
        # 1 boolean flag indicating in the agent will be ITS turn
        state = np.zeros((1 + 2 * self.n_agents, self.n_agents))

        for a in range(self.n_agents):
            state[0, a] = 1
            for b in range(self.n_agents):
                state[1 + a, b] = 0
                state[1 + self.n_agents + a, b] = 0
        return state

    def get_observations(self, from_reset=False):

        obs = {}

        for a in range(self.n_agents):

            # init the observation tensor
            obs[a] = np.zeros((2 * (self.n_agents - 1) + 2 * self.n_agents, 1), dtype=int)

            # copy the state of the agent's resource
            # obs[a][0] = self.state[0, a]

            # copy the demands performed by other agents to agent-a
            d = 0
            for b in range(self.n_agents):
                if b != a:
                    obs[a][d] = self.state[1 + b, a]
                    d += 1

            # copy the offers performed by other agents to agent-a
            o = 0
            for b in range(self.n_agents):
                if b != a:
                    obs[a][self.n_agents - 1 + o] = self.state[1 + self.n_agents + b, a]
                    o += 1

            # indicate the agent that should be on-guard
            if AGENT_TURN_RECOMMENDED:
                for b in range(self.n_agents):
                    if b == self.agent_turn:
                        obs[a][(self.n_agents - 1) * 2 + b] = 1

            # indicate the the agents being blacklisted
            if AGENT_TURN_IMPOSED and COOP_BLACKLIST and self.n_low_activity_periods > 0:
                # print("ici")
                for b in range(self.n_agents):
                    if self.blacklisted_agents[b] == 1:
                        obs[a][(self.n_agents - 1) * 2 + self.n_agents + b] = 1
                        # print("la")

            obs[a] = obs[a].reshape((-1))

        # if self.debug:
        #    print('get_observations() - obs:', obs)
        return obs

    def get_kwh_really_burnt_from_dlt(self, a):

        kwh = self.kwh_base[a]
        kwh = kwh * (1 + random.uniform(ENERGY_DISPERSION * -1.0, ENERGY_DISPERSION))

        if self.debug:
            print('get_kwh_really_burnt_from_dlt() a%d burnt %f kwh' % (a, kwh))

        return kwh

    def calcultate_ratios(self):

        for a in range(self.n_agents):
            if self.demands_tot_ok[a] < 1 and self.offers_tot_ok[a] < 1:
                self.ratios_offer_demand[a] = 1.0
            elif self.demands_tot_ok[a] < 1 and self.offers_tot_ok[a] >= 1:
                self.ratios_offer_demand[a] = float(self.offers_tot_ok[a]) * 2.0
            else:
                self.ratios_offer_demand[a] = float(int(self.offers_tot_ok[a] * 10000 / self.demands_tot_ok[a]) / 100)

    def reset(self):

        if DLT_CHECK:
            n = self.n_low_activity_periods
            if n >= 0:
                kwh_mean = self.kwh_mean[self.agent_turn]
                kwh_new = self.get_kwh_really_burnt_from_dlt(self.agent_turn)
                kwh_tot_burnt = self.kwh_tot_burnt[self.agent_turn]
                # correct the total amount previously done
                if self.debug:
                    print('kwh_tot_burnt:', kwh_tot_burnt)
                    print('kwh_mean:', kwh_mean)
                    print('kwh_new:', kwh_new)
                self.kwh_tot_burnt[self.agent_turn] = kwh_tot_burnt - kwh_mean + kwh_new
                t = self.onguard_periods[self.agent_turn]
                if self.debug:
                    print('reset(): self.kwh_base__:', self.kwh_base)
                    print('reset(): self.kwh_tot_burnt__:', self.kwh_tot_burnt)
                    print('reset(): self.kwh_mean__:', self.kwh_mean)
                    print('reset(): t:', t)
                if t != 0:
                    self.kwh_mean[self.agent_turn] = (kwh_mean * (t - 1) + kwh_new) / t
                if self.debug:
                    print('reset(): self.kwh_mean_:', self.kwh_mean)
        if self.debug:
            print('reset(): self.kwh_base:', self.kwh_base)
            print('reset(): self.kwh_tot_burnt:', self.kwh_tot_burnt)
            print('reset(): self.kwh_mean:', self.kwh_mean)

        # check if previous agent in charge of helping other performed its job
        if AGENT_TURN_IMPOSED and COOP_BLACKLIST:
            if not self.cooperating_agents[self.agent_turn]:
                if self.blacklisted_agents[self.agent_turn] == 0:
                    if self.n_low_activity_periods > -1:
                        if self.debug:
                            print('reset a%d : blacklisted at reset %d' \
                                  % (self.agent_turn, self.n_low_activity_periods))
                        self.blacklisted_agents[self.agent_turn] = 1
                        self.blacklisted_day_agents[self.agent_turn] = self.n_low_activity_periods
            else:
                # agent has cooperated during the previous round, un-blacklist it
                self.blacklisted_agents[self.agent_turn] = 0

        # attribute to check if an agent has sent an action or not
        # self.agents_winner = [False] * int(self.n_agents)
        self.calcultate_ratios()

        # if self.debug and self.n_low_activity_periods % 1 == 0:
        if self.n_steps == 0:
            print('reset(%d): before now, was n_steps:%d, n_low_activity_periods=%d, n_rounds:%d' %
                  (N_RESETS, self.n_steps, self.n_low_activity_periods, self.n_rounds))
            print('reset() - self.agents_return:', self.agents_return)
            print('reset() - self.sw:', self.sw)
            print('reset() - self.efficiency:', self.efficiency)
            print('reset() - self.periods:', self.periods)
            print('reset() - self.periods_off:', self.periods_off)
            print('reset() - self.periods_on:', self.periods_on)
            print('reset() - self.demands_tot_ok:', self.demands_tot_ok)
            print('reset() - self.offers_tot_ok:', self.offers_tot_ok)
            print('reset() - self.ratios_offer_demand:', self.ratios_offer_demand)

        # trick to display the debug messages of a single episode
        episode_to_watch = -1500
        if self.n_low_activity_periods == episode_to_watch:
            print('self.debug=True')
            self.debug = True
        elif self.n_low_activity_periods == episode_to_watch + 1:
            print('self.debug=False')
            self.debug = False

        # day represent the number of resets
        self.n_low_activity_periods += 1

        # attribute to check if an agent has left the  game or not
        self.agents_done = [False] * self.n_agents
        self.agents_done_next_turn = [False] * self.n_agents

        self.agents_return = [0.0] * self.n_agents

        self.n_rounds = 0

        self.n_steps_episode = 0

        self.state = self.get_init_state()
        self.init_state = self.state.copy()
        self.previous_state = self.state.copy()

        self.at_least_1_offer_ok = [False] * self.n_agents
        self.at_least_1_demand_ok = [False] * self.n_agents

        self.offers_per_epi_ok = 0
        self.demands_per_epi_ok = 0
        self.n_offerers = 0

        if AGENT_TURN_IMPOSED and COOP_BLACKLIST:
            for a in range(self.n_agents):
                if self.blacklisted_agents[a]:
                    day_bl = self.n_low_activity_periods - self.blacklisted_day_agents[a]
                    if day_bl <= self.n_agents:  # blacklisted for n_agents periods
                        if self.debug:
                            print("reset() - a%d is blacklisted at day:%d" % (a, day_bl))
                    else:
                        if self.debug:
                            print("reset() - a%d is whitelisted at day:%d" % (a, self.n_low_activity_periods))
                        self.blacklisted_agents[a] = False

        self.agent_turn = self.select_agent_turn()

        obs = self.get_observations(from_reset=True)

        if self.debug:
            print('reset(): initial self.state:\n', self.state)
            print('reset(): initial obs:\n', obs)

        return obs

    def step(self, action_dict):

        global N_MNOs

        self.n_steps += 1
        self.n_steps_episode += 1
        self.n_rounds += 1

        if self.debug:
            print('step() - ts:%d, ts_episode:%d' % (self.n_steps, self.n_steps_episode))
            print('step() - action_dict:', action_dict)

        obs, rew, done, info = {}, {}, {}, {}
        done["__all__"] = False

        for a in range(self.n_agents):
            if self.agents_done_next_turn[a]:
                self.agents_done[a] = True
                self.agents_done_next_turn[a] = False

            if not self.agents_done[a]:
                rew[a] = self.rew_timestep

        list_of_agents = list(range(self.n_agents))
        random.shuffle(list_of_agents)
        for a in list_of_agents:
            # for a in range(self.n_agents):

            if self.agents_done[a]:
                if self.debug:
                    print('step() - a%d finished, do not care about its actions' % a)
                rew[a] = 0.0
                done[a] = False
                continue

            agent_actions = action_dict[a]
            if self.debug:
                print('step() - a%d agent_actions:%s' % (a, str(agent_actions)))

            # PROCESS DEMANDS
            # ===============
            dem_actions = agent_actions[1:1 + self.n_agents - 1]
            if self.debug:
                print('step() - a%d: dem_actions:%s' % (a, str(dem_actions)))

            i = 0
            for b in range(self.n_agents):
                if b != a:
                    if self.agents_done[b]:
                        if self.debug:
                            print('step() - b%d finished, do not care about its demand actions' % b)
                        continue

                    if AGENT_TURN_IMPOSED and COOP_BLACKLIST:
                        if self.blacklisted_agents[a] == 1:
                            if self.debug:
                                print('step() - a%d is blacklisted, do not care about its actions' % a)
                            continue

                    dem = dem_actions[i]
                    if dem == 1:
                        # a cheater is an agent that makes many more demands than offers
                        # (but an agent that makes many more offers than demands in not a cheater)
                        if CHEATER_COUNTER_MEASURE and self.n_steps > 8000 \
                                and self.ratios_offer_demand[a] < 0.75:
                            if self.debug:
                                print('step() - ratio:%f => a%d is a cheater; intercept its demand' % \
                                      (self.ratios_offer_demand[a], a))
                            self.state[1 + a, b] = 0
                        else:
                            if self.debug:
                                print('step() - a%d made demand to a%d: OK' % (a, b))
                            self.state[1 + a, b] = 1
                            # print('-', end='')
                    elif dem == 0:
                        if self.debug:
                            print('step() - no demand')
                        self.state[1 + a, b] = 0
                    else:
                        print('step() - ERR unexpected dem=', dem)
                    i += 1

            # PROCESS OFFERS
            # ==============
            off_actions = agent_actions[self.n_agents:]
            if self.debug:
                print('step() - a%d: off_actions:%s' % (a, str(off_actions)))

            # limit the number of offerers, if needed
            offer_refused = False
            if self.additional_rules:
                i = 0
                n_max_offerers = 1
                for b in range(self.n_agents):
                    if b != a and self.additional_rules:
                        off = off_actions[i]
                        if off == 1 and a != self.cheater_agent:
                            if self.at_least_1_offer_ok[a]:
                                # refuse a's offer if it already got a demand accepted
                                offer_refused = True
                            elif (self.get_n_offerers() + 1) > n_max_offerers:
                                # refuse a's offer if there is already the max number of offerers
                                offer_refused = True

            i = 0
            for b in range(self.n_agents):
                if b != a:

                    # if self.at_least_1_demand_ok[a]:
                    #    if self.debug:
                    #        print('step() - a%d is not allowed to offer, because it had a demand already accepted' % a)
                    #    #rew[a] += -0.1
                    #    continue

                    if self.agents_done[b]:
                        if self.debug:
                            print('step() - b%d finished, do not care about its offer actions' % b)
                        continue

                    if AGENT_TURN_IMPOSED and COOP_BLACKLIST:
                        if self.blacklisted_agents[a] == 1:
                            if self.debug:
                                print('step() - a%d is blacklisted, do not care about its actions' % a)
                            continue

                    off = off_actions[i]

                    # if we want to simulate a cheater (i.e. an agent that always demands
                    # and never offers its networks to other, then uncomment following 2 lines
                    # cheater = self.n_agents-1
                    if off == 1 and a != self.cheater_agent and not offer_refused:

                        if AGENT_TURN_IMPOSED and a != self.agent_turn:
                            if self.debug:
                                print("step() - it's not the turn of a%d to offer to a%d" % (a, b))

                        # check that b made a demand to a during the previous step
                        elif self.previous_state[1 + b, a] == 1:
                            if self.debug:
                                print('step() - a%d made offer to a%d: OK' % (a, b))
                            self.state[1 + self.n_agents + a, b] = 1

                            if AGENT_TURN_IMPOSED and COOP_BLACKLIST:
                                self.cooperating_agents[a] = True

                                # print('+', end='')
                            self.onguard_periods[a] += 1
                            self.kwh_tot_burnt[a] += self.kwh_mean[a]
                            self.kwh_tot_s_b_o[a] += self.kwh_mean[b]

                            self.kwh_tot_saved[b] += self.kwh_mean[b]

                            rew[b] += self.rew_offer_received  # WON !!!

                            # if it is the first offer that agent-a offers, then penalize it
                            if not self.at_least_1_offer_ok[a]:
                                rew[a] += self.rew_guard_given

                            self.state[0, b] = 0
                            self.agents_done_next_turn[b] = True
                            self.offers_tot_ok[a] += 1
                            self.demands_tot_ok[b] += 1
                            self.at_least_1_offer_ok[a] = True
                            self.at_least_1_demand_ok[b] = True
                            self.offers_per_epi_ok += 1
                            self.demands_per_epi_ok += 1

                            if self.debug:
                                print('step() - a%d can benefit from a%d offer' % (b, a))
                                print('step() - self.onguard_periods:', self.onguard_periods)
                                print('step() - self.kwh_tot_burnt:', self.kwh_tot_burnt)
                                print('step() - self.kwh_tot_saved:', self.kwh_tot_saved)
                                print('step() - self.kwh_tot_s_b_o:', self.kwh_tot_s_b_o)
                                print('step() - rew[%d]:%f' % (b, rew[b]))
                                print('step() - rew[%d]:%f' % (a, rew[a]))

                        else:
                            if self.debug:
                                print('step() - a%d did not previsouly make a demand to a%d' % (b, a))
                    elif off == 0:
                        if self.debug:
                            print('step() - no offer')
                        self.state[1 + self.n_agents + a, b] = 0
                    # else:
                    #    print('step() -ERR unexpected off=', off)
                    i += 1
        obs = self.get_observations()

        n_agents_done = 0
        for a in range(self.n_agents):
            if self.state[0, a] == 0:
                n_agents_done += 1
                done[a] = False
            elif CIRCUIT_BREAKER and self.agents_return[a] < -10.0:
                n_agents_done += 1
                done[a] = False

        bl_agents = 0
        if AGENT_TURN_IMPOSED and COOP_BLACKLIST:
            bl_agents = self.blacklisted_agents.sum()

        active_agents = self.n_agents - bl_agents - n_agents_done

        if self.debug:
            print('self.n_agents:', self.n_agents)
            print('bl_agents:', bl_agents)
            print('n_agents_done:', n_agents_done)
            print('active_agents:', active_agents)

        if active_agents <= 1 or self.n_rounds == self.nmax_rounds:
            done["__all__"] = True
            for a in range(self.n_agents):
                if not self.at_least_1_demand_ok[a] and not self.at_least_1_offer_ok[a]:
                    rew[a] += self.rew_active_remains
                done[a] = True
                self.periods[a] += 1
                if self.state[0, a] == 0:
                    self.periods_off[a] += 1
                else:
                    self.periods_on[a] += 1

        if self.n_rounds == self.nmax_rounds:
            if self.debug:
                print('step() - self.n_rounds:%d MAX reached !!!!!!!!!!!!!!!!!!!!!' % self.n_rounds)

        # save return in order to implement a circuit-breaker
        for a in range(self.n_agents):
            self.agents_return[a] += rew[a]

        # update the efficiency, safety and incentive compatibility metrics
        self.compute_social_metrics()

        jain_index = self.get_jain_index()
        for a in range(self.n_agents):
            info[a] = str(jain_index)

        self.previous_state = self.state.copy()

        if self.debug:
            print('step() - self.state:', self.state)
            print('step() - obs:', obs)
            print('step() - rew:', rew)
            print('step() - done:', done)
            print('step() - self.periods:', self.periods)
            print('step() - self.periods_off:', self.periods_off)
            print('step() - self.periods_on:', self.periods_on)
            print('step() - self.demands_tot_ok:', self.demands_tot_ok)
            print('step() - self.offers_tot_ok:', self.offers_tot_ok)

        # print(obs)
        # print(self.observation_space)
        return obs, rew, done, info

    def get_n_offerers(self):
        n_offerers = 0
        for a in range(self.n_agents):
            if self.at_least_1_offer_ok[a]:
                n_offerers += 1
        return n_offerers

    def compute_social_metrics(self):

        self.sw = 0.0
        for a in range(self.n_agents):
            if self.at_least_1_demand_ok[a] == 1 and self.at_least_1_offer_ok[a] == 0:
                self.sw += 1.0
            else:
                self.sw += 0.0
        # sw_0 += 0.0 * self.n_agents
        sw_opt = 1.0 * (self.n_agents - 1)

        self.efficiency = self.sw / sw_opt

        u_ccc = (self.n_agents - 1) / self.n_agents
        u_ddd = 0.0
        if self.mode == 'imposed':
            u_cdd = (self.n_agents - 1) / (self.n_agents) * \
                    (self.n_low_activity_periods / (self.n_low_activity_periods + 1))
        else:  # self.mode == 'free' or self.mode == 'recommended':
            u_cdd = 0.0
        self.safety = u_cdd - u_ddd
        self.safety_norm = self.safety / (u_ccc - u_ddd) + 1

        u_ccc = (self.n_agents - 1) / (self.n_agents)
        u_dcc = 0.0
        u_ddd = 0.0
        self.incentive_compatibility = u_ccc - u_dcc
        self.incentive_compatibility_norm = self.incentive_compatibility / (u_ccc - u_ddd)

    def get_n_agents(self):
        return self.n_agents

    def get_n_low_activity_periods(self):
        return self.n_low_activity_periods

    def get_demands_per_epi_ok(self):
        return self.demands_per_epi_ok

    def get_offers_per_epi_ok(self):
        return self.offers_per_epi_ok

    def get_offerers_per_epi_ok(self):
        sum = 0
        for a in range(self.n_agents):
            if self.at_least_1_offer_ok[a]:
                sum += 1
        return sum

    def get_pctg_guard_ok(self):

        n_offerers = 0
        n_demanders = 0
        offerer_id = -1

        for a in range(self.n_agents):
            if self.at_least_1_offer_ok[a]:
                n_offerers += 1
                offerer_id = a

        for a in range(self.n_agents):
            if a == offerer_id:
                continue
            if self.at_least_1_demand_ok[a]:
                n_demanders += 1

        if n_offerers == 1 and n_demanders == (self.n_agents - 1):
            return 1
        else:
            return 0

    def get_demanders_per_epi_ok(self):
        sum = 0
        for a in range(self.n_agents):
            if self.at_least_1_demand_ok[a]:
                sum += 1
        return sum

    def get_social_welfare(self):
        return self.sw

    def get_efficiency(self):
        return self.efficiency

    def get_safety(self):
        return self.safety

    def get_safety_norm(self):
        return self.safety_norm

    def get_incentive_compatibility(self):
        return self.incentive_compatibility

    def get_incentive_compatibility_norm(self):
        return self.incentive_compatibility_norm

    def get_return(self, a):
        return self.agents_return[a]

    def get_onguard_periods(self, a):
        return self.onguard_periods[a]

    def get_kwh_base(self, a):
        return self.kwh_base[a]

    def get_kwh_tot_burnt(self, a):
        return self.kwh_tot_burnt[a]

    def get_kwh_tot_saved(self, a):
        return self.kwh_tot_saved[a]

    def get_pct_kwh_tot_saved(self, a):
        if self.n_low_activity_periods == -1 or self.kwh_base[a] == 0.0:
            return 1
        else:
            return self.kwh_tot_saved[a] / ((self.n_low_activity_periods + 1) * self.kwh_base[a])

    def get_kwh_tot_s_b_o(self, a):
        return self.kwh_tot_s_b_o[a]

    def get_offers_tot_ok(self, a):
        return self.offers_tot_ok[a]

    def get_demands_tot_ok(self, a):
        return self.demands_tot_ok[a]

    def get_ratios_offer_demand(self, a):
        return self.ratios_offer_demand[a]

    def get_at_least_1_demand_ok(self, a):
        return self.at_least_1_demand_ok[a]

    def get_at_least_1_offer_ok(self, a):
        return self.at_least_1_offer_ok[a]

    def get_jain_index(self):
        kwh_tot_saveds = []
        for a in range(self.n_agents):
            if AGENT_TURN_IMPOSED and COOP_BLACKLIST and self.n_low_activity_periods > 0:
                # if not blacklisted, take it into account
                if self.blacklisted_agents[a] == 0:
                    kwh_tot_saveds.append(self.kwh_tot_saved[a])
                # else, exclude agent's resources from the jain calculus
            else:
                kwh_tot_saveds.append(self.kwh_tot_saved[a])
        jain_index = compute_jain_index(kwh_tot_saveds)

        return jain_index

    def render(self, mode=None):
        for a in range(self.n_agents):
            if self.state[0, a] == 1:
                print('a%d, power=ON |' % a, end="")
            elif self.state[0, a] == 0:
                print('a%d, power=OFF|' % a, end="")
            else:
                print('!!!!!!a:', a, 'state:', self.state[0, a])
            print("DEMANDS TO ", end="")
            d = 0
            for b in range(self.n_agents):
                if a == b:
                    continue
                else:
                    print('b%d:%d, ' % (b, self.state[1 + a, b]), end='')
                    d += 1
            print("OFFERS TO ", end="")
            o = 0
            for b in range(self.n_agents):
                if a == b:
                    continue
                else:
                    print('a%d:%d, ' % (b, self.state[1 + self.n_agents + a, b]), end='')
                    d += 1
            print('')
        # print("‾" * (self.width + 2))
        # print(f"{'!!Collision!!' if self.collision else ''}")
        # print("R1={: .1f}".format(self.agent1_R))
        # print("R2={: .1f}".format(self.agent2_R))
