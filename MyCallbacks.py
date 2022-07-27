from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode import MultiAgentEpisode


class MyCallbacks(DefaultCallbacks):

    def on_episode_end(self,
                       *,
                       worker,
                       base_env,
                       policies,
                       episode: MultiAgentEpisode,
                       env_index,
                       **kwargs):

        n_agents = base_env.get_unwrapped()[0].get_n_agents()

        for a in range(n_agents):

            label_return = 'return_' + str(a)
            label_periods = 'periods_' + str(a)
            label_periods_off = 'periods_off_' + str(a)
            label_periods_on = 'periods_on_' + str(a)
            label_otok = 'offers_tot_ok_' + str(a)
            label_dtok = 'demands_tot_ok_' + str(a)
            label_rod = 'ratios_offer_demand_' + str(a)
            label_onguard_periods = 'onguard_periods_' + str(a)
            label_kwh_tot_burnt = 'kwh_tot_burnt_' + str(a)
            label_kwh_tot_saved = 'kwh_tot_saved_' + str(a)
            label_kwh_tot_s_b_o = 'kwh_tot_s_b_o_' + str(a)
            label_pct_kwh_tot_saved = 'pct_kwh_tot_saved_' + str(a)
            label_at_least_1_offer_ok = 'at_least_1_offer_ok_' + str(a)
            label_at_least_1_demand_ok = 'at_least_1_demand_ok_' + str(a)

            try:
                agent_obs = episode.last_observation_for(a)
                episode.custom_metrics[label_periods] = 1
                # episode.hist_data[label_periods] = 1
                state = agent_obs[0]
                if state == 1:
                    episode.custom_metrics[label_periods_off] = 1
                    # episode.hist_data[label_periods_off] = 0
                    episode.custom_metrics[label_periods_on] = 0
                    # episode.hist_data[label_periods_on] = 1
                elif state == 0:
                    episode.custom_metrics[label_periods_off] = 0
                    # episode.hist_data[label_periods_off] = 1
                    episode.custom_metrics[label_periods_on] = 1
                    # episode.hist_data[label_periods_on] = 0
                else:
                    foo = 1
                episode.custom_metrics[label_return] = base_env.get_unwrapped()[0].get_return(a)
                episode.custom_metrics[label_otok] = base_env.get_unwrapped()[0].get_offers_tot_ok(a)
                episode.custom_metrics[label_dtok] = base_env.get_unwrapped()[0].get_demands_tot_ok(a)
                episode.custom_metrics[label_rod] = base_env.get_unwrapped()[0].get_ratios_offer_demand(a)
                episode.custom_metrics[label_onguard_periods] = base_env.get_unwrapped()[0].get_onguard_periods(a)
                episode.custom_metrics[label_kwh_tot_burnt] = base_env.get_unwrapped()[0].get_kwh_tot_burnt(a)
                episode.custom_metrics[label_kwh_tot_saved] = base_env.get_unwrapped()[0].get_kwh_tot_saved(a)
                episode.custom_metrics[label_kwh_tot_s_b_o] = base_env.get_unwrapped()[0].get_kwh_tot_s_b_o(a)
                episode.custom_metrics[label_pct_kwh_tot_saved] = base_env.get_unwrapped()[0].get_pct_kwh_tot_saved(a)
                episode.custom_metrics[label_at_least_1_offer_ok] = base_env.get_unwrapped()[0].get_at_least_1_offer_ok(
                    a)
                episode.custom_metrics[label_at_least_1_demand_ok] = base_env.get_unwrapped()[
                    0].get_at_least_1_demand_ok(a)

            except:
                foo = 2

        label_demands_per_epi_ok = 'demands_per_epi_ok'
        label_offers_per_epi_ok = 'offers_per_epi_ok'
        label_demanders_per_epi_ok = 'demanders_per_epi_ok'
        label_offerers_per_epi_ok = 'offerers_per_epi_ok'
        label_jain = 'jain_index'
        label_social_welfare = 'social_welfare'
        label_efficiency = 'efficiency'
        label_safety = 'safety'
        label_safety_norm = 'safety_norm'
        label_incentive_compatibility = 'incentive_compatibility'
        label_incentive_compatibility_norm = 'incentive_compatibility_norm'
        label_pctg_guard_ok = 'pctg_guard_ok'
        label_n_low_activity_periods = 'n_low_activity_periods'

        episode.custom_metrics[label_social_welfare] = base_env.get_unwrapped()[0].get_social_welfare()
        episode.custom_metrics[label_efficiency] = base_env.get_unwrapped()[0].get_efficiency()
        episode.custom_metrics[label_safety] = base_env.get_unwrapped()[0].get_safety()
        episode.custom_metrics[label_safety_norm] = base_env.get_unwrapped()[0].get_safety_norm()
        episode.custom_metrics[label_incentive_compatibility] = base_env.get_unwrapped()[
            0].get_incentive_compatibility()
        episode.custom_metrics[label_incentive_compatibility_norm] = base_env.get_unwrapped()[
            0].get_incentive_compatibility_norm()
        episode.custom_metrics[label_jain] = base_env.get_unwrapped()[0].get_jain_index()
        episode.custom_metrics[label_pctg_guard_ok] = base_env.get_unwrapped()[0].get_pctg_guard_ok()

        episode.custom_metrics[label_demands_per_epi_ok] = base_env.get_unwrapped()[0].get_demands_per_epi_ok()
        episode.custom_metrics[label_offers_per_epi_ok] = base_env.get_unwrapped()[0].get_offers_per_epi_ok()
        episode.custom_metrics[label_demanders_per_epi_ok] = base_env.get_unwrapped()[0].get_demanders_per_epi_ok()
        episode.custom_metrics[label_offerers_per_epi_ok] = base_env.get_unwrapped()[0].get_offerers_per_epi_ok()

        episode.custom_metrics[label_n_low_activity_periods] = base_env.get_unwrapped()[0].get_n_low_activity_periods()
