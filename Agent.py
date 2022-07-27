import time
import re
from transformers import AutoModelForCausalLM
from transformers import pipeline, set_seed


class Agent:
    def __init__(self, nb_agents, num_agent, tokenizer):
        self.nb_agents = nb_agents
        self.num_agent = num_agent
        self.tokenizer = tokenizer
        self.sep = ":"
        self.generation = pipeline('text-generation',
                                   model=AutoModelForCausalLM.from_pretrained(
                                       "./agents/agent-" + str(self.nb_agents) + "-" + str(self.num_agent)),
                                   tokenizer=self.tokenizer, device=0)

    def first_block(self, string):
        return re.split(self.sep, string)[0].rstrip()

    def complete_code(self, prompt, max_length=57, num_completions=1, seed=1):  # 3 agents :  = 15 , a agents : 21 ,
        # 8 agents : 45 , 10 agents : 58
        set_seed(seed)
        gen_kwargs = {"temperature": 0.1, "top_p": 0.95, "top_k": 0, "num_beams": 1, "do_sample": True}
        code_gens = self.generation(prompt, num_return_sequences=num_completions, max_length=max_length, **gen_kwargs,
                                    pad_token_id=self.tokenizer.pad_token_id)
        code_strings = []
        for code_gen in code_gens:
            generated_code = self.first_block(code_gen['generated_text'][len(prompt):])
            code_strings.append(generated_code)
        return ''.join(code_strings)

    def predict(self, observation):
        # changer le format pour utiliser le modele offline
        obs_agent = observation[self.num_agent]
        obs_agent = ','.join(str(obs_agent)[1:-1].replace(" ", ""))
        if self.nb_agents == 10:
            truc = obs_agent[2 * 2 * (self.nb_agents - 1):len(obs_agent) - 2 * self.nb_agents - 2]
        else :
            truc = obs_agent[2 * 2 * (self.nb_agents - 1):len(obs_agent) - 2 * self.nb_agents]
        prompt = truc + self.sep
        # determiner action en fonction de obs_agent
        pred = self.complete_code(prompt)[1:]
        # changer le format de l'action pour que l'environnement la comprenne
        action = list(map(int, pred.split(",")))
        return action
