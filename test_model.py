import gym as gym
import time
import re
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline, set_seed


def first_block(string):
    return re.split(SEP, string)[0].rstrip()


def complete_code(pipe, prompt, max_length=57, num_completions=1, seed=1):
    set_seed(seed)
    gen_kwargs = {"temperature": 0.1, "top_p": 0.95, "top_k": 0, "num_beams": 1, "do_sample": True}
    code_gens = generation(prompt, num_return_sequences=num_completions, max_length=max_length, **gen_kwargs,
                           pad_token_id=tokenizer.pad_token_id)
    code_strings = []
    for code_gen in code_gens:
        generated_code = first_block(code_gen['generated_text'][len(prompt):])
        code_strings.append(generated_code)
    return ''.join(code_strings)


def interprate(trajectorie, env):
    prompt = trajectorie + SEP
    prediction = complete_code(generation, prompt)
    prediction = prediction.split(';')
    print("pred:", prediction[0])
    predict_action = int(prediction[0].split(',')[1])
    print(f"predict_action: {predict_action}")
    state, reward, done, info = env.step(predict_action)
    env.render(mode="ansi")
    time.sleep(1)
    if state < 10:
        new_trajectorie = trajectorie + prediction[0] + ';0' + str(state)
    else:
        new_trajectorie = trajectorie + prediction[0] + ';' + str(state)
    print(prediction[0])
    predict_reward = int(prediction[0].split(',')[2])
    print("new", new_trajectorie)
    if not done:
        if predict_reward != 1:
            interprate(new_trajectorie, env)
    pass


tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
model = AutoModelForCausalLM.from_pretrained("agents/agent-8-3")

generation = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

SEP = ':'



words = ['1,0,0,0,0,0,0,0,0,0',
         '0,1,0,0,0,0,0,0,0,0',
         '0,0,1,0,0,0,0,0,0,0',
         '0,0,0,1,0,0,0,0,0,0',
         '0,0,0,0,1,0,0,0,0,0',
         '0,0,0,0,0,1,0,0,0,0',
         '0,0,0,0,0,0,1,0,0,0',
         '0,0,0,0,0,0,0,1,0,0',
         '0,0,0,0,0,0,0,0,1,0',
         '0,0,0,0,0,0,0,0,0,1'
         ]

words = ['1,0,0,0,',
         '0,1,0,0,',
         '0,0,1,0,',
         '0,0,0,1,']

words = ['1,0,0,0,0,0,0,0',
         '0,1,0,0,0,0,0,0',
         '0,0,1,0,0,0,0,0',
         '0,0,0,1,0,0,0,0',
         '0,0,0,0,1,0,0,0',
         '0,0,0,0,0,1,0,0',
         '0,0,0,0,0,0,1,0',
         '0,0,0,0,0,0,0,1']
for word in words:
    prompt = word + SEP
    predict = complete_code(generation, prompt)
    print('word:%s => predict:%s' % (word, predict))
