import logging
import datetime
import os
import sys
import ConstantLenghtDataset
import datasets
import transformers
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, set_seed, AdamW, get_scheduler
from datasets import load_dataset
from tqdm import tqdm
from argparse import Namespace
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator  # device = 'gpu'


def batch_iterator(batch_size=500):
    for j in range(0, len(dataset['train']), batch_size):
        yield dataset['train'][j: j + batch_size][FEATURE]
        # print(i)


def setup_logging():
    if not os.path.exists("log"):
        os.makedirs("log")
    logger_setup = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
        handlers=[logging.FileHandler(f"log/debug_{accelerator.process_index}.log", mode='w'), logging.StreamHandler()])

    if accelerator.is_main_process:  # We only want to set up logging once
        tb_writer_setup = SummaryWriter()
        tb_writer_setup.add_hparams(vars(args), {'0': 0})
        logger_setup.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_debug()
        transformers.utils.logging.set_verbosity_info()
    else:
        tb_writer_setup = None
        # run_name = ''
        logger_setup.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger_setup, tb_writer_setup


def log_metrics(step, metrics):
    logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        # wandb.log(metrics)
        [tb_writer.add_scalar(k, v, step) for k, v in metrics.items()]


def create_dataloaders(num_agent, nb_agent, delta_retraining=0):
    dataset_name = "data/train-watt-saving-agent-" + str(nb_agent) + "-" + str(num_agent) + ".csv"
    train_data = load_dataset('csv', data_files=dataset_name)

    dataset_name = "data/test-watt-saving-agent-" + str(nb_agent) + "-" + str(num_agent) + ".csv"
    valid_data = load_dataset('csv', data_files=dataset_name)

    train_dataset = ConstantLenghtDataset.ConstantLengthDataset(tokenizer, train_data['train'][delta_retraining:],
                                                                seq_length=args.seq_length,
                                                                num_of_sequences=args.num_of_sequences,
                                                                chars_per_token=args.chars_per_token)
    valid_dataset = ConstantLenghtDataset.ConstantLengthDataset(tokenizer, valid_data,
                                                                seq_length=args.seq_length,
                                                                num_of_sequences=args.num_of_sequences,
                                                                chars_per_token=args.chars_per_token)

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)

    return train_dataloader, eval_dataloader


def get_grouped_params(model, no_decay=None):
    if no_decay is None:
        no_decay = ["bias", "LayerNorm.weight"]
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [{'params': params_with_wd, 'weight_decay': args.weight_decay},
            {'params': params_without_wd, 'weight_decay': 0.0}]


def evaluate(model, eval_dataloader):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        # print('step-e:', step)
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if 0 < args.max_eval_steps <= step:
            break
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = torch.tensor(float("inf"))
    return loss.item(), perplexity.item()


def get_lr():
    return optimizer.param_groups[0]['lr']


CONFIG = {"train_batch_size": 50,
          "valid_batch_size": 10,
          "weight_decay": 0.1,
          "shuffle_buffer": 1000,
          "learning_rate": 2e-3,  # 3 et 4 agents : 2e-2 , 8 et 10 agents : 2e-3
          "lr_scheduler_type": "cosine",
          "num_warmup_steps": 2000,
          "gradient_accumulation_steps": 1,
          "max_train_steps": 2000,
          "max_eval_steps": 1000,
          "seq_length": 60,  # 3 agents : 21 , 4 agents : 32 , 8 agents : 50 , 10 agents : 60
          "num_of_sequences": 1,
          "chars_per_token": 1,
          "seed": 1,
          "save_checkpoint_steps": 10000}  # 15000

args = Namespace(**CONFIG)

NOMBRE_AGENTS = 8
NOMBRE_AGENTS = sys.argv[1]
FEATURE = 'sequence'
TRAIN_TOKENIZER = False
CHARS = ['\n', '0', '1', ';', ',', '"']

if TRAIN_TOKENIZER:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # https://huggingface.co/docs/transformers/v4.17.0/en/internal/tokenization_utils
    tokenizer.bos_token = '>'
    tokenizer.eos_token = '<'
    tokenizer.pad_token = '#'

    if tokenizer.bos_token in CHARS:
        print('bos_token ERROR !!!')
    if tokenizer.eos_token in CHARS:
        print('eos_token ERROR !!!')
    if tokenizer.pad_token in CHARS:
        print('pad_token ERROR !!! ')
    print('my_chars:\n', len(CHARS), "\n")

    new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=len(CHARS),
                                                      initial_alphabet=CHARS)
    new_tokenizer.save_pretrained("./tokenizer")

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

print('info tokenizer:\n', tokenizer, "\n")

print("ensemble du vocabulaire :\n", tokenizer.get_vocab(), "\n")

# ----------------------------------------------------- MODELE ---------------------------------------------------------

t1 = datetime.datetime.now()

#for num_agent in range(NOMBRE_AGENTS):
for num_agent in sys.argv[2:]:
    # num_agent = 9
    # dataset = load_dataset("AntoineLB/watt-saving-agent-" + str(num_agent))
    test_path = "data/test-watt-saving-agent-" + str(NOMBRE_AGENTS) + "-" + str(num_agent) + ".csv"
    train_path = "data/train-watt-saving-agent-" + str(NOMBRE_AGENTS) + "-" + str(num_agent) + ".csv"
    dataset = load_dataset('csv', data_files={'train': train_path, 'test': test_path})

    # tokenizer : done above
    config_small = AutoConfig.from_pretrained("gpt2", vocab_size=len(tokenizer))
    model = AutoModelForCausalLM.from_config(config_small)

    shuffled_dataset = dataset['train'].shuffle()
    constant_length_dataset = ConstantLenghtDataset.ConstantLengthDataset(tokenizer, shuffled_dataset,
                                                                          seq_length=args.seq_length,
                                                                          num_of_sequences=args.num_of_sequences,
                                                                          chars_per_token=args.chars_per_token)
    dataset_iterator = iter(constant_length_dataset)

    lengths = [len(b) for _, b in zip(range(5), dataset_iterator)]
    print(f"Lengths of the sequences: {lengths}")

    set_seed(args.seed)

    # Accelerator
    accelerator = Accelerator()
    samples_per_step = accelerator.state.num_processes * args.train_batch_size

    # Logging
    logger, tb_writer = setup_logging()
    logger.info(accelerator.state)

    # Prepare the optimizer and learning rate scheduler
    optimizer = AdamW(get_grouped_params(model), lr=args.learning_rate)
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer,
                                 num_warmup_steps=args.num_warmup_steps,
                                 num_training_steps=args.max_train_steps, )

    # Load dataset and dataloader
    train_dataloader, eval_dataloader = create_dataloaders(num_agent, NOMBRE_AGENTS)

    # Prepare everything with our `accelerator` (order of args is not important)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader,
                                                                              eval_dataloader)

    # Train model
    model.train()

    completed_steps = 0
    cpt = 0
    for step, batch in enumerate(train_dataloader, start=1):
        if step % 1000 == 0:
            print('step-t:', step)
        loss = model(batch, labels=batch).loss
        log_metrics(step, {'lr': get_lr(), 'samples': step * samples_per_step, 'steps': completed_steps,
                           'loss/train': loss.item()})
        loss = loss / args.gradient_accumulation_steps
        accelerator.backward(loss)
        if step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
        if step % args.save_checkpoint_steps == 0:
            # on rentre pas dans la condition
            logger.info('Evaluating and saving model checkpoint')
            eval_loss, perplexity = evaluate(model, eval_dataloader)
            log_metrics(step, {'loss/eval': eval_loss, 'perplexity': perplexity})
            logger.info("test1")
            accelerator.wait_for_everyone()
            logger.info("test2")
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                unwrapped_model.save_pretrained("./")
                # hf_repo.push_to_hub(commit_message=f'step {step}')
            model.train()
        if loss.item() <= 0.05:
            cpt += 1
            if cpt >= 10:
                accelerator.wait_for_everyone()
                break
        if completed_steps >= args.max_train_steps:
            break

            # Evaluate and save the last checkpoint
    # logger.info('Evaluating and saving model after training')
    # eval_loss, perplexity = evaluate(model, eval_dataloader)
    # log_metrics(step, {'loss/eval': eval_loss, 'perplexity': perplexity})
    print('ok')
    accelerator.wait_for_everyone()
    print('ok')
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained("./agents/agent-" + str(NOMBRE_AGENTS) + "-" + str(num_agent))
        # hf_repo.push_to_hub(commit_message=f'final model')

t2 = datetime.datetime.now()
print(f"Training time : {t2 - t1}")
