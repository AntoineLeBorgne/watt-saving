import torch
from torch.utils.data import IterableDataset

FEATURE = 'sequence'


class ConstantLengthDataset(IterableDataset):

    def __init__(self, tokenizer, dataset, seq_length=1024,
                 num_of_sequences=100, chars_per_token=1):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences
        self.n_buf = 0

    def __iter__(self):
        iterator = iter(self.dataset[FEATURE])
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    m = f"Buffer full: {buffer_len}>={self.input_characters:.0f}"
                    self.n_buf += 1
                    # print('self.n_buf:', self.n_buf)
                    break
                try:
                    m = f"Fill buffer: {buffer_len}<{self.input_characters:.0f}"
                    # print(m)
                    entry = next(iterator)
                    if len(entry) > self.input_characters:
                        print('WARNING len(%s) > %.2f !' % (entry, self.input_characters))
                    # pad the entry to limit 1 entry per sample (i.e. to 64 chars)
                    entry += self.tokenizer.pad_token * (self.input_characters - len(entry))
                    # print('len(entry):', len(entry))
                    # print('self.input_characters:', self.input_characters)
                    # print('')
                    buffer.append(entry)
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    # print('Exception!!')
                    iterator = iter(self.dataset[FEATURE])

            all_token_ids = []
            tokenized_inputs = self.tokenizer(buffer, truncation=False)
            for tokenized_input in tokenized_inputs['input_ids']:
                all_token_ids.extend(tokenized_input)
            # print('all_token_ids:', all_token_ids)
            # print('self.seq_length:', self.seq_length)
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i: i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    # print('yield:')
                    yield torch.tensor(input_ids)
