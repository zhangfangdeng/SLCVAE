#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University

import numpy as np


# Data feed
class LongDataLoader(object):
    """A special efficient data loader for TBPTT"""
    batch_size = 0
    backward_size = 0
    step_size = 0
    ptr = 0
    num_batch = None
    batch_indexes = None
    grid_indexes = None
    indexes = None
    data_lens = None
    data_size = None
    prev_alive_size = 0
    name = None

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self, cur_grid, prev_grid):
        raise NotImplementedError("Have to override prepare batch")

    def epoch_init(self, batch_size, backward_size, step_size, shuffle=True, intra_shuffle=False):

        self.ptr = 0
        self.batch_size = batch_size
        # self.backward_size = backward_size
        self.step_size = step_size
        self.prev_alive_size = batch_size

        # create batch indexes
        temp_num_batch = self.data_size // batch_size
        self.batch_data = {}
        self.batch_source = []
        self.batch_target = []

        for i in range(temp_num_batch):
            self.batch_source.append(self.source[i * self.batch_size:(i + 1) * self.batch_size])
        for i in range(temp_num_batch):
            self.batch_target.append(self.target[i * self.batch_size:(i + 1) * self.batch_size])

        left_over = self.data_size-temp_num_batch*batch_size

        # shuffle batch indexes
        # if shuffle:
        #     self._shuffle_batch_indexes()

        # create grid indexes
        self.grid_indexes = []

        for i in range(temp_num_batch):
            b_ids = self.batch_source[i]
            all_source_lens = [len(s) for s in b_ids]
            max_source_len = max(all_source_lens)
            min_source_len = min(all_source_lens)

            t_ids = self.batch_target[i]
            all_target_lens = [len(t) for t in t_ids ]

            max_target_len = max(all_target_lens)
            min_target_len = min(all_target_lens)

            new_grids = {"b_ids":b_ids, "all_source_lens":all_source_lens, "max_source_len":max_source_len,"min_source_len":min_source_len,
                         "t_ids": t_ids, "all_target_lens": all_target_lens, "max_target_len": max_target_len,
                         "min_target_len": min_target_len}
            self.grid_indexes.append(new_grids)


        self.num_batch = len(self.grid_indexes)
        print("%s begins with %d batches with %d left over samples" % (self.name, self.num_batch, left_over))

    def next_batch(self):
        if self.ptr < self.num_batch:
            current_grid = self.grid_indexes[self.ptr]
            if self.ptr > 0:
                prev_grid = self.grid_indexes[self.ptr-1]
            else:
                prev_grid = None
            self.ptr += 1
            return self._prepare_batch(cur_grid=current_grid, prev_grid=prev_grid)
        else:
            return None


class SWDADataLoader(LongDataLoader):
    def __init__(self, name, source, target, config):
        self.name = name
        self.source = source
        self.target = target
        self.data_size = len(target)
        self.source_lens = all_source_lens = [len(line) for line in self.source]
        self.target_lens = all_target_lens = [len(line) for line in self.target]
        self.max_utt_size = config.max_utt_len

    def pad_to(self, tokens, max_len):
        res = []
        for line in tokens:
            if len(line) < max_len:
                line = line + [0] * (max_len - len(line))
            res.append(line)
        return res

    def _prepare_batch(self, cur_grid, prev_grid):
        # the batch index, the starting point and end point for segment
        b_ids = cur_grid["b_ids"]
        all_source_lens = cur_grid["all_source_lens"]
        max_source_len = cur_grid["max_source_len"]
        min_source_len = cur_grid["min_source_len"]
        t_ids = cur_grid["t_ids"]
        all_target_lens = cur_grid["all_target_lens"]
        max_target_len = cur_grid["max_target_len"]
        min_target_len = cur_grid["min_target_len"]

        pad_source = self.pad_to(b_ids, max_source_len)

        pad_target = self.pad_to(t_ids, max_target_len)


        return pad_source, pad_target, all_source_lens, all_target_lens








