# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import datetime
from collections import OrderedDict
from typing import Dict, Optional, Union

import torch
import torch.distributed as dist

from .baselog import LoggerHook


class TextLoggerHook(LoggerHook):
    """Logger hook in text.

    In this logger hook, the information will be printed on terminal and
    saved in json file.

    Args:
        by_epoch (bool, optional): Whether EpochBasedRunner is used.
            Default: True.
        interval (int, optional): Logging interval (every k iterations).
            Default: 10.
        ignore_last (bool, optional): Ignore the log of last iterations in each
            epoch if less than :attr:`interval`. Default: True.
        reset_flag (bool, optional): Whether to clear the output buffer after
            logging. Default: False.
        interval_exp_name (int, optional): Logging interval for experiment
            name. This feature is to help users conveniently get the experiment
            information from screen or log file. Default: 1000.
        out_suffix (str or tuple[str], optional): Those filenames ending with
            ``out_suffix`` will be copied to ``out_dir``.
            Default: ('.log.json', '.log', '.py').
        keep_local (bool, optional): Whether to keep local log when
            :attr:`out_dir` is specified. If False, the local log will be
            removed. Default: True.
    """

    def __init__(
        self,
        by_epoch: bool = True,
        interval: int = 10,
        ignore_last: bool = True,
        reset_flag: bool = False,
        interval_exp_name: int = 1000,
        out_suffix: Union[str, tuple] = (".log.json", ".log", ".py"),
        keep_local: bool = True,
    ):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.by_epoch = by_epoch
        self.time_sec_tot = 0
        self.interval_exp_name = interval_exp_name
        self.out_suffix = out_suffix
        self.keep_local = keep_local

    def before_run(self, runner) -> None:
        super().before_run(runner)
        self.start_iter = runner.iter

    def _get_max_memory(self, runner) -> int:
        device = getattr(runner.model, "output_device", None)
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor(
            [int(mem) // (1024 * 1024)], dtype=torch.int, device=device
        )
        if runner.world_size > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

    def _log_info(self, log_dict: Dict, runner) -> None:
        # print exp name for users to distinguish experiments
        # at every ``interval_exp_name`` iterations and the end of each epoch
        if runner.meta is not None and "exp_name" in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                self.by_epoch and self.end_of_epoch(runner)
            ):
                exp_info = f'Exp name: {runner.meta["exp_name"]}'
                runner.logger.info(exp_info)
        if log_dict["mode"] == "train":
            if isinstance(log_dict["lr"], dict):
                lr_str = []
                for k, val in log_dict["lr"].items():
                    lr_str.append(f"lr_{k}: {val:.3e}")
                lr_str = " ".join(lr_str)  # type: ignore
            else:
                lr_str = f'lr: {log_dict["lr"]:.3e}'  # type: ignore

            # by epoch: Epoch [4][100/1000]
            # by iter:  Iter [100/100000]
            if self.by_epoch:
                log_str = (
                    f'Epoch [{log_dict["epoch"]}]'
                    f'[{log_dict["iter"]}/{len(runner.data_loader)}]\t'
                )
            else:
                log_str = f'Iter [{log_dict["iter"]}/{runner.max_iters}]\t'
            log_str += f"{lr_str}, "

            if "time" in log_dict.keys():
                self.time_sec_tot += log_dict["time"] * self.interval
                time_sec_avg = self.time_sec_tot / (runner.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f"eta: {eta_str}, "
                log_str += (
                    f'time: {log_dict["time"]:.3f}, '
                    f'data_time: {log_dict["data_time"]:.3f}, '
                )
                # statistic memory
                if torch.cuda.is_available():
                    log_str += f'memory: {log_dict["memory"]}, '
        else:
            # val/test time
            # here 1000 is the length of the val dataloader
            # by epoch: Epoch[val] [4][1000]
            # by iter: Iter[val] [1000]
            if self.by_epoch:
                log_str = (
                    f'Epoch({log_dict["mode"]}) '
                    f'[{log_dict["epoch"]}][{log_dict["iter"]}]\t'
                )
            else:
                log_str = f'Iter({log_dict["mode"]}) [{log_dict["iter"]}]\t'

        log_items = []
        for name, val in log_dict.items():
            # TODO: resolve this hack
            # these items have been in log_str
            if name in [
                "mode",
                "Epoch",
                "iter",
                "lr",
                "time",
                "data_time",
                "memory",
                "epoch",
            ]:
                continue
            if isinstance(val, float):
                val = f"{val:.4f}"
            log_items.append(f"{name}: {val}")
        log_str += ", ".join(log_items)

        runner.logger.info(log_str)

    def _round_float(self, items):
        if isinstance(items, list):
            return [self._round_float(item) for item in items]
        elif isinstance(items, float):
            return round(items, 5)
        else:
            return items

    def log(self, runner) -> OrderedDict:
        if "eval_iter_num" in runner.log_buffer.output:
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = runner.log_buffer.output.pop("eval_iter_num")
        else:
            cur_iter = self.get_iter(runner, inner_iter=True)

        log_dict = OrderedDict(
            mode=self.get_mode(runner), epoch=self.get_epoch(runner), iter=cur_iter
        )

        # only record lr of the first param group
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            log_dict["lr"] = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            log_dict["lr"] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict["lr"].update({k: lr_[0]})

        if "time" in runner.log_buffer.output:
            # statistic memory
            if torch.cuda.is_available():
                log_dict["memory"] = self._get_max_memory(runner)

        log_dict = dict(log_dict, **runner.log_buffer.output)  # type: ignore

        self._log_info(log_dict, runner)
        return log_dict

    def after_run(self, runner) -> None:
        pass