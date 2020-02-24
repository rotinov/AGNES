import os
import os.path as osp
import time
import typing
import abc
import errno

import numpy
from torch.utils.tensorboard import SummaryWriter


def safemean(xs):
    return numpy.nan if len(xs) == 0 else numpy.mean(xs)


def explained_variance(ypred: numpy.ndarray, y: numpy.ndarray):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y.shape == ypred.shape, "Shapes are different"
    y = y.reshape(-1)
    ypred = ypred.reshape(-1)
    vary = numpy.var(y)
    return numpy.nan if vary == 0 else 1 - numpy.var(y-ypred) / vary


class _BaseLogger(abc.ABC):
    folder_name = ""
    log_path = ""

    def __call__(self, kvpairs, nupdates) -> None:
        pass

    def info(self, kvpairs: typing.Dict) -> None:
        self.folder_name = osp.join("_".join([
            kvpairs.get("env_name"),
            kvpairs.get("NN type")
        ]),
            kvpairs.get("algo"),
            str(time.strftime("%Y_%m_%dT%H_%M_%S", time.gmtime())))

    def _create_dirs(self):
        if not osp.isdir(self.log_path):
            try:
                os.makedirs(self.log_path)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

    def close(self) -> None:
        pass


class StandardLogger(_BaseLogger):
    def __init__(self):
        pass

    def info(self, kvpairs) -> None:
        pass

    def __call__(self, kvpairs, nupdates) -> None:
        key2str = {}
        for (key, val) in sorted(kvpairs.items()):
            if isinstance(val, float):
                valstr = '%-8.3g' % (val,)
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        if len(key2str) == 0:
            print('WARNING: tried to write empty key-value dict')
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

            # Write out the data
        dashes = '-' * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append('| %s%s | %s%s |' % (
                key,
                ' ' * (keywidth - len(key)),
                val,
                ' ' * (valwidth - len(val)),
            ))
        lines.append(dashes)
        print('\n'.join(lines))

    @staticmethod
    def _truncate(s):
        maxlen = 30
        return s[:maxlen - 3] + '...' if len(s) > maxlen else s

    def close(self) -> None:
        return


log = StandardLogger()


class TensorboardLogger(_BaseLogger):
    first = True
    writer = None
    folder_name = ""

    def __init__(self,
                 root_dir: str = ".logs/"):
        self.root_dir = root_dir

    def open(self):
        if self.first:
            self.log_path = osp.join(self.root_dir, self.folder_name)
            self._create_dirs()

            filename = osp.join(self.log_path, 'tensorboard')

            self.writer = SummaryWriter(log_dir=filename)
            self.first = False

    def info(self, kvpairs: typing.Dict) -> None:
        super().info(kvpairs)

        self.open()

        for (key, val) in sorted(kvpairs.items()):
            self.writer.add_text(key, str(val), 0)

    def __call__(self, kvpairs: typing.Dict, nupdates: int) -> None:
        self.open()

        for (key, val) in sorted(kvpairs.items()):
            self.writer.add_scalar(key, val, nupdates)

        self.last_nupdate = nupdates

    def __del__(self):
        self.close()

    def close(self) -> None:
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception as e:
                pass


class CsvLogger(_BaseLogger):
    file = None
    folder_name = ""
    log_path = ""

    def __init__(self, root_dir: str = ".logs/"):

        self.root_dir = root_dir

        self.keys = []
        self.sep = ','

    def __call__(self, kvs: typing.Dict, nupdates: int) -> None:
        if self.file is None:
            self.log_path = osp.join(self.root_dir, self.folder_name)

            self._create_dirs()

            filename = osp.join(self.log_path, 'progress.csv')
            self.file = open(filename, 'w+t')

        # Add our current row to the history
        extra_keys = list(kvs.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(',')
                self.file.write(k)
            self.file.write('\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write('\n')
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write('\n')
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __del__(self):
        pass


class ListLogger(_BaseLogger):
    loggers = None

    def __init__(self, *args):
        first_std_logger = False
        for item in args:
            if isinstance(item, StandardLogger):
                assert not first_std_logger, "You can't make more than one StandardLogger."
                first_std_logger = True

        self.loggers = args

    def info(self, kvpairs: typing.Dict) -> None:
        for logger in self.loggers:
            logger.info(kvpairs)

    def __call__(self, kvpairs: typing.Dict, nupdates: int, print_out: bool = True) -> None:
        for logger in self.loggers:
            if (isinstance(logger, StandardLogger) and print_out) or not isinstance(logger, StandardLogger):
                logger(kvpairs, nupdates)

    def stepping_environment(self) -> None:
        for logger in self.loggers:
            if isinstance(logger, StandardLogger):
                print("Stepping environment...")
                return

    def done(self) -> None:
        for logger in self.loggers:
            if isinstance(logger, StandardLogger):
                print("Done.")
                return

    def __del__(self):
        self.close()

    def close(self) -> None:
        if self.loggers is not None:
            for logger in self.loggers:
                logger.close()
