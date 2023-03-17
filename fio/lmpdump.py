import gzip
import io
from pathlib import Path

import pandas as pd

from . import Box


class lammpsdump(object):
    @classmethod
    def read(cls, fname):
        return cls(fname)

    def __init__(self, fname):
        self._fr = []
        self.fname = Path(fname)

        # handle compressed file
        if self.fname.suffix == ".gz":
            self._open = gzip.open
        else:
            self._open = open

        # record starting position of each frame
        Nbyte = 0
        with self._open(self.fname, 'rb') as f:
            for line in f:
                if line.decode().startswith('ITEM: TIMESTEP'):
                    self._fr.append(Nbyte)
                Nbyte += len(line)
            self._fr.append(Nbyte)

        self.Nfr = len(self._fr) - 1  # total num of fr

    def __getitem__(self, fr):
        def _ix(fr):
            if fr < 0:
                fr = self.Nfr + fr
            fr = min(max(0, fr), self.Nfr - 1)
            return fr

        if isinstance(fr, slice):
            sss = [
                default if fr is None else int(fr)
                for (fr, default) in zip(
                    [fr.start, fr.stop, fr.step], [0, self.Nfr, 1]
                )
            ]
            sss[0] = _ix(sss[0])
            sss[1] = _ix(sss[1]) + 1
            out = self[sss[0]]
            for i in range(sss[0] + sss[2], *sss[1:]):
                out += '\n' + self[i]
            return out

        elif isinstance(fr, int):
            ix = _ix(fr)
            with self._open(self.fname, 'rb') as f:
                f.seek(self._fr[ix])
                fbytes = f.read(self._fr[ix + 1] - self._fr[ix])
                fstr = fbytes.decode().rstrip()
                return self.parse(fstr)

    def __iter__(self):
        for fr in range(self.Nfr):
            yield self[fr]

    def readfr(self, fr):
        return self[fr]

    def parse(self, fstr):

        f = io.StringIO(fstr)
        out = {}

        # read box information
        Nheader = 0
        for line in f:
            Nheader += 1
            if line.startswith('ITEM: BOX BOUNDS'):
                if 'xy xz yz' in line:
                    tilt = ''
                else:
                    tilt = ' 0.0 '
                # notilt = not ('xy xz yz' in line)
                Nheader += 3
                box = Box.get(
                    (
                        f.readline()
                        + tilt
                        + f.readline()
                        + tilt
                        + f.readline()
                        + tilt
                    ).split(),
                    typ='dump',
                )
                break

        # read column labels
        for line in f:
            Nheader += 1
            if line.startswith('ITEM: ATOMS'):
                col_labels = line.split()[3:]
                s_ = {col: i for i, col in enumerate(col_labels)}
                break

        # read atoms and create dataframe
        out = pd.read_csv(f, sep=r'\s+', header=None, names=col_labels)
        out.sort_index(inplace=True)
        out.attrs['filetype'] = 'lammps_dump'
        out.attrs['s_'] = s_
        out.attrs['box'] = box

        return out
