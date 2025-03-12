import sys
from pathlib import Path

import sfio

fpath = Path(sys.argv[1])

df = sfio.read(fpath).df

# delete hydrogens and charges
H_atoms = df['type'] == 1
df = df.drop(df[H_atoms].index).drop('q', axis=1)

# change O atoms to WT beads
df['type'] = 1  # set O atoms to type 1
df.attrs['masses'].update({'id': ['1'], 'mass': [18.0153], 'label': ['WT']})

# delete bonds and angles
df.attrs.pop('bonds')
df.attrs.pop('angles')

# output
sfio.write(f'{fpath.stem}_cg.data', df, style='atomic')
