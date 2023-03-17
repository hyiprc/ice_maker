import json
from itertools import product
from pathlib import Path

import numpy as np

import fio

all_stacking = [
    s
    # ABC: cis_dimer, abc: trans_dimer, - inverted along x
    for s in product(
        *[
            [
                'A+',
                'B+',
                'C+',
                'a+',
                'b+',
                'c+',
                'A-',
                'B-',
                'C-',
                'a-',
                'b-',
                'c-',
            ]
        ]
        * 3
    )
    if (
        s[0][0].upper() != s[1][0].upper()
        and s[1][0].upper() != s[2][0].upper()
    )
]


# initialize

f1 = fio.Lmpdata('dimer_cis.data')
f2 = fio.Lmpdata('dimer_trans.data')
Path('all_stacking').mkdir(exist_ok=True)

dimer = {
    'cis': f1.atoms[:, f1.s_['xyz']],
    'trans': f2.atoms[:, f2.s_['xyz']],
}

offset = {
    'A': np.array([0.0, 0.0]),
    'B': np.array([2.579019, 0.0]),
    'C': np.array([1.28951, 2.233496]),
    'A_pbc': np.array([3.836912, 2.225212]),
    'B_pbc': np.array([3.836912, 2.225212]),
    'C_pbc': np.array([3.868528, -2.233496]),
    'z': 3.647284,
}
typ = {
    'A': 'cis',
    'B': 'cis',
    'C': 'cis',
    'a': 'trans',
    'b': 'trans',
    'c': 'trans',
}


possible_stacking = []
for stacking in all_stacking:

    # prepare stacking layers

    N = len(stacking) * 2  # *2 to handle pbc
    f3 = fio.Lmpdata('dimer_cis.data')
    f3.box['zhi'] = 0.5 * N * offset['z']
    f3.duplicate(N)
    xyz = f3.atoms[:, f3.s_['xyz']]

    cursor = 0
    for i, layer in enumerate(stacking):
        i0, i1, i2 = cursor, cursor + 6, cursor + 12
        xyz[i0:i1] = dimer[typ[layer[0]]]
        if layer[-1] == '-':
            xyz[i0:i1, 0] *= -1
        S = layer[0].upper()
        xyz[i0:i1, :2] += offset[S]
        xyz[i1:i2, :2] = xyz[i0:i1, :2] + offset[f'{S}_pbc']
        xyz[i0:i2, 2] += i * offset['z']
        cursor += 12  # 4 water molecules

    skip = False
    atyp = f3.atoms[:, f3.s_['type']]
    xyz_H = f3.atoms[atyp == 1, f3.s_['xyz']]
    for j, H in enumerate(xyz_H):
        dist = np.sum((xyz_H[j + 1 :] - H) ** 2.0, axis=1) ** 0.5
        bad = ((np.abs(dist - 1.5139) > 0.003) & (dist < 2.0)) | (
            np.abs(dist - 3.5285) < 0.01
        )
        if np.sum(bad) > 0:
            skip = True
            break
    if skip:
        continue
    else:
        possible_stacking.append(''.join(stacking))

    name = ''.join(stacking)
    name = name.replace('a', 'i')
    name = name.replace('b', 'j')
    name = name.replace('c', 'k')
    f3.write(f"all_stacking/{name}.data")


with open('possible_stacking_blank.json', 'w', encoding='utf8') as f:
    json.dump(
        {'possible_stacking': {s: "" for s in possible_stacking}},
        f,
        indent=4,  # pretty print
        ensure_ascii=False,
    )
