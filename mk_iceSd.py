import json
import logging

import numpy as np

import fio

# logging.basicConfig(level=logging.DEBUG)

with open('possible_stacking.json', encoding='utf-8') as f:
    jfile = json.load(f)
possible_stacking = jfile['possible_stacking']


# stacking = 'A+B+C+'  # cubic
# stacking = 'a+b-'  # hexagonal

Nlayer = 8

while True:

    random = np.random.randint(0, len(possible_stacking), size=Nlayer)
    stacking = ''.join([list(possible_stacking.keys())[i] for i in random])

    # ----- check the stacking sequence -----

    cubic_hex = ''

    logging.debug('--- check layers ---')
    if len(stacking) >= 6:

        stacking_todo = stacking
        stacking = ''

        # check each triplet
        while True:
            layers = stacking_todo[:6]
            if len(layers) < 6:
                break
            if layers in possible_stacking:
                logging.debug(f"{layers}, good")
                stacking = stacking[:-4] + layers
                stacking_todo = stacking_todo[2:]
                cubic_hex += possible_stacking[layers]
            else:
                logging.debug(f"{layers}, bad")
                stacking_todo = stacking_todo[:4] + stacking_todo[6:]

    # check periodic boundary
    logging.debug('--- check pbc ---')
    while True:
        layers = stacking[-4:] + stacking[:2]
        if len(layers) < 6:
            break
        if layers not in possible_stacking:
            logging.debug(f"{layers}, bad")
            stacking = stacking[:-2]
            cubic_hex = cubic_hex[:-1]
            continue
        else:
            logging.debug(f"{layers}, good")
            layers0 = layers
            layers = stacking[-2:] + stacking[:4]
            if len(layers) < 6:
                break
            if layers not in possible_stacking:
                logging.debug(f"{layers}, good")
                stacking = stacking[:-2]
                cubic_hex = cubic_hex[:-1]
                continue
        logging.debug(f"{layers}, good")
        cubic_hex = possible_stacking[layers] + cubic_hex
        cubic_hex += possible_stacking[layers0]
        break

    if 0.5 * len(stacking) == Nlayer:
        break

if len(stacking) == 0:
    print('no stacking to do, exit')
    raise SystemExit


print('--- create stacking ---')
print('stacking:', stacking)
print('cubic/hex:', cubic_hex)
summary = (
    f"{len(cubic_hex):d} layers,"
    f"{100*cubic_hex.count('c')/len(cubic_hex):.1f}% cubic,"
    f"{100*cubic_hex.count('h')/len(cubic_hex):.1f}% hexagonal\n"
)
print(summary)


# ----- initialize -----

f1 = fio.Lmpdata('dimer_cis.data', verbose=False)
f2 = fio.Lmpdata('dimer_trans.data', verbose=False)

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


# ----- create stacking layers -----

N = len(stacking)
f3 = fio.Lmpdata('dimer_cis.data', verbose=False)
f3.box['zhi'] = 0.5 * N * offset['z']
f3.duplicate(N)
xyz = f3.atoms[:, f3.s_['xyz']]

cursor = 0
for i, (layer, invert) in enumerate(zip(stacking[::2], stacking[1::2])):
    i0, i1, i2 = cursor, cursor + 6, cursor + 12
    xyz[i0:i1] = dimer[typ[layer]]
    if invert == '-':
        xyz[i0:i1, 0] *= -1
    S = layer.upper()
    xyz[i0:i1, :2] += offset[S]
    xyz[i1:i2, :2] = xyz[i0:i1, :2] + offset[f'{S}_pbc']
    xyz[i0:i2, 2] += i * offset['z']
    cursor += 12  # 4 water molecules per layer

f3.verbose = True
f3.header = [
    json.dumps(
        {
            "stacking": stacking,
            "cubic/hex": cubic_hex,
            "summary": summary,
        }
    )
]
f3.write("unit_IceSd_111.data")
