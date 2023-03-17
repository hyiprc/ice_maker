import sys
from functools import partial

import numpy as np


def flatten(alist: list):
    """flatten a nested list

    Args:
        alist: A list of items

    Returns:
        flatterned_list, convert to list by list()
    """
    if not isinstance(alist, list):
        yield alist
    else:
        for s in alist:
            if isinstance(s, list):
                for x in flatten(s):
                    yield x
            else:
                yield s


def print_col(
    items: list,
    lw: int = 72,
    ncol: int = 3,
    sort=True,
    exclude=lambda s: not s,
) -> None:
    """print items in column format

    Args:
        items: A list of items to print.
        lw: Line width.
        ncol: Number of columns.
        exclude (lambda_func): Items to exclude.
    """
    ss = [
        str(s) + ' ' * (int(lw / ncol) - len(str(s)))
        for s in items
        if not exclude(s)
    ]
    if sort:
        ss = sorted(ss)
    for i in range(0, len(ss), ncol):
        print(' '.join(ss[i : i + ncol]))


def normalize(a, order=2, axis=-1):
    """
    Returns:
        Normalize row-listed vectors of a.
    """
    try:
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    except Exception:
        # for numpy version < 1.9
        l2 = np.atleast_1d(np.apply_along_axis(np.linalg.norm, axis, a, order))
    l2[l2 == 0] = 1.0
    return np.atleast_1d(np.squeeze(a / np.expand_dims(l2, axis)))


class Box(dict):
    """Conversion between different box formats."""

    order = {
        'basis': [
            'v_a[0]',
            'v_a[1]',
            'v_a[2]',
            'v_b[0]',
            'v_b[1]',
            'v_b[2]',
            'v_c[0]',
            'v_c[1]',
            'v_c[2]',
        ],
        'vmd': ['a', 'b', 'c', 'alpha', 'beta', 'gamma'],
        'dcd': ['a', 'cos_gamma', 'b', 'cos_beta', 'cos_alpha', 'c'],
        'lmp': ['xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi', 'xy', 'xz', 'yz'],
        'dump': ['xlo', 'xhi', 'xy', 'ylo', 'yhi', 'xz', 'zlo', 'zhi', 'yz'],
    }

    attr = []
    for s in order:
        attr += order[s]
    attr = list(set(attr))

    def __init__(self, boxinfo, **kwargs):
        if isinstance(boxinfo, str):
            args = Box.format_argv(boxinfo)
            if args is None:
                Box.usage()
                return
        dict.__init__(self, self.get(boxinfo, **kwargs))
        self.pprint = partial(Box.pprint, self)
        self.bbcheck = partial(Box.bbcheck, self)
        self.extend = partial(Box.extend, self)
        self.wrap = partial(Box.wrap, self)
        self.ghost = partial(Box.ghost, self)

    @staticmethod
    def format_argv(argv):
        # process string
        if isinstance(argv, str):
            argv = argv.split()

        # filename
        if len(argv) == 0:
            return None
        if len(argv) == 1:
            print('TODO: read argv from file')
            return None
        elif not len(argv) in (6, 9):
            return None
        else:
            # remove ,
            if isinstance(argv[0], str):
                argv = [s.replace(',', '') for s in argv if not s == ',']
        # convert to floats
        return np.array(argv, dtype=float)

    @staticmethod
    def guess_typ(argv):

        argv = Box.format_argv(argv)
        N = argv.shape[0]

        if N == 9:
            if argv[0] < argv[1] and argv[2] < argv[3] and argv[4] < argv[5]:
                return 'lmp'
            elif argv[0] < argv[1] and argv[3] < argv[4] and argv[6] < argv[7]:
                return 'dump'
            else:
                return 'basis'  # cell basis vectors (POSCAR)
        elif N == 6:
            if all(
                argv[i] <= 1
                for i in (
                    i
                    for i, s in enumerate(Box.order['dcd'])
                    if s in ('cos_gamma', 'cos_beta', 'cos_alpha')
                )
            ):
                return 'dcd'
            else:
                return 'vmd'
        else:
            return 'unknown'

    @staticmethod
    def get(argv, typ=None, check=True):

        obox = {}

        if isinstance(argv, dict):
            if typ not in (Box.order):
                raise ValueError(
                    "need keyword argument 'typ'"
                    + " ("
                    + ', '.join([s for s in Box.order])
                    + ")"
                )
            # try to get xlo,ylo,zlo
            if all([s in argv for s in ('xlo', 'ylo', 'zlo')]):
                obox['xlo'] = argv['xlo']
                obox['ylo'] = argv['ylo']
                obox['zlo'] = argv['zlo']
            # take only essential arguments
            if typ == 'basis' and 'v' in argv:
                argv = np.ravel(argv['v'])
            else:
                argv = [argv[s] for s in Box.order[typ]]

        if check:

            typ_guess = Box.guess_typ(argv)
            if typ is None:
                typ = typ_guess
            elif not typ == typ_guess:
                return Box.usage()

            argv = Box.format_argv(argv)
            if argv is None:
                return Box.usage()

        if typ == 'dump':
            typ, argv = 'lmp', np.take(argv, [0, 1, 3, 4, 6, 7, 2, 5, 8])

        if typ == 'lmp':
            for i, s in enumerate(Box.order['lmp']):
                obox[s] = argv[i]
            for s in ('x', 'y', 'z'):
                obox['l' + s] = obox[s + 'hi'] - obox[s + 'lo']
            for s in [('xy', 'lx'), ('xz', 'lx'), ('yz', 'ly')]:
                while obox[s[0]] > obox[s[1]]:
                    obox[s[0]] -= obox[s[1]]
            lv = np.array(
                [
                    [obox['lx'], 0.0, 0.0],  # v_a
                    [obox['xy'], obox['ly'], 0.0],  # v_b
                    [obox['xz'], obox['yz'], obox['lz']],  # v_c
                ]
            )

        if typ == 'basis':
            lv = argv.reshape(3, 3)

        if typ == 'basis' or typ == 'lmp':
            obox['a'] = np.sum(lv[0] * lv[0]) ** 0.5
            obox['b'] = np.sum(lv[1] * lv[1]) ** 0.5
            obox['c'] = np.sum(lv[2] * lv[2]) ** 0.5
            u = [normalize(lv[i]).reshape(3) for i in (0, 1, 2)]
            obox['cos_gamma'] = np.dot(u[0], u[1])  # between a b
            obox['cos_beta'] = np.dot(u[0], u[2])  # between a c
            obox['cos_alpha'] = np.dot(u[1], u[2])  # between b c

        elif typ == 'dcd':
            for i, s in enumerate(Box.order['dcd']):
                obox[s] = argv[i]

        elif typ == 'vmd':
            for i, s in enumerate(Box.order['vmd']):
                obox[s] = argv[i]
                if s in ('gamma', 'beta', 'alpha'):
                    obox['cos_' + s] = np.cos(obox[s] * np.pi / 180.0)

        if not typ == 'lmp':
            # lammps box format
            # http://lammps.sandia.gov/doc/Section_howto.html#howto-12
            obox['lx'] = obox['a']
            obox['xy'] = obox['cos_gamma'] * obox['b']
            # obox['ly'] = np.sum(np.cross(lv[0]/LA.norm(lv[0]),lv[1]))**0.5
            obox['ly'] = (obox['b'] ** 2.0 - obox['xy'] ** 2.0) ** 0.5
            obox['xz'] = obox['cos_beta'] * obox['c']
            # obox['yz'] = (np.dot(lv[1],lv[2])-obox['xy']*obox['xz'])/obox['ly']
            if obox['ly'] == 0.0:
                obox['yz'] = 0.0
            else:
                obox['yz'] = (
                    obox['cos_alpha'] * obox['b'] * obox['c']
                    - obox['xy'] * obox['xz']
                ) / obox['ly']
            obox['lz'] = (
                obox['c'] ** 2.0 - obox['xz'] ** 2.0 - obox['yz'] ** 2.0
            ) ** 0.5
        # fix box with large tilt
        for s in [('xy', 'lx'), ('xz', 'lx'), ('yz', 'ly')]:
            while obox[s[0]] > obox[s[1]]:
                obox[s[0]] -= obox[s[1]]

        # alpha, beta, gamma
        if not typ == 'vmd':
            for i, s in enumerate(('alpha', 'beta', 'gamma')):
                obox[s] = np.arccos(obox['cos_' + s]) * 180.0 / np.pi

        # v_a, v_b, v_c
        # if not typ == 'basis': # with this v_a might not align with x
        obox['v'] = np.array(
            [
                [obox['lx'], 0.0, 0.0],  # v_a
                [obox['xy'], obox['ly'], 0.0],  # v_b
                [obox['xz'], obox['yz'], obox['lz']],  # v_c
            ]
        )

        # arbitary xlo,xhi,ylo,yhi,zlo,zhi
        for s in ('xlo', 'ylo', 'zlo'):
            if s not in obox:
                obox[s] = 0.0
        for s in ('x', 'y', 'z'):
            obox[s + 'hi'] = obox[s + 'lo'] + obox['l' + s]

        obox['u'] = normalize(obox['v'])  # useful for coordinate transform
        obox['u_inv'] = np.linalg.inv(
            obox['u']
        )  # useful for undo coordinate transform

        # face normal, useful for cartesian to crystal fractional
        def cross(v, u):
            return normalize(np.cross(v, u))

        obox['bn'] = np.r_[
            cross(obox['u'][1], obox['u'][2]),
            cross(obox['u'][2], obox['u'][0]),
            cross(obox['u'][0], obox['u'][1]),
        ].reshape(-1, 3)

        # get rid of small zero
        p = 10  # number < 1E-p is 0
        for s in obox:
            obox[s] = np.round(obox[s], p)

        return obox

    @staticmethod
    def pprint(obox, typ='all'):
        def argget(t):
            return (
                ' '.join(['%g' % obox[s] for s in Box.order[t]])
                + '  '
                + ' '.join(Box.order[t])
            )

        def fmt_lmp():
            return (
                ' %.7f %.7f  xlo xhi\n' % (obox['xlo'], obox['xhi'])
                + ' %.7f %.7f  ylo yhi\n' % (obox['ylo'], obox['yhi'])
                + ' %.7f %.7f  zlo zhi\n' % (obox['zlo'], obox['zhi'])
                + ' %.7f %.7f %.7f  xy xz yz'
                % (obox['xy'], obox['xz'], obox['yz'])
            )

        def fmt_dump():
            return (
                ' %.7f %.7f %.7f  xlo xhi xy\n'
                % (obox['xlo'], obox['xhi'], obox['xy'])
                + ' %.7f %.7f %.7f  ylo yhi xz\n'
                % (obox['ylo'], obox['yhi'], obox['xz'])
                + ' %.7f %.7f %.7f  zlo zhi yz'
                % (obox['zlo'], obox['zhi'], obox['yz'])
            )

        def fmt_basis():
            v = obox['v']
            return (
                ' %15.9f  %15.9f  %15.9f  basis\n'
                % (v[0, 0], v[0, 1], v[0, 2])
                + ' %15.9f  %15.9f  %15.9f\n' % (v[1, 0], v[1, 1], v[1, 2])
                + ' %15.9f  %15.9f  %15.9f' % (v[2, 0], v[2, 1], v[2, 2])
            )

        def add_lo():
            return ' %.7f %.7f %.7f  xlo ylo zlo' % (
                obox['xlo'],
                obox['ylo'],
                obox['zlo'],
            )

        def add_l():
            return ' %.7f %.7f %.7f  lx ly lz' % (
                obox['lx'],
                obox['ly'],
                obox['lz'],
            )

        if typ == 'all':
            print('\n# ----- poscar -----')
            print(fmt_basis())
            print('\n# ----- vmd -----')
            print(argget('vmd'))
            print('# alpha is between b c, beta a c, gamma a b')
            print('\n# ----- dcd -----')
            print(argget('dcd'))
            print('\n# ----- lammps -----')
            print(add_l())
            print(fmt_lmp())
            print('\n# ----- lammps dump -----')
            print(fmt_dump())
        elif typ == 'lmp':
            print(fmt_lmp())
        elif typ == 'dump':
            print(fmt_dump())
        elif typ == 'basis':
            print(fmt_basis())
        elif typ == 'vmd':
            print(argget('vmd'))
        elif typ == 'dcd':
            print(argget('dcd'))
        else:
            print(fmt_basis())
            print(add_lo())

    @staticmethod
    def usage():
        print("\nUsage: box convert [boxtype] arguments")
        print('-' * 72)
        print_col(
            flatten(
                ['# boxtype', 'arguments']
                + [[s, ' '.join(Box.order[s])] for s in Box.order]
            ),
            lw=20,
            ncol=2,
            sort=False,
        )

    # -----------------------------------------------
    @staticmethod
    def fractional_xyz(box: dict, pts: np.ndarray, verbose=False):
        lo = np.array([box[s] for s in ('xlo', 'ylo', 'zlo')])
        norm = np.sum(np.dot(box['v'], box['bn'].T) ** 2.0, axis=1) ** 0.5
        return np.dot(pts - lo, box['bn'].T) / norm

    # -----------------------------------------------
    @staticmethod
    def bbcheck(box: dict, pts: np.ndarray, verbose=False):
        """check which points in pts is within bounding box"""
        # TODO: pbc on selected faces
        # pbc = (list(flatten(pbc))*3)[:3]  # direction a, b, c
        lo = np.array([box['xlo'], box['ylo'], box['zlo']]) - pts
        hi = lo + np.sum(box['v'], axis=0)

        # normal vectors of box faces
        bn = box['bn']

        # distance from origin to box faces
        # (face_xlo face_ylo face_zlo face_xhi face_yhi face_zhi)
        dist = np.c_[
            np.dot(-bn, np.atleast_2d(lo).T).T,
            np.dot(bn, np.atleast_2d(hi).T).T,
        ]
        inside = np.min(dist, axis=1) >= 0
        N = pts.shape[0] - np.sum(inside)

        if N > 0 and verbose:
            print(
                f"{N} points outside of bounding box, " + "0-index:\n",
                np.where(~inside)[0],
            )

        return {
            'ix_in': inside,
            'ix_out': ~inside,
            'dist': dist,
        }

    @staticmethod
    def extend(box: dict, pts: np.ndarray, bbcheck=None, pbc=False):
        """extend bounding box to accommodate pts"""
        if bbcheck is None:
            bbcheck = Box.bbcheck(box, pts)
        if np.sum(bbcheck['ix_in']) == pts.shape[0]:
            return box
        pbc = (list(flatten(pbc)) * 3)[:3]  # direction a, b, c
        lo0 = np.array([box['xlo'], box['ylo'], box['zlo']])
        # edit lo end
        lo1 = (box['xlo'], box['ylo'], box['zlo']) = np.min(
            np.r_[pts[bbcheck['ix_out']] - 1e-7, np.atleast_2d(lo0)], axis=0
        ) * (~np.array(pbc)).astype(int)
        shift = np.dot(box['u'], np.atleast_2d(lo1 - lo0).T).T
        # edit hi end
        d = pts[bbcheck['ix_out']] - lo0
        d_abc = np.dot(box['u'], np.atleast_2d(d).T).T
        box['v'] = box['u'] * (
            np.atleast_2d(
                np.maximum(
                    (np.max(d_abc, axis=0) + 1e-7)
                    * (~np.array(pbc)).astype(int),
                    np.sum(box['v'] ** 2.0, axis=1) ** 0.5,
                )
            ).T
            - shift.T
        )
        # sync keys in box
        return Box(box, typ='basis')

    @staticmethod
    def wrap(box: dict, pts: np.ndarray, bbcheck=None, pbc=True):
        """move pts to wrap them within bounding box"""
        if bbcheck is None:
            bbcheck = Box.bbcheck(box, pts)
        if np.sum(bbcheck['ix_in']) == pts.shape[0]:
            return pts
        pbc = (list(flatten(pbc)) * 3)[:3]  # direction a, b, c
        rep = np.abs(
            np.floor_divide(
                bbcheck['dist'],
                np.tile(np.sum(box['v'] ** 2.0, axis=1) ** 0.5, 2),
            )
            * np.tile(pbc, 2).astype(int)
        )
        rep[bbcheck['dist'] >= 0] = 0
        shift = np.sum(
            np.multiply(
                np.ravel(rep).reshape(-1, 1),
                np.tile(np.r_[box['v'], -box['v']], (pts.shape[0], 1)),
            ).reshape(-1, 6, 3),
            axis=1,
        )
        return pts + shift

    @staticmethod
    def ghost(box: dict, pts: np.ndarray, pbc=True):
        """get ghost pts in periodic images.

        Must wrap out-of-bound atoms first
        i.e.,  pts = box.wrap(pts)
               for pt in box.ghost(pts):
                   ....
        """
        pbc = (list(flatten(pbc)) * 3)[:3]  # direction a, b, c
        ref = pts  # np.copy(pts)
        L = np.array([box['a'] * pbc[0], box['b'] * pbc[1], box['c'] * pbc[2]])
        shift = L[0] * box['u'][0] + L[1] * box['u'][1] + L[2] * box['u'][2]
        side = (
            np.argmin(
                np.c_[
                    np.sum((pts - shift) ** 2.0, axis=1),
                    np.sum((pts + shift) ** 2.0, axis=1),
                ],
                axis=1,
            )
            * 2
            - 1
        )
        yield ref + np.outer(side, shift)
        for i in (0, 1, 2):
            shift = L[i] * box['u'][i]
            yield ref + np.outer(side, shift)
        for i, j in zip((0, 1, 2), (1, 2, 0)):
            shift = L[i] * box['u'][i] + L[j] * box['u'][j]
            yield ref + np.outer(side, shift)


if __name__ == '__main__':

    argv = Box.format_argv(sys.argv[1:])
    typ = Box.guess_typ(argv)
    print('input: (%s)\n' % typ + ' '.join(map(str, argv)))
    Box.pprint(Box.get(argv, typ))
