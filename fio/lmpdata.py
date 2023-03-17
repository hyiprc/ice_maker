import copy
import io
import os
import sys
from pathlib import Path
from textwrap import indent

import numpy as np

from . import Box, Lmpdump, fobj


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


class alias_list(list):
    """Assign alias to list index
    # e.g., masses['label'] == masses[2]

    masses = alias_list([1,28.085,'Si'])
    masses.set_alias('type',0)
    masses.set_alias('mass',1)
    masses.set_alias('label',2)

    print(masses)
    print(masses.alias)
    """

    def __init__(self, *args, **kwargs):
        list.__init__(self, *args, **kwargs)
        self.alias = kwargs.get('alias', {})

    def __getitem__(self, ix):
        try:
            return list.__getitem__(self, self.alias.get(ix, ix))
        except Exception:
            return list.__getitem__(self, ix)

    def __setitem__(self, ix, value):
        try:
            return list.__setitem__(self, self.alias.get(ix, ix), value)
        except Exception:
            return list.__setitem__(self, ix, value)

    def set_alias(self, alias: str, ix: int):
        self.alias[str(alias)] = int(ix)


class lammpsdata(fobj.sectioned):
    # TODO: guess angle, dihedral from bond list

    def __init__(self, fname, style=None, **kwargs):
        self.verbose = kwargs.get('verbose', True)
        preload = kwargs.get('preload', True)
        fix_style = kwargs.get('fix_style', True)
        """
        Program to read and write LAMMPS data file format

        Available styles::

            'atomic','bond','angle','molecular','full',
            'charge','sphere','ellipsoid','dipole'

        Custom styles::

            'basic','dihedral'

        Correct style override 'sections' and 'addcols

        For 'hybrid' style::

            f1 = lammpsdata('data.test',style='hybrid',
             addcols=['q','volume','a'],addrfmt={'a':int})

            print(f1.box)
            print(f1.atoms[:,f1.s_['type']])
            print(f1.atoms[:,f1.s_['xyz']])

        available attributes (depend on style)::

            'box'
            'masses',
            'atoms',
            'atom_labels' = [
                'id','mol','type','q','x','y','z',
                'mux','muy','muz','diameter','density',
                'ellipsoidflag','volume',
            ]
            'bonds','bond_labels',
            'angles','angle_labels',
            'dihedrals','dihedral_labels',
            'impropers','improper_labels',
            'velocities'

        Usage::

            f2 = lammpsdata('data.test','w',style='hybrid',
             addcols=['q','volume','a'],addwfmt={'a':'%d'})

            # change mass info
            type = '2'
            f2.masses[type]['label'] = 'O'
            f2.masses[type]['mass'] = 16.0
            f2.sync()

            # ---- change atom type ----
            m = f2.masses = [ [1, 12.0,'C'], [2, 16.0, 'O'] ]

            # method 1: direct assignment
            m['1']['type'] = '2'
            m['2']['type'] = '1'
            f2.sync()

            # method 2: swap list items
            m['1'], m['2'] = m['2'], m['1']
            f2.sync()


            f2.write('test.data',add={q:newq})

            # export to other file formats
            f2.export('test.psf')
            f2.export('test.xyz')

        TODO::

            1. image flag: 9 1 4 5.46 -3.0556 -3.4000 -1.644 0 0 0
            2. check data support alias

        """

        if fname == '':
            fname = Path(sys.path[0]) / 'fio/sample.lmp'

        # style input check
        self.style = style_auto = self.detect_style(fname)
        if style_auto is None and style is None:
            print("Please define 'style' keyword argument or in file")
            raise SystemExit
        elif style is None:
            fix_style = False
        elif style_auto is None:
            self.style = style
            fix_style = False
        elif style != style_auto:
            if not fix_style:
                print(
                    f"specified style '{style}' != detected "
                    + f"style '{style_auto}', try fix_style=True"
                )
                raise SystemExit
        else:
            fix_style = False

        # initialize
        self.reset_style(self.style, fix=False, **kwargs)
        Nmax = np.max([len(s) + 5 for s in self.sections])
        kwargs = {'sections': self.sections, 'Nmax': Nmax}
        super(self.__class__, self).__init__(fname, **kwargs)
        self._read_box()
        self._read_masses()
        if preload:
            self._read_atoms()
            topo = ['Bonds', 'Angles', 'Dihedrals', 'Impropers']
            for s in topo:
                self._read_topo(s)
            self._read_velocities()

        # convert style
        if fix_style:
            self.reset_style(style, fix=True)

        # print info
        if self.verbose:
            print('-' * 5 + ' Read Lammps data file ' + '-' * 5)
            if fix_style:
                print(
                    f"{self.fname} (style='{self.style}', fix_style=True" + ")"
                )
                print(
                    f"  Changed style from detected '{style_auto}' to specified '{style}'"
                )
            else:
                print(f"{self.fname} (style='{self.style}')")
            print(self)

    def __str__(self):
        if 'Masses' in self.sections:
            massinfo = f"  masses = {[s[:3] for s in self.masses]}\n"
        else:
            massinfo = ''
        return (
            f"  sections = {self.sections}\n"
            + f"  box = {self.box}\n"
            + massinfo
            + f"  atom_labels = {self.atom_labels}\n"
        )

    def detect_style(self, fname):
        with open(fname, 'r') as f:
            for line in f:
                line = str(line)
                if 'Atoms' in line:
                    ln = line.split('#')
                    if len(ln) > 1:
                        l1 = ln[1].strip()
                        if l1 != '':
                            return l1
                    break
        if self.verbose:
            print("No styles definied in file (line: Atoms  # style)")
        return None

    def set_style(self, *args, **kwargs):
        kwargs['fix'] = True
        self.reset_style(*args, **kwargs)

    def reset_style(self, style, **kwargs):  # noqa: C901
        verbose = kwargs.get('verbose', False)
        sections = kwargs.get('sections', [])
        addcols = kwargs.get('addcols', [])
        addrfmt = kwargs.get('addrfmt', {})
        addwfmt = kwargs.get('addwfmt', {})
        # ----- define sections to include -----
        basic = ['header', 'Masses', 'Atoms', 'Velocities']
        topo = ['Bonds', 'Angles', 'Dihedrals', 'Impropers']
        atom_style = {
            'basic': basic,
            'dihedral': basic + topo[:3],
            'bond': basic + [topo[0]],
            'angle': basic + topo[:2],
            'molecular': basic + topo,
            'full': basic + topo,
        }
        for sty in ['atomic', 'sphere', 'ellipsoid', 'charge', 'dipole']:
            atom_style.update({sty: basic + topo})
        # -----------------------------------
        try:
            sections = atom_style[style]
        except Exception:
            if self.verbose:
                print(f"Non-standard style '{style}', startard are:")

                print('-' * 72)
                print_col(atom_style.keys())
                print('-' * 72)
            if not sections:
                # default sections
                sections = basic + topo
        self.sections = sections

        # fix style
        fix_style = kwargs.get('fix', False)
        if fix_style:
            atom_labels0 = copy.deepcopy(self.atom_labels)
            s_0 = copy.deepcopy(self.s_)

        # ----- define atom columns -----
        def insert(alist, item, pos):
            return alist[:pos] + item + alist[pos:]

        atomic = ['id', 'type', 'x', 'y', 'z']
        molecular = insert(atomic, ['mol'], 1)
        charge = insert(atomic, ['q'], 2)
        atom_labels = {
            'basic': atomic,
            'atomic': atomic,
            'charge': charge,
            'full': insert(molecular, ['q'], 3),
            'dipole': charge + ['mux', 'muy', 'muz'],
            'sphere': insert(atomic, ['diameter'], 2),
            'ellipsoid': insert(atomic, ['ellipsoidflag', 'density'], 2),
        }
        for sty in ['angle', 'bond', 'dihedral', 'molecular']:
            atom_labels.update({sty: molecular})
        # -----------------------------------
        try:
            self.atom_labels = atom_labels[style]
        except Exception:
            style = 'atomic' + ('+ ' + ' '.join(addcols)) * (len(addcols) > 0)
            self.atom_labels = atom_labels['basic'] + addcols
        self.s_ = {col: i for i, col in enumerate(self.atom_labels)}
        self.s_['xyz'] = np.s_[self.s_['x'] : self.s_['z'] + 1]

        # ----- define atom column formats -----
        cols_wfmt = {
            'id': '%d',
            'type': '%d',
            'mol': '%d',
            'q': '%.6f',
            'x': '%.6f',
            'y': '%.6f',
            'z': '%.6f',
            'mux': '%.6f',
            'muy': '%.6f',
            'muz': '%.6f',
            'diameter': '%f',
            'density': '%f',
            'ellipsoidflag': '%d',
            'volume': '%f',
        }
        cols_wfmt.update(addwfmt)
        # -----------------------------------
        self.cols_wfmt = []
        for f in self.atom_labels:
            try:
                self.cols_wfmt.append(cols_wfmt[f])
            except Exception:
                # default to float if not found
                self.cols_wfmt.append('%f')

        # ----- define atom column formats -----
        map_fmt = {'d': int, 'f': float, 's': str}
        cols_rfmt = {}
        for col in cols_wfmt:
            cols_rfmt[col] = map_fmt[cols_wfmt[col][-1]]
        cols_rfmt.update(addrfmt)
        # -----------------------------------
        self.cols_rfmt = []
        for f in self.atom_labels:
            try:
                self.cols_rfmt.append(cols_rfmt[f])
            except Exception:
                # default to float if not found
                self.cols_rfmt.append(float)

        # attribute alias
        self.attr_alias = {
            'box': [
                'xlo',
                'xhi',
                'ylo',
                'yhi',
                'zlo',
                'zhi',
                'xy',
                'xz',
                'yz',
            ],
            'masses': [],
            'atoms': ['atom_labels', 'xyz'] + self.atom_labels,
            'bonds': ['bond_labels'],
            'angles': ['angle_labels'],
            'dihedrals': ['dihedral_labels'],
            'impropers': ['improper_labels'],
            'velocities': [],
        }

        # style change, fix atoms section
        if fix_style:
            atoms = np.zeros((self.atoms.shape[0], len(self.atom_labels)))
            need_input = self.atom_labels[:]
            for s in atom_labels0:
                if s not in self.atom_labels:
                    continue
                atoms[:, self.s_[s]] = self.atoms[:, s_0[s]]
                need_input.remove(s)
            self.atoms = atoms
            self.style = style

            if self.verbose and len(need_input) != 0:
                print('-' * 72)
                print('Warning: style changed, check \n', need_input)
                if verbose:
                    print('\n', self.atom_labels)
                    print('atoms =\n', self.atoms[:5, :], '\n ...')
                print('-' * 72, '\n')

    def _read_box(self):
        tilt = False
        self.box = {}
        self.header = []
        for line in self.read('header'):
            if ' atoms' in line:
                break
            ix = line.find('#')
            self.header.append(line[ix + 1 :].strip())
        for line in self.read('header'):
            ix = line.find('#')
            if ix >= 0:
                line = line[:ix]
            line = line.strip()
            if not line:
                continue
            if ' xlo xhi' in line:
                ln = line.split(None)[:2]
                self.box['xlo'] = float(ln[0])
                self.box['xhi'] = float(ln[1])
            elif ' ylo yhi' in line:
                ln = line.split(None)[:2]
                self.box['ylo'] = float(ln[0])
                self.box['yhi'] = float(ln[1])
            elif ' zlo zhi' in line:
                ln = line.split(None)[:2]
                self.box['zlo'] = float(ln[0])
                self.box['zhi'] = float(ln[1])
            elif ' xy xz yz' in line:
                ln = line.split(None)[:3]
                tilt = True
                self.box['xy'] = float(ln[0])
                self.box['xz'] = float(ln[1])
                self.box['yz'] = float(ln[2])
        #        self.box = fobj.attr_dict(self)
        for s in self.attr_alias['box']:
            if s in ('xy', 'xz', 'yz') and not tilt:
                setattr(self, s, 0.0)
                self.box.update({s: 0.0})
                continue

    #            self.box.update({s:getattr(self,s)})
    # ----------------------------------------------------
    def new_masses(self, N):
        N = int(N)
        return alias_list(
            [
                alias_list(
                    # ID, mass, label
                    [str(i), 'NULL', '', i],
                    alias={'type': 0, 'mass': 1, 'label': 2, 'from': 3},
                )
                for i in range(1, N + 1)
            ],
            alias={str(i + 1): i for i in range(N)},
        )

    def trim_masses(self):
        typ = self.atoms[:, self.s_['type']]
        self.masses = alias_list(
            self.masses[: int(np.max(typ))], alias=self.masses.alias
        )

    def set_masses(self, masses=[]):
        if not masses:
            N = np.max(self.atoms[:, self.s_['type']])
        else:
            N = len(masses)
        out = self.masses = self.new_masses(N)
        for s in masses:
            out[str(s[0])]['mass'] = float(s[1])
            try:
                out[str(s[0])]['label'] = str(s[2])
            except Exception:
                pass
            try:
                out[str(s[0])]['from'] = s[3]
            except Exception:
                pass
        return out

    def _read_masses(self):
        for line in self.read('header'):
            if ' atom types' in line:
                N = int(line.split(None, 1)[0])
                break
        self.masses = self.new_masses(N)
        if 'Masses' not in self.sections:
            return
        for line in self.read('Masses', skiprows=2):
            line = line.strip()
            if line == '' or line[0] == '#':
                continue
            # TODO: Pair coeffs, Bond coeffs, Angle coeffs, etc.
            if ' Coeffs' in line:
                break
            m = line.split(None, 2)
            try:
                mass = float(m[1])
            except Exception:
                mass = None
            self.masses[m[0]]['mass'] = mass
            try:
                name = m[2].replace('#', '').strip()
                if name:
                    label = name
                else:
                    label = ''
            except Exception:
                label = ''
            self.masses[m[0]]['label'] = label

    # ----------------------------------------------------
    def _update_ID_type(self):
        # ID_type mapping
        IDs = self.atoms[:, self.s_['id']].astype(int)
        typ = self.atoms[:, self.s_['type']].astype(int)
        self.ID_type = np.zeros(np.max(IDs), dtype=int)
        self.ID_type[IDs - 1] = typ

    #        self.trim_masses()

    def _read_atoms(self):
        self.atoms = np.atleast_2d(
            np.loadtxt(
                io.BytesIO(next(self.read('Atoms', skiprows=2, raw=True))),
                dtype=float,
            )
        )
        self._update_ID_type()
        # strip image flags, TODO: maybe handle this
        if self.atoms.shape[1] == len(self.atom_labels) + 3:
            self.atoms = self.atoms[:, :-3]

    # ----------------------------------------------------
    def _get_topo_labels(self, s2):
        try:
            labels = np.take(self.ID_type, getattr(self, s2) - 1)
        except IndexError:
            print(
                f"'{s2.capitalize()}' section has invalid atom 'id',"
                f'\n  {self.fname}',
                trace=0,
            )
            raise SystemExit
        ix = labels[:, 0] > labels[:, -1]
        labels[ix, :] = labels[ix, ::-1]
        _, ix1 = np.unique(labels, axis=0, return_index=True)
        out = [
            '%d|' % (i + 1) + '_'.join('%d' % s for s in l)
            for i, l in enumerate(labels[np.sort(ix1)])
        ]
        out2 = np.zeros(labels.shape[0])
        for ln in out:
            l1, l2 = ln.split('|')
            l2 = [int(l3) for l3 in l2.split('_')]
            ix = labels[:, 0] == l2[0]
            for i, j in enumerate(l2[1:]):
                ix = ix & (labels[:, i + 1] == j)
            out2[ix] = int(l1)
        setattr(self, s2[:-1] + '_types', out2)
        return out

    def _read_topo(self, s):
        if s not in self.sections:
            return
        s2 = s.lower()
        setattr(
            self,
            s2,
            np.atleast_2d(
                np.loadtxt(
                    io.BytesIO(next(self.read(s, skiprows=2, raw=True))),
                    dtype=int,
                )
            )[:, 2:],
        )
        setattr(self, s2[:-1] + '_labels', self._get_topo_labels(s2))

    # ---
    def _read_bonds(self):
        self._read_topo('Bonds')

    def _read_angles(self):
        self._read_topo('Angles')

    def _read_dihedrals(self):
        self._read_topo('Dihedrals')

    def _read_impropers(self):
        self._read_topo('Impropers')

    # ----------------------------------------------------
    def _read_velocities(self):
        if 'Velocities' not in self.sections:
            return
        self.velocities = np.atleast_2d(
            np.loadtxt(
                io.BytesIO(
                    next(self.read('Velocities', skiprows=2, raw=True))
                ),
                dtype=float,
            )
        )
        if not np.any(self.velocities[:, 1:]):
            del self.velocities
            self.sections.remove('Velocities')

    def sync(self):
        self._update_ID_type()
        # add newly defined
        topo = ['Bonds', 'Angles', 'Dihedrals', 'Impropers', 'Velocities']
        for section in topo:
            attr = section.lower()
            if section not in self.sections:
                try:
                    delattr(self, attr)
                except Exception:
                    pass
            elif hasattr(self, attr):
                # ensure 2d array
                setattr(self, attr, np.atleast_2d(getattr(self, attr)))
                if section not in self.sections:
                    self.sections.append(section)
            else:
                if section in self.sections:
                    self.sections.remove(section)

        # atom type mapping

        if len(set([str(s['type']) for s in self.masses])) != len(
            self.masses
        ) or len(self.masses) != len(
            set([int(s['from']) for s in self.masses])
        ):
            print(
                "Duplicated 'type' (index 0) "
                "or 'from' (index 3) in masses" + f"\n{self.masses}",
                trace=0,
            )
            raise SystemExit

        remap = {}
        for i, s in enumerate(self.masses, 1):
            atyp = int(s['type'])
            if s['from'] != atyp:
                remap[s['from']] = atyp
                self.masses[str(i)]['from'] = atyp
            elif s['from'] != i:
                remap[s['from']] = i
                self.masses[i - 1]['from'] = i
                self.masses[i - 1]['type'] = str(i)
        if remap:
            col = self.s_['type']
            tmp = self.atoms[:, col].copy()
            for i in remap:
                self.atoms[tmp == i, col] = remap[i]
            self.masses = alias_list(
                sorted(self.masses, key=lambda _: int(_['from'])),
                alias=self.masses.alias,
            )

    def write(self, fname, **kwargs):  # noqa: C901
        """
        style: str         # atom style

        header: str=''     # custom header
        dated: bool=True   # add timestamp to header

        add: dict={}       # add atom column
        """
        # TODO: remove atom if id==0, clean up topo list if any 0

        self.sync()

        # get target sections and atom_labels
        style = kwargs.get('style', self.style)
        fout = lammpsdata('', style=style, verbose=False)
        fout.reset_style(style)
        fout.sections = kwargs.get('sections', fout.sections)

        if self.verbose:
            print('-' * 5 + ' Write Lammps data file ' + '-' * 5)
            print(f"{fname} (style='{style}')")
            print(f"atom_labels = {fout.atom_labels}\n")
            print(f"  namespace = {self.fname} (style='{self.style}')")
            print(self)

        # check for missing atom columns
        add = kwargs.get('add', {})
        chk = [s for s in fout.atom_labels if not (s in self.s_ or s in add)]
        if chk:
            add = {s: [] for s in chk}
            print(
                'atoms section missing columns, add using'
                + f'\n\n    f.write("{fname}"'
                + f', add={add})\n',
                trace=0,
            )
            raise SystemExit

        # add atom columns
        for s in fout.atom_labels:
            if s in self.s_:
                continue
            col = np.array(add[s])
            if isinstance(add[s], (int, float)):
                col = np.repeat(col, self.atoms.shape[0])
            self.atoms = np.c_[self.atoms, col]
            self.s_[s] = self.atoms.shape[1] - 1

        # write data to file
        f = open(fname, 'w')
        # ----- header -----
        header0 = 'LAMMPS data file.'
        header = [s for s in self.header if not s.strip() == header0]
        header = kwargs.get('header', '\n'.join(header).strip())

        if kwargs.get('dated', True):
            import time

            t = time.time()
            header = time.strftime(
                f'%Y-%m-%d %Z %H:%M:%S , timestamp = {t}'
                + f'\n{header}' * bool(header.strip()),
                time.localtime(t),
            )
        f.write(indent(f"{header0}\n{header}", '# ', lambda line: True) + '\n')
        f.write('\n %d atoms\n' % getattr(self, 'atoms').shape[0])
        # -------------------------
        topo = ['Bonds', 'Angles', 'Dihedrals', 'Impropers']
        for section in topo:
            attr = section.lower()
            c1 = section not in fout.sections
            c2 = not hasattr(self, attr)
            c3 = section in self.sections
            if c1 or c2:
                if not c1:
                    fout.sections.remove(section)
                if self.verbose and c1 != c2 and c3:
                    print(
                        'WARNING: '
                        + section
                        + ' not' * c1
                        + ' in self.sections,'
                        + ' but '
                        + 'no ' * c2
                        + attr
                        + ' defined'
                    )
                continue
            f.write(' %d %s\n' % (getattr(self, attr).shape[0], attr))
        f.write(' %d %s types\n' % (len(self.masses), 'atom'))
        for section in fout.sections:
            if section not in topo:
                continue
            attr = section.lower()
            labels = self._get_topo_labels(attr)
            setattr(self, attr + '_labels', labels)
            f.write(' %d %s types\n' % (len(labels), attr[:-1]))
        # ----- unit cell dimension -----
        f.write(' %.7f %.7f  xlo xhi\n' % (self.box['xlo'], self.box['xhi']))
        f.write(' %.7f %.7f  ylo yhi\n' % (self.box['ylo'], self.box['yhi']))
        f.write(' %.7f %.7f  zlo zhi\n' % (self.box['zlo'], self.box['zhi']))
        if self.box['xy'] + self.box['xz'] + self.box['yz'] == 0:
            ortho = '#'
        else:
            ortho = ''
        ortho = kwargs.get('ortho', ortho)
        f.write(
            '%s %.7f %.7f %.7f  xy xz yz\n'
            % (ortho, self.box['xy'], self.box['xz'], self.box['yz'])
        )
        # ----- Pair Coeffs -----
        # Pair Coeffs
        #
        # 1  Al
        # 2  O
        # 3  H
        # 4  OT
        # 5  HT

        # Bond Coeffs
        #
        # 1  OT-HT

        # Angle Coeffs
        #
        # 1  HT-OT-HT
        # ----- Masses -----
        f.write('\n Masses\n\n')
        for (type, mass, label, ID) in sorted(
            self.masses, key=lambda _: int(_[0])
        ):
            if label:
                label = '# ' + label
            try:
                mass = '%.6g' % mass
            except Exception:
                mass = 'NULL'
            f.write('%d %s  %s\n' % (int(type), mass, label))
        # ----- Atoms -----
        f.write('\n Atoms' + ' # %s' % style + '\n\n')
        f.close()
        index = np.argsort(self.atoms[:, 0])
        if np.sum(np.bincount(index) > 1):
            print('WARNING: duplicated id')
        f = open(fname, 'ab')
        cols = [self.s_[s] for s in fout.atom_labels]
        np.savetxt(f, self.atoms[:, cols][index], fmt=fout.cols_wfmt)
        f.close()
        f = open(fname, 'a')
        # ----- output Bonds, Angles, Dihedrals, Impropers -----
        for section in fout.sections:
            try:
                getattr(self, section.lower())
            except Exception:
                continue
            if section not in topo:
                continue
            s = section.lower()
            f.write('\n %s\n\n' % section)
            f.close()
            # output bonds
            ID = np.arange(getattr(self, s).shape[0]) + 1
            typ = getattr(self, s[:-1] + '_types')
            cols = np.c_[ID, typ, getattr(self, s)]
            f = open(fname, 'ab')
            np.savetxt(f, cols, fmt='%d ' * cols.shape[1])
            f.close()
            f = open(fname, 'a')
        # ----- output Velocities -----
        try:
            if (
                'Velocities' in fout.sections
                and self.velocities.shape[0] == self.atoms.shape[0]
            ):
                f.write('\n Velocities\n\n')
                f.close()
                f = open(fname, 'ab')
                np.savetxt(f, self.velocities, fmt='%d %.6f %.6f %.6f')
                f.close()
            f = open(fname, 'a')
        except Exception:
            f = open(fname, 'a')
        # ---- end -----
        f.write('\n')
        if self.verbose:
            print('file written: %s\n' % fname)
        return self

    def duplicate(self, num):
        self.sync()
        N = self.atoms.shape[0]
        self.seg = np.repeat(np.arange(num) + 1, N)
        maxID = int(np.max(self.atoms[:, self.s_['id']]))
        self.atoms = np.tile(self.atoms, (num, 1))
        for i in range(1, num):
            self.atoms[i * N : i * N + N, 0] += i * N
        for s in ['Bonds', 'Angles', 'Dihedrals', 'Impropers']:
            if s not in self.sections:
                continue
            attr = s.lower()
            N = getattr(self, attr).shape[0]
            setattr(self, attr, np.tile(getattr(self, attr), (num, 1)))
            for i in range(1, num):
                getattr(self, attr)[i * N : i * N + N, :] += i * maxID
        return self.seg

    def update(self, dumpfile, fr=0):
        """read dumpfile, update box and xyz"""
        self.sync()
        # read dump file
        dumpdata = Lmpdump.read(dumpfile)[fr]
        # update box
        for s in self.box:
            self.box[s] = dumpdata.attrs['box'][s]
        # update atoms
        s_ = dumpdata.attrs['s_']
        atoms = dumpdata.values
        ix0 = (np.argsort(dumpdata.index),)
        self.atoms = np.zeros(
            (atoms.shape[0], len(self.atom_labels)), dtype=float
        )
        self.atoms[:, 0] = ix0
        for s in self.atom_labels[1:]:
            self.atoms[:, self.s_[s]] = atoms[ix0, s_[s]]
        return self

    def compress_IDs(self, sort=None, remap=False):
        """ensure sorted id is is continuous

        sort = ['mol','type','xyz']
        # sort by 'mol' then 'type' then
                  'x' then 'y' then 'z'

        remap = True
        # overwrite id, i.e., remap [1,4,2,3] to [1,2,3,4]
        """

        self.sync()

        # --- sort atom rows ---
        if sort is not None:

            def getint(s):
                try:
                    return int(s)
                except Exception:
                    pass
                # s_ label
                try:
                    ss = self.s_[s]
                except Exception:
                    return None
                try:
                    return int(ss)
                except Exception:
                    pass
                # slice object
                return [i for i in range(ss.start, ss.stop)]

            ix = list(
                flatten(
                    [
                        getint(s)
                        for s in flatten([sort])
                        if not getint(s) is None
                    ]
                )
            )
            ix += [getint(s) for s in self.atom_labels if not self.s_[s] in ix]
            # lexsort sort last in list first
            self.atoms = self.atoms[
                np.lexsort([self.atoms[:, i] for i in ix[::-1]])
            ]

        # --- remap atom ID ---
        IDs0 = np.copy(self.atoms[:, self.s_['id']]).astype(int)
        if remap:
            self.atoms[:, self.s_['id']] = np.arange(1, len(IDs0) + 1)
        else:
            sort = np.argsort(IDs0)
            IDs2 = IDs0[sort] + (1 - np.min(IDs0))
            # find gaps
            diff = IDs2[1:] - IDs2[:-1] - 1
            cut = np.where(diff != 0)[0] + 1
            for i in range(len(cut)):
                IDs2[cut[i] :] -= diff[cut[i] - 1]
            IDs2 = IDs2[np.argsort(sort)]  # undo sort
            self.atoms[:, self.s_['id']] = IDs2

        # --- remap bonds ---
        IDmap = np.zeros(np.max(IDs0) + 1, dtype=int)
        IDmap[IDs0] = self.atoms[:, self.s_['id']]
        for s in ['Bonds', 'Angles', 'Dihedrals', 'Impropers']:
            if s not in self.sections:
                continue
            ary = np.take(IDmap, getattr(self, s.lower()))
            mask = np.all(ary, axis=1)  # remove non-existing
            setattr(self, s.lower(), ary[mask])

    def export(self, outfile, modify={}, Ndup=1):
        # check file support
        supported = ['xyz', 'pdb', 'poscar']
        outname = os.path.splitext(outfile)
        ext = outname[1][1:].lower()
        if ext not in supported:
            print('only support {0}'.format(','.join(supported)))
            raise SystemExit
        # initialize
        self.sync()
        if self.verbose:
            print('\n----- Export data -----')
        N = self.atoms.shape[0]
        #        exportdata = type('',(),{})() # an empty object
        #        setattr(exportdata,'sourcefile',self.fname)
        #        for attr in modify:
        #            if len(modify[attr]) == 1:
        #                setattr(exportdata,attr,modify[attr]*N)
        #            elif len(modify[attr]) != N:
        #                print('len(%s) != Natoms'%attr)
        #                sys.exit(0)
        #            else:
        #                setattr(exportdata,attr,modify[attr])
        #
        #        # PSF
        #        if ext == 'psf':
        #            psf = fio.load('psf')
        #            alias = {'type':'type','resid':'mol'}
        #            for col in psf.cols:
        #                if not hasattr(exportdata,col):
        #                    try:
        #                        if col in alias:
        #                            setattr(exportdata,col,getattr(self,alias[col]))
        #                        else:
        #                            setattr(exportdata,col,getattr(self,col))
        #                    except Exception:
        #                        setattr(exportdata,col,None)
        #                sys.stdout.write('%7s: '%col)
        #                dg = getattr(exportdata,col)
        #                if type(dg) is np.ndarray or dg is None:
        #                    print(dg)
        #                else:
        #                    t = type(dg[0])
        #                    fmt = (t is int)*'%d '+(t is float)*'%g '+(t is str)*'%s '
        #                    sys.stdout.write('[')
        #                    for i in range(3):
        #                        sys.stdout.write(fmt%dg[i])
        #                    sys.stdout.write('..., ')
        #                    for i in range(3):
        #                        sys.stdout.write(fmt%dg[i])
        #                    sys.stdout.write(']\n')
        #            # bond, angle, dihedral, improper
        #            for sect in psf.optional_sections:
        #                try:
        #                    dat = getattr(self,sect)[:,2:]
        #                    setattr(exportdata,sect,dat)
        #                    print('%d %s:'%(len(dat),sect))
        #                    print(dat[:5,:])
        #                except Exception:
        #                    setattr(exportdata,sect,None)
        #                    sys.stdout.write('%7s: '%sect)
        #                    print(getattr(exportdata,sect))
        #            psf.write(outfile,exportdata,Ndup=Ndup)

        # XYZ
        if ext == 'xyz':
            f = open(outfile, 'w')
            s = ['xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi', 'xy', 'xz', 'yz']
            f.write(
                '%d\n' % (N)
                + '# '
                + ','.join(s)
                + ' = '
                + ','.join('%g' % self.box[p] for p in s)
                + '\n'
            )
            f.close()
            f = open(outfile, 'ab')
            index = np.argsort(self.atoms[:, self.s_['id']])
            ix = [self.s_[s] for s in ['x', 'y', 'z']]
            dt = np.dtype(
                [('name', 'U6'), ('x', float), ('y', float), ('z', float)]
            )
            out = np.empty(N, dtype=dt)
            m = np.array(
                [s[2][:2] if s[2] else str(s[0]) for s in self.masses]
            )
            typ = np.take(m, self.atoms[:, self.s_['type']].astype(int) - 1)
            out['name'] = typ
            out['x'] = self.atoms[:, ix[0]]
            out['y'] = self.atoms[:, ix[1]]
            out['z'] = self.atoms[:, ix[2]]
            np.savetxt(f, out, fmt='%s %.10g %.10g %.10g')
            f.close()
            print('--- output: %s' % outfile)

        # PDB
        if ext == 'pdb':
            f = open(outfile, 'wb')
            index = np.argsort(self.atoms[:, self.s_['id']])
            for j in range(Ndup):
                np.savetxt(
                    f,
                    self.xyz[index],
                    fmt='ATOM      1  C   U   P   1    '
                    + '%8.3g' * 3
                    + '  0.00  0.00      P',
                )
            f.close()
            print('--- output: %s' % outfile)

        # POSCAR
        if ext == 'poscar':
            bb = Box.get(self.box, typ='lmp')
            self.compress_IDs(sort=['type', 'xyz'], remap=True)
            self.atoms[:, self.s_['xyz']] -= np.array(
                [bb['xlo'], bb['ylo'], bb['zlo']]
            )
            label = ''
            with open(outfile, 'w') as f:
                f.write(
                    '\n'.join(
                        [
                            f'{label} POSCAR file',
                            '1.0',
                            '\n'.join(
                                [
                                    ' '.join(
                                        [
                                            '%.6f' % bb['v'][i, j]
                                            for j in (0, 1, 2)
                                        ]
                                    )
                                    for i in (0, 1, 2)
                                ]
                            ),
                            ' '.join([s['label'] for s in self.masses]),
                            ' '.join(
                                [
                                    '%d' % d
                                    for d in np.bincount(
                                        self.atoms[:, self.s_['type']].astype(
                                            int
                                        )
                                    )[1:]
                                ]
                            ),
                            'Cartesian',
                        ]
                    )
                    + '\n'
                )
            with open(outfile, 'ab') as f:
                np.savetxt(
                    f, self.atoms[:, self.s_['xyz']], fmt='%.7f %.7f %.7f'
                )
            print(f'exported {self.fname} to %s' % outfile)


if __name__ == '__main__':
    pass

"""
http://www.smcm.iqfr.csic.es/docs/lammps/atom_style.html
angle   bonds and angles    bead-spring polymers with stiffness
atomic  only the default values coarse-grain liquids, solids, metals
bond    bonds   bead-spring polymers
charge  charge  atomic system with charges
dipole  charge and dipole moment    system with dipolar particles
electron    charge and spin and eradius electronic force field
ellipsoid   shape, quaternion for particle orientation, angular momentum   extended aspherical particles
full    molecular + charge  bio-molecules
line    end points, angular velocity    rigid bodies
meso    rho, e, cv  SPH particles
molecular   bonds, angles, dihedrals, impropers uncharged molecules
peri    mass, volume    mesocopic Peridynamic models
sphere  diameter, mass, angular velocity    granular models
tri corner points, angular momentum rigid bodies
wavepacket  charge, spin, eradius, etag, cs_re, cs_im   AWPMD

http://www.smcm.iqfr.csic.es/docs/lammps/read_data.html
angle   atom-ID molecule-ID atom-type x y z
atomic  atom-ID atom-type x y z
bond    atom-ID molecule-ID atom-type x y z
charge  atom-ID atom-type q x y z
dipole  atom-ID atom-type q x y z mux muy muz
electron    atom-ID atom-type q spin eradius x y z
ellipsoid   atom-ID atom-type ellipsoidflag density x y z
full    atom-ID molecule-ID atom-type q x y z
line    atom-ID molecule-ID atom-type lineflag density x y z
meso    atom-ID atom-type rho e cv x y z
molecular   atom-ID molecule-ID atom-type x y z
peri    atom-ID atom-type volume density x y z
sphere  atom-ID atom-type diameter density x y z
tri atom-ID molecule-ID atom-type triangleflag density x y z
wavepacket  atom-ID atom-type charge spin eradius etag cs_re cs_im x y z
hybrid  atom-ID atom-type x y z sub-style1 sub-style2 ...
"""
