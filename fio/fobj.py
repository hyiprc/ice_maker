class sectioned(object):
    """For file type with sections. The subroutine
    records byte position of each line and the line
    number that mark the beginning of a section.

    Args:
        fname (str): Filename.


    """

    def __init__(self, fname, **kwargs):
        """
        only line < Nmax are potential headline
        (for efficiency purpose)

        self.
        lb        start byte position of each line
        sect      start and end line number of all sections
        sections  all available sections
        """
        # process args
        default = {'sections': None, 'Nmax': 60, 'comments': '#'}
        for kw in default:
            try:
                setattr(self, kw, kwargs.pop(kw))
            except Exception:
                setattr(self, kw, default[kw])
        self.f = open(fname, 'rb')
        self.fname = fname
        # initialize
        self.sections0 = ['header']
        self.lb, self.sect = [0], {'header': [1, None]}
        self._parse()
        return

    def __repr__(self):
        return 'file: %s(%s)' % (self.__class__.__name__, self.fname)

    def __str__(self):
        print('ID   Lines  Section_Name')
        for i, sect in enumerate(self.sections):
            Nline = self.sect[sect][1] - self.sect[sect][0] + 1
            print('%-4d %-6d %s' % (i, Nline, sect))
        print('-' * 40)
        if self.__class__.__name__ == 'sectioned':
            print('self.read(Section_Name, skiprows=0) to extract')
        return 'file: %s("%s")' % (self.__class__.__name__, self.fname)

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def _parse(self):
        """
        Parse byte positions of linebreak.
        This is used for seeking file with variable
        line length.

        This procedure also process sections
        """
        Nbyte = 0  # byte counter
        for nl, line in enumerate(self.f):
            # byte position of linebreak
            Nbyte += len(line)
            self.lb.append(Nbyte)
            # save time
            line = line.decode('utf-8')
            ix = line.find(self.comments)
            if ix >= 0:
                line = line[:ix]
            line = line.strip()
            if (not line) or (len(line) > self.Nmax):
                continue
            # process sections
            try:
                for i, section in enumerate(self.sections):
                    if section in line:
                        # line number of section
                        self.sect[self.sections0[-1]][1] = nl
                        self.sections0.append(section)
                        self.sections.pop(i)
                        self.sect.update({section: [nl + 1, None]})
            except Exception:
                import re

                if re.search(r'\s{2,}', line) or not re.search(
                    '^[a-zA-Z_]', line
                ):
                    # contain 2 or more consecutive space
                    # not begin with letter or _
                    continue
                self.sect[self.sections0[-1]][1] = nl
                self.sections0.append(line)
                self.sect.update({line: [nl + 1, None]})
        self.sections = self.sections0
        # last line of last section = end of file
        self.sect[self.sections[-1]][1] = nl + 1
        # rewind the file
        self.f.seek(0)
        return

    def read(self, section=None, skiprows=0, raw=False):
        """
        Extract section
        use .decode("utf-8") to get nice printing
        """
        if section not in self.sections:
            return False
        if self.sect[section][0] + skiprows >= self.sect[section][1] + 1:
            return False
        nl0 = self.sect[section][0] - 1 + skiprows
        nl1 = self.sect[section][1]
        self.f.seek(self.lb[nl0])
        if raw:
            yield self.f.read(self.lb[nl1] - self.lb[nl0])
        else:
            for nl in range(nl0, nl1):
                yield self.f.read(self.lb[nl + 1] - self.lb[nl]).decode(
                    'utf-8'
                )
