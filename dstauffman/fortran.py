# -*- coding: utf-8 -*-
r"""
Functions that make it easier to deal with Fortran code, specifically creating unit tests.

Notes
-----
#.  Written by David C. Stauffer in December 2019.

"""

#%% Imports
import doctest
import glob
import os
import unittest

from dstauffman.utils import line_wrap, read_text_file, write_text_file, pprint_dict

#%% Constants
# Maximum line length to use in any Fortran file
_MAX_LINE_LENGTH = 132

#%% Classes - _FortranSource
class _FortranSource():
    r"""
    Class to contain information about the Fortran source code.

    Parameters
    ----------
    mod_name : str
        Module name

    Examples
    --------
    >>> from dstauffman.fortran import _FortranSource
    >>> code = _FortranSource('test_mod')

    """
    def __init__(self, mod_name='', prog_name=''):
        r"""
        Creates the instance of the class.
        """
        self.prog_name   = prog_name
        self.mod_name    = mod_name
        self.uses        = []
        self.types       = []
        self.functions   = []
        self.subroutines = []
        self.prefix      = ''

    def validate(self):
        r"""
        Validates that the resulting parse is good.
        """
        if self.prog_name:
            # If a program, then it should only have uses
            assert not self.mod_name, 'Programs should not define modules.'
            assert len(self.types) == 0, 'Programs should not define types.'
            assert len(self.functions) == 0, 'Programs should not define functions.'
            assert len(self.subroutines) == 0, 'Programs should not define subroutines.'
        elif self.mod_name:
            assert not self.prog_name, 'Modules should not also define programs.'
        else:
            assert False, 'Either a module or program unit should be defined.'

    def pprint(self, indent=1, align=True):
        r"""
        Displays a pretty print version of the class.

        Parameters
        ----------
        indent : int, optional, default is 1
            Number of characters to indent before all the fields
        align : bool, optional, default is True
            Whether to align all the equal signs

        Examples
        --------
        >>> from dstauffman.fortran import _FortranSource
        >>> code = _FortranSource('test_mod')
        >>> code.pprint() # doctest: +NORMALIZE_WHITESPACE
        _FortranSource
         prog_name   =
         mod_name    = test_mod
         uses        = []
         types       = []
         functions   = []
         subroutines = []
         prefix      =

        """
        pprint_dict(self.__dict__, name=self.__class__.__name__, indent=indent, align=align)

    @property
    def name(self):
        r"""Gets the name of the code, whether module or program."""
        if self.prog_name:
            if self.mod_name:
                raise AssertionError('Code should not have a program and module defined.')
            return self.prog_name
        else:
            return self.mod_name

    @property
    def has_setup(self):
        r"""Whether the Module has a setup routine or not."""
        return 'setup' in self.subroutines

    @property
    def has_setupclass(self):
        r"""Whether the Module has a setupclass routine or not."""
        return 'setupmethod' in self.subroutines # TODO get real name from FRUIT

    @property
    def has_teardown(self):
        r"""Whether the Module has a teardown routine or not."""
        return 'teardown' in self.subroutines # TODO get real name from FRUIT

    @property
    def has_teardownclass(self):
        r"""Whether the Module has a teardownclass routine or not."""
        return 'teardownclass' in self.subroutines # TODO get real name from FRUIT

#%% Functions - _parse_source
def _parse_source(filename, assert_single=True):
    r"""
    Parses the individual fortran source file into relevant information.

    Parameters
    ----------
    filename : str
        Name of file to parse

    Returns
    -------
    code : dict
        Parsing of source code with the following keys:
            uses
            types
            functions
            subroutines

    Examples
    --------
    >>> from dstauffman.fortran import _parse_source
    >>> from dstauffman import get_root_dir
    >>> import os
    >>> filename = os.path.abspath(os.path.join(get_root_dir(), '..', '..', 'forsat', 'unit_tests', \
    ...     'test_utils_unit_vec.f90'))
    >>> code = _parse_source(filename) # doctest: +SKIP

    """
    # read the file
    text = read_text_file(filename)
    # get the lines individually
    lines = text.split('\n')
    # create the output dictionary
    code      = []
    this_name = ''
    this_code  = None
    for line in lines:
        this_line = line.strip().lower()
        if not this_line:
            # empty line
            pass
        elif this_line.startswith('!'):
            # comment line
            pass
        elif this_line.startswith('program '):
            # program declaration
            this_name = this_line.split(' ')[1]
            this_code = _FortranSource(prog_name=this_name)
        elif this_line.startswith('module '):
            # module declaration
            this_name = this_line.split(' ')[1]
            this_code = _FortranSource(mod_name=this_name)
        elif this_line.startswith('end program') or this_line.startswith('endprogram'):
            # program ending
            this_name = ''
            code.append(this_code)
        elif this_line.startswith('end module') or this_line.startswith('endmodule'):
            # module ending
            this_name = ''
            code.append(this_code)
        elif this_line.startswith('use'):
            # use statements
            temp = this_line.split(' ')[1]
            this_use = temp.split(',')[0]
            assert bool(this_use), 'Use statement should not be empty.'
            assert this_name == this_code.name, 'Mismatch in module name "{}" vs "{}".'.format(this_name, this_code.name)
            this_code.uses.append(this_use)
        elif this_line.startswith('type') and not this_line.startswith('type('):
            # type declaration
            temp = this_line.split('::')[1]
            this_type = temp.split(' ')[0]
            assert bool(this_type), 'Type statement should not be empty.'
            assert this_name == this_code.name, 'Mismatch in module name "{}" vs "{}".'.format(this_name, this_code.name)
            this_code.types.append(this_type)
        elif this_line.startswith('function'):
            # function declaration
            temp = this_line.split(' ')[1]
            this_function = temp.split('(')[0]
            assert bool(this_function), 'Function name should not be empty.'
            assert this_name == this_code.name, 'Mismatch in module name "{}" vs "{}".'.format(this_name, this_code.name)
            this_code.functions.append(this_function)
        elif this_line.startswith('subroutine'):
            # subroutine declaration
            temp = this_line.split(' ')[1]
            this_subroutine = temp.split('(')[0]
            assert bool(this_subroutine), 'Subroutine name should not be empty.'
            assert this_name == this_code.name, 'Mismatch in module name "{}" vs "{}".'.format(this_name, this_code.name)
            this_code.subroutines.append(this_subroutine)
        else:
            # normal code line
            pass
    # Validate the final objects
    for this_code in code:
        this_code.validate()
    # Downselect to just the one instance if required
    if assert_single:
        assert len(code) == 1, 'Only one program or module should be in each individual file.'
        return code[0]
    return code

#%% Functions - _write_unit_test
def _write_unit_test(filename, code):
    r"""
    Writes a unit test for the given module.

    Parameters
    ----------
    code : class _FortranSource
        Code breakdown

    """
    # build headers
    lines = []
    lines.append('! Builds the unit test into a program and runs it')
    lines.append('! Autobuilt by dstauffman Fortran code')
    lines.append('')
    lines.append('program run_' + code.mod_name)
    lines.append('    use fruit')
    lines.append('    use ' + code.mod_name)
    lines.append('    call init_fruit')
    # overall setup
    if code.has_setupclass:
        lines.append('    call setupclass')
    # loop through tests
    for this_sub in code.subroutines:
        if not this_sub.startswith('test_'):
            continue
        if code.has_setup:
            lines.append('    call setup')
        lines.append('    call ' + this_sub)
        if code.has_teardown:
            lines.append('    call teardown')
    # overall teardown
    if code.has_teardownclass:
        lines.append('    call teardownclass')
    # end file
    lines.append('    call fruit_summary')
    lines.append('    call fruit_finalize')
    lines.append('end program run_' + code.mod_name)
    lines.append('')
    text = '\n'.join(lines)
    # write out to disk and print updates
    print('Writing "{}".'.format(filename))
    write_text_file(filename, text)

#%% Functions - _write_all_unit_test
def _write_all_unit_test(filename, all_code):
    r"""
    Writes a wrapper run_all_tests program to run all the unit tests.

    Parameters
    ----------
    filename : str
        Name of the file to write
    all_code : dict
        Contents for the unit tests

    Examples
    --------
    # TODO

    """
    # loop through each file
    counter = 1
    for code in all_code:
        code.prefix = 't{}_'.format(counter)
        counter += 1
    # build headers
    lines = []
    lines.append('! Builds all the unit tests into a single program and runs it')
    lines.append('! Autobuilt by dstauffman Fortran code')
    lines.append('')
    lines.append('program run_all_tests')
    lines.append('    !! imports')
    lines.append('    use fruit')
    lines.append('')
    for code in all_code:
        subs = ', '.join((code.prefix + x + '=>' + x for x in code.subroutines))
        this_line = '    use ' + code.mod_name + ', only: ' + subs
        # wrap long lines as appropriate
        if len(this_line) <= _MAX_LINE_LENGTH:
            lines.append(this_line)
        else:
            start_ix = this_line.find('only: ')
            new_lines = line_wrap([this_line], wrap=_MAX_LINE_LENGTH, indent=start_ix+len('only: '), line_cont='&')
            lines.extend(new_lines)
    # initializations
    lines.append('    !! fruit initialization')
    lines.append('    call init_fruit')
    lines.append('')
    lines.append('    !! tests')
    for code in all_code:
        lines.append('    ! ' + code.mod_name)
        if code.has_setupclass:
            lines.append('    call ' + code.prefix + 'setupclass')
        for this_sub in code.subroutines:
            if this_sub.startswith('test_'):
                if code.has_setup:
                    lines.append('    call ' + code.prefix + 'setup')
                lines.append('    call ' + code.prefix + this_sub)
    # finalization
    lines.append('')
    lines.append('    !! Fruit finalization')
    lines.append('    call fruit_summary')
    lines.append('    call fruit_finalize')
    lines.append('end program run_all_tests')
    lines.append('')
    text = '\n'.join(lines)
    # write out to disk and print updates
    print('Writing "{}".'.format(filename))
    write_text_file(filename, text)

#%% Functions - _write_makefile
def _write_makefile(makefile, template, code, sources=None):
    r"""
    Reads the given makefile template and inserts the relevant rules based on the given source code.

    Parameters
    ----------
    makefile : str
        Name of the makefile to generate
    template : str
        Contents of the template for the makefile boilerplate
    code : list of class _FortranSource
        Parsed source code objects for building the rules from
    sources : list of str, optional
        Names of the dependent source files that are already built

    Examples
    --------
    >>> from dstauffman.fortran import _write_makefile
    >>> from dstauffman import get_root_dir
    >>> import os
    >>> folder = os.path.abspath(os.path.join(get_root_dir(), '..', '..', 'forsat', 'unit_tests'))
    >>> makefile = os.path.join(folder, 'unit_tests.make')
    >>> template = os.path.join(folder, 'unit_tests_template.txt')
    >>> code = [] # TODO: write this line
    >>> _write_makefile(makefile, template, code) # doctest: +SKIP

    """
    # hard-coded values
    token_src = 'OBJS   = \\'
    token_run = '# main executable'
    token_obj = '# object file dependencies'
    len_line  = 200

    # optional inputs
    if sources is None:
        sources = {}

    # read the template into lines
    orig_lines = template.split('\n')

    # build the program rules
    run_rules = []
    runners   = []
    for this_code in code:
        this_name = this_code.name
        if this_name.startswith('run_'):
            run_rules.append('')
            runners.append(this_name)
            this_rule = this_name + '.exe : ' + this_name + '.f90 $(B)' + this_name + '.obj'
            run_rules.append(this_rule)
            this_depd = ['$(OBJLOC)/' + x + '.obj' if x in sources else '$(B)' + x + '.obj' for x in this_code.uses]
            this_rule = '\t$(FC) $(FCFLAGS) -o ' + this_name + '.exe ' + this_name + \
                '.f90 -I$(OBJDIR) -I$(OBJLOC) ' + ' '.join(this_depd) + ' $(addprefix $(OBJLOC)/,$(OBJS))'
            if this_name == 'run_all_tests':
                this_rule = line_wrap(this_rule, wrap=len_line, indent=8, line_cont='\\')
            run_rules.append(this_rule)
    all_rule = 'all : ' + ' '.join([x + '.exe' for x in runners])

    # build the object file rules
    obj_rules = ['$(B)fruit.obj : fruit.f90']
    for this_code in code:
        obj_rules.append('')
        this_name = this_code.name
        this_depd = ['$(OBJLOC)/' + x + '.obj' if x in sources else '$(B)' + x + '.obj' for x in this_code.uses]
        this_rule = '$(B)' + this_name + '.obj : ' + this_name + '.f90 ' + ' '.join(this_depd)
        if this_name == 'run_all_tests':
            this_rule = line_wrap(this_rule, wrap=len_line, indent=8, line_cont='\\')
        obj_rules.append(this_rule)

    # Update the relevant sections of text
    new_lines = []
    for line in orig_lines:
        new_lines.append(line)
        if line == token_src and sources is not None:
            new_lines.extend(sorted(['       ' + x + '.obj \\' for x in sources]))
        if line == token_run:
            new_lines.append(all_rule)
            new_lines.extend(run_rules)
        if line == token_obj:
            new_lines.extend(obj_rules)

    # write out to disk and print updates
    text = '\n'.join(new_lines)
    print('Writing "{}".'.format(makefile))
    write_text_file(makefile, text)

#%% Functions - create_fortran_unit_tests
def create_fortran_unit_tests(folder, sources=None, template=None):
    r"""
    Parses the given folder for Fortran unit test files to build programs that will execute them.

    Parameters
    ----------
    folder : str
        Folder location to look for unit tests

    Returns
    -------
    (None) - creates run_*.f90 files in the same folder location

    Examples
    --------
    >>> from dstauffman import create_fortran_unit_tests, get_root_dir
    >>> import os
    >>> folder = os.path.abspath(os.path.join(get_root_dir(), '..', '..', 'forsat', 'unit_tests'))
    >>> create_fortran_unit_tests(folder) # doctest: +SKIP

    """
    # find all the files to process
    files = glob.glob(os.path.join(folder, 'test*.f90'))
    # initialize code information
    all_code = []
    # process each file
    for file in files:
        # parse the source code
        code = _parse_source(file, assert_single=True)
        newfile = os.path.join(folder, 'run_' + os.path.split(file)[1])

        # build the individual unit test
        _write_unit_test(newfile, code)
        # save this code for information for the makefile
        all_code.append(code)

    # write run_all_tests file
    _write_all_unit_test(os.path.join(folder, 'run_all_tests.f90'), all_code)

    # Re-parse the source code once all the new files have been written
    files = glob.glob(os.path.join(folder, 'run_*.f90'))
    for file in files:
        code = _parse_source(file, assert_single=True)
        all_code.append(code)

    # Sort the parsed code
    all_code.sort(key=lambda x: x.name)

    # write the master Makefile
    if template is not None:
        makefile = os.path.join(folder, 'unit_tests.make')
        _write_makefile(makefile, template, all_code, sources=sources)

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_fortran', exit=False)
    doctest.testmod(verbose=False)
