r"""
Functions that make it easier to deal with Fortran code, specifically creating unit tests.

Notes
-----
#.  Written by David C. Stauffer in December 2019.
"""

# %% Imports
from __future__ import annotations

import doctest
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, overload, Union
import unittest

from dstauffman.classes import Frozen
from dstauffman.utils import line_wrap, read_text_file, write_text_file

# %% Constants
# Maximum line length to use in any Fortran file
_MAX_LINE_LENGTH = 132
# Intrinsic modules
_INTRINSIC_MODS = {"ieee_arithmetic", "iso_c_binding", "iso_fortran_env"}
# Object file extension
_OBJ_EXT = ".obj"


# %% Classes - _FortranSource
class _FortranSource(Frozen):
    r"""
    Class to contain information about the Fortran source code.

    Parameters
    ----------
    mod_name : str
        Module name

    Examples
    --------
    >>> from dstauffman.fortran import _FortranSource
    >>> code = _FortranSource("test_mod")

    """

    def __init__(self, mod_name: str = "", prog_name: str = ""):
        r"""Creates the instance of the class."""
        # fmt: off
        self.prog_name: str         = prog_name
        self.mod_name: str          = mod_name
        self.uses: List[str]        = []
        self.types: List[str]       = []
        self.functions: List[str]   = []
        self.subroutines: List[str] = []
        self.prefix: str            = ""
        # fmt: on

    def validate(self) -> None:
        r"""Validates that the resulting parse is good."""
        if self.prog_name:
            # If a program, then it should only have uses
            assert not self.mod_name, "Programs should not define modules."
            assert len(self.types) == 0, "Programs should not define types."
            assert len(self.functions) == 0, "Programs should not define functions."
            assert len(self.subroutines) == 0, "Programs should not define subroutines."
        elif self.mod_name:
            assert not self.prog_name, "Modules should not also define programs."
        else:
            assert False, "Either a module or program unit should be defined."

    @property
    def name(self) -> str:
        r"""Gets the name of the code, whether module or program."""
        if self.prog_name:
            if self.mod_name:
                raise AssertionError("Code should not have a program and module defined.")
            return self.prog_name
        return self.mod_name

    @property
    def has_setup(self) -> bool:
        r"""Whether the Module has a setup routine or not."""
        return "setup" in self.subroutines

    @property
    def has_setupclass(self) -> bool:
        r"""Whether the Module has a setupclass routine or not."""
        return "setupmethod" in self.subroutines  # TODO get real name from FRUIT

    @property
    def has_teardown(self) -> bool:
        r"""Whether the Module has a teardown routine or not."""
        return "teardown" in self.subroutines  # TODO get real name from FRUIT

    @property
    def has_teardownclass(self) -> bool:
        r"""Whether the Module has a teardownclass routine or not."""
        return "teardownclass" in self.subroutines  # TODO get real name from FRUIT


# %% Functions - _parse_source
@overload
def _parse_source(filename: Path, assert_single: Literal[True] = ...) -> _FortranSource:
    ...


@overload
def _parse_source(filename: Path, assert_single: Literal[False]) -> List[_FortranSource]:
    ...


def _parse_source(filename: Path, assert_single: bool = True) -> Union[_FortranSource, List[_FortranSource]]:
    r"""
    Parses the individual fortran source file into relevant information.

    Parameters
    ----------
    filename : class pathlib.Path
        Name of file to parse

    Returns
    -------
    code : list of _FortranSource class instances
        Parsed code

    Examples
    --------
    >>> from dstauffman.fortran import _parse_source
    >>> from dstauffman import get_root_dir
    >>> from pathlib import Path
    >>> filename = get_root_dir().parent.parent.joinpath("forsat", "unit_tests", "test_utils_unit_vec.f90")
    >>> code = _parse_source(filename) # doctest: +SKIP

    """
    # read the file
    text = read_text_file(filename)
    # get the lines individually
    lines = text.split("\n")
    # create the output dictionary
    code: List[_FortranSource] = []
    this_name = ""
    this_code: Optional[_FortranSource] = None
    for line in lines:
        this_line = line.strip().lower()
        if not this_line:
            # empty line
            pass
        elif this_line.startswith("!"):
            # comment line
            pass
        elif this_line.startswith("program "):
            # program declaration
            this_name = this_line.split(" ")[1].strip()
            this_code = _FortranSource(prog_name=this_name)
        elif this_line.startswith("module ") and not this_line.startswith("module procedure"):
            # module declaration
            this_name = this_line.split(" ")[1].strip()
            this_code = _FortranSource(mod_name=this_name)
        elif this_line.startswith("end program") or this_line.startswith("endprogram"):
            # program ending
            this_name = ""
            assert this_code is not None, "_FortranSource class should already be instantiated."
            code.append(this_code)
        elif this_line.startswith("end module") or this_line.startswith("endmodule"):
            # module ending
            this_name = ""
            assert this_code is not None, "_FortranSource class should already be instantiated."
            code.append(this_code)
        elif this_line.startswith("use "):
            # use statements
            if "::" in this_line:
                temp = this_line.split("::")[1].strip()
            else:
                temp = this_line.split(" ")[1].strip()
            this_use = temp.split(",")[0]
            assert bool(this_use), "Use statement should not be empty."
            assert this_code is not None, "_FortranSource class should already be instantiated."
            assert this_name == this_code.name, f'Mismatch in module name "{this_name}" vs "{this_code.name}".'
            this_code.uses.append(this_use)
        elif this_line.startswith("type ") and not this_line.startswith("type("):
            # type declaration
            assert "::" in this_line, 'Line is missing "::" characters: ' + this_line + f' in "{filename}".'
            temp = this_line.split("::")[1].strip()
            this_type = temp.split(" ")[0]
            assert bool(this_type), "Type statement should not be empty."
            assert this_code is not None, "_FortranSource class should already be instantiated."
            assert this_name == this_code.name, f'Mismatch in module name "{this_name}" vs "{this_code.name}".'
            this_code.types.append(this_type)
        elif this_line.startswith("function "):
            # function declaration
            temp = this_line.split(" ")[1].strip()
            this_function = temp.split("(")[0]
            assert bool(this_function), "Function name should not be empty for line: " + this_line + f' in "{filename}".'
            assert this_code is not None, "_FortranSource class should already be instantiated."
            assert this_name == this_code.name, f'Mismatch in module name "{this_name}" vs "{this_code.name}".'
            this_code.functions.append(this_function)
        elif this_line.startswith("subroutine "):
            # subroutine declaration
            temp = this_line.split(" ")[1].strip()
            this_subroutine = temp.split("(")[0]
            assert bool(this_subroutine), "Subroutine name should not be empty for line: " + this_line + f' in "{filename}".'
            assert this_code is not None, "_FortranSource class should already be instantiated."
            assert this_name == this_code.name, f'Mismatch in module name "{this_name}" vs "{this_code.name}".'
            this_code.subroutines.append(this_subroutine)
        else:
            # normal code line
            pass
    # Validate the final objects
    for this_code in code:
        this_code.validate()
    # Downselect to just the one instance if required
    if assert_single:
        assert len(code) == 1, "Only one program or module should be in each individual file."
        return code[0]
    return code


# %% Functions - _write_unit_test
def _write_unit_test(filename: Path, code: _FortranSource, header: Optional[str] = None) -> None:
    r"""
    Writes a unit test for the given module.

    Parameters
    ----------
    filename : class pathlib.Path
        Name of the test file to write
    code : class _FortranSource
        Code breakdown
    header : str, optional
        If specified, write this additional header information

    """
    # build headers
    lines = []
    lines.append("! Builds the unit test into a program and runs it")
    lines.append("! Autobuilt by dstauffman Fortran tools")
    if header:
        lines.append(header)
    lines.append("")
    lines.append("program run_" + code.mod_name)
    lines.append("    use fruit")
    lines.append("    use " + code.mod_name)
    lines.append("    call init_fruit")
    # overall setup
    if code.has_setupclass:
        lines.append("    call setupclass")
    # loop through tests
    for this_sub in code.subroutines:
        if not this_sub.startswith("test_"):
            continue
        if code.has_setup:
            lines.append("    call setup")
        lines.append("    call " + this_sub)
        if code.has_teardown:
            lines.append("    call teardown")
    # overall teardown
    if code.has_teardownclass:
        lines.append("    call teardownclass")
    # end file
    lines.append("    call fruit_summary")
    lines.append("    call fruit_finalize")
    lines.append("end program run_" + code.mod_name)
    lines.append("")
    text = "\n".join(lines)
    # write out to disk and print updates
    print(f'Writing "{filename}".')
    write_text_file(filename, text)


# %% Functions - _write_all_unit_test
def _write_all_unit_test(filename: Path, all_code: List[_FortranSource], header: Optional[str] = None) -> None:
    r"""
    Writes a wrapper run_all_tests program to run all the unit tests.

    Parameters
    ----------
    filename : class pathlib.Path
        Name of the file to write
    all_code : list of _FortranSource instantes
        Contents for the unit tests
    header : str, optional
        If specified, write this additional header information

    Examples
    --------
    # TODO

    """
    # loop through each file
    counter = 1
    for code in all_code:
        code.prefix = f"t{counter}_"
        counter += 1
    # build headers
    lines = []
    lines.append("! Builds all the unit tests into a single program and runs it")
    lines.append("! Autobuilt by dstauffman Fortran tools")
    if header:
        lines.append(header)
    lines.append("")
    lines.append("program run_all_tests")
    lines.append("    !! imports")
    lines.append("    use fruit")
    lines.append("")
    lines.append("    ! Note that these renames need to happen to avoid potential name conflicts between different test files.")
    for code in all_code:
        subs = ", ".join((code.prefix + x + "=>" + x for x in code.subroutines))
        this_line = "    use " + code.mod_name + ", only: " + subs
        # wrap long lines as appropriate
        if len(this_line) <= _MAX_LINE_LENGTH:
            lines.append(this_line)
        else:
            start_ix = this_line.find("only: ")
            new_lines = line_wrap([this_line], wrap=_MAX_LINE_LENGTH, indent=start_ix + len("only: "), line_cont="&")
            lines.extend(new_lines)
    # initializations
    lines.append("    !! fruit initialization")
    lines.append("    call init_fruit")
    lines.append("")
    lines.append("    !! tests")
    for code in all_code:
        lines.append("    ! " + code.mod_name)
        if code.has_setupclass:
            lines.append("    call " + code.prefix + "setupclass")
        for this_sub in code.subroutines:
            if this_sub.startswith("test_"):
                if code.has_setup:
                    lines.append("    call " + code.prefix + "setup")
                lines.append("    call " + code.prefix + this_sub)
    # finalization
    lines.append("")
    lines.append("    !! Fruit finalization")
    lines.append("    call fruit_summary")
    lines.append("    call fruit_finalize")
    lines.append("end program run_all_tests")
    lines.append("")
    text = "\n".join(lines)
    # write out to disk and print updates
    print(f'Writing "{filename}".')
    write_text_file(filename, text)


# %% Functions - _makefile_template
def _get_template(
    compiler: str = "gfortran",
    program: str = "prog",
    is_debug: bool = False,
    build: str = "",
    *,
    fcflags: Optional[Dict[str, str]] = None,
    dbflags: Optional[Dict[str, str]] = None,
    use_preprocessor: bool = False,
) -> str:
    r"""
    Creates a template for the given compiler and debug settings.

    Parameters
    ----------
    compiler
    program
    is_debug
    build
    fcflags
    dbflags

    Returns
    -------
    template : str
        Makefile template text

    Examples
    --------
    >>> from dstauffman.fortran import _get_template
    >>> template = _get_template("gfortran")

    """
    # fmt: off
    # default compiler flags and build settings
    if fcflags is None:
        fcflags             = {}
        fcflags["gfortran"] = "-O3 -ffree-form -ffree-line-length-none -fdefault-real-8 -std=f2018 -march=native"
        fcflags["ifort"]    = "-O3 -standard-semantics"
        fcflags["win"]      = "/O3 /free /extend-source:132 /real-size:64 /Qm64 /standard-semantics /define:SKIP_ASSERTS"
    if dbflags is None:
        dbflags             = {}
        dbflags["gfortran"] = "-Og -g -Wall -fimplicit-none -fcheck=all -fbacktrace -Wno-maybe-uninitialized"
        dbflags["ifort"]    = "-O0 -g -traceback -check bounds -check uninit -standard-semantics"
        dbflags["win"]      = "/Od /warn:all /traceback /check:bounds /check:uninit"
    preproc             = {}
    preproc["gfortran"] = "-cpp"
    preproc["ifort"]    = "-fpp"
    preproc["win"]      = "/fpp"
    mods             = {}
    mods["gfortran"] = r"-J$(OBJDIR) -I$(OBJDIR)"
    mods["ifort"]    = r"-module $(OBJDIR)"
    mods["win"]      = r"/module:$(B)"
    # fmt: on
    if not build:
        build = "debug" if is_debug else "release"
    this_fcflags = fcflags[compiler]
    if is_debug:
        this_dbflags = dbflags[compiler]
        this_fcflags = " ".join(this_fcflags.split(" ")[1:])
    else:
        this_dbflags = ""
    if use_preprocessor:
        this_fcflags += " " + preproc[compiler]
    # build Unix template
    # fmt: off
    if compiler != "win":
        template = (
            "# compiler and flags\n"
            + "FC      = " + (compiler if compiler != "win" else "ifort") + "\n"
            + "FCFLAGS = " + this_fcflags + "\n"
            + "DBFLAGS = " + this_dbflags + "\n"
            + "\n"
            + "# configuration\n"
            + "SRCDIR = source\n"
            + "OBJDIR = " + build + "/" + compiler + "\nOBJS   = \\\n" + r"""
# no implicit rules
.SUFFIXES:

# auxiliary programs
RM:=rm -f
MKDIR:=mkdir -p
TEST:=test -d

# create the build directory; define slashed version for convenience with short names
ifneq ($(OBJDIR),)
  $(shell $(TEST) $(OBJDIR) || $(MKDIR) $(OBJDIR))
  B:=$(OBJDIR)/
else
  B:=./
endif
ifneq ($(SRCDIR),)
  S:=$(SRCDIR)/
else
  S:=./
endif

# main executable

# object file implicit rules
$(B)%.obj : $(S)%.f90
""" + "\t$(FC) -c $(FCFLAGS) $(DBFLAGS) $(FPPFLAGS) " + mods[compiler] + r""" -o $@ $<

# object file dependencies

# clean-up
""" + ".PHONY : all clean " + program + r"""
clean :
""" + "\t$(RM) $(B)*.obj $(B)*.mod $(B)*.smod " + program + r""".exe
#""" + "\t" + r"""$(TEST) -d $(OBJDIR) && $(RM) -r $(OBJDIR)

"""
        )
    else:
        template = (
            "# compiler and flags\n"
            + "FC      = ifort\n"
            + "FCFLAGS = " + this_fcflags + "\n"
            + "DBFLAGS = " + this_dbflags + "\n"
            + "\n"
            + "# configuration\n"
            + "S      = source\n"
            + "B      = " + build + "\\" + compiler + "\nOBJS   = \\\n" + r"""
# no implicit rules
.SUFFIXES:

# main executable

# generic obj rule
.SUFFIXES: .f90
{$(S)}.f90{$(B)}.obj:
""" + "\t$(FC) /c $(FCFLAGS) $(DBFLAGS) $(FPPFLAGS) " + mods[compiler] + r""" /object:$@ $<

create_dirs:
""" + "\t@if not exist $(B) md $(B)" + r"""

# object file dependencies

# clean-up
clean :
""" + "\t" + r"""@del /f /q $(B)\*.obj $(B)\*.mod $(B)\*.smod """ + program + r""".exe
""" + "\t" + r"""@if exist $(B) rmdir $(B)

.PHONY : clean all create_dirs

"""
        )
    # fmt: on
    return template


# %% Functions - _write_makefile
def _write_makefile(
    makefile: Path,
    code: List[_FortranSource],
    *,
    template: Optional[str] = None,
    program: Optional[str] = None,
    compiler: str = "gfortran",
    is_debug: bool = False,
    sources: Optional[Iterable[str]] = None,
    external_sources: Optional[Iterable[str]] = None,
    replacements: Optional[Dict[str, str]] = None,
) -> None:
    r"""
    Reads the given makefile template and inserts the relevant rules based on the given source code.

    Parameters
    ----------
    makefile : class pathlib.Path
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
    >>> folder = get_root_dir().parent.parent.joinpath("forsat", "unit_tests")
    >>> makefile = folder / "unit_tests.make"
    >>> template = folder / "unit_tests_template.txt"
    >>> code = [] # TODO: write this line
    >>> _write_makefile(makefile, code=code, template=template) # doctest: +SKIP

    """

    def _build_dependencies(uses: Iterable[str]) -> List[str]:
        r"""Build dependencies with checks for external or intrinsics."""
        # create sorted list of uses that are not intrinsic or external
        sorted_uses = sorted(
            list({x for x in uses if x not in _INTRINSIC_MODS and (external_sources is None or x not in external_sources)})
        )
        # build the normal dependencies
        dependencies = []
        for x in sorted_uses:
            dependencies.append(prefix_bld + x + _OBJ_EXT)
        # append any external dependencies at the end, but still in sorted order
        if external_sources is not None:
            for x in sorted(external_sources):
                if x in uses:
                    dependencies.append(prefix_ext + x + _OBJ_EXT)
        return dependencies

    # hard-coded values
    token_src = "OBJS   = \\"
    token_run = "# main executable"
    token_obj = "# object file dependencies"
    len_line  = 200  # fmt: skip

    # optional inputs
    is_unit_test = program is None
    is_win = compiler == "win"
    if sources is None:
        sources = []
    if external_sources is None:
        external_sources = []
    lowercase_map = {x.lower(): x for x in sources}

    # prefixes
    prefix_bld = "$(B)"
    prefix_src = "" if is_unit_test else "$(S)"
    prefix_ext = "$(OBJLOC)/"
    prefix_obj = "$(B)\\" if is_win else ""
    if is_win:
        prefix_bld += "\\"
        prefix_src += "\\"

    # read the template into lines
    if template is None:
        assert program is not None
        template = _get_template(compiler=compiler, program=program, is_debug=is_debug)
    orig_lines = template.split("\n")

    # build the program rules
    if is_unit_test:
        run_rules = []
        runners = []
        for this_code in code:
            this_name = this_code.name
            if this_name.startswith("run_"):
                run_rules.append("")
                runners.append(this_name)
                this_rule = this_name + ".exe : " + this_name + ".f90 $(B)" + this_name + _OBJ_EXT
                run_rules.append(this_rule)
                this_depd = _build_dependencies(this_code.uses)
                # fmt: off
                if is_win:
                    this_rule = (
                        "\t$(FC) $(FCFLAGS) /exe:" + this_name + ".exe " + this_name
                        + ".f90 /module:$(B) " + " ".join(this_depd) + " /include:$(B) $(OBJS)"
                    )
                else:
                    this_rule = (
                        "\t$(FC) $(FCFLAGS) -o " + this_name + ".exe " + this_name
                        + ".f90 -I$(OBJDIR) -I$(OBJLOC) " + " ".join(this_depd) + " $(addprefix $(OBJLOC)/,$(OBJS))"
                    )
                # fmt: on
                if this_name == "run_all_tests":
                    this_rule = line_wrap(this_rule, wrap=len_line, indent=8, line_cont="\\")
                run_rules.append(this_rule)
    else:
        assert isinstance(program, str)  # for mypy
        run_rules = [""]
        run_rules.append(program + " : " + prefix_src + program + ".f90 " + prefix_bld + program + _OBJ_EXT)
        # fmt: off
        if is_win:
            run_rules.append(
                "\t$(FC) $(FCFLAGS) $(DBFLAGS) $(FPPFLAGS) /exe:" + program
                + ".exe " + prefix_src + program + ".f90 /include:$(B) $(OBJS)"
            )
        else:
            run_rules.append(
                "\t$(FC) $(FCFLAGS) $(DBFLAGS) $(FPPFLAGS) -o " + program + ".exe " + prefix_src
                + program + ".f90 -I$(OBJDIR) $(addprefix " + prefix_bld + ",$(OBJS))"
            )
        # fmt: on
    if is_unit_test:
        all_rule = "all : " + " ".join([x + ".exe" for x in runners])
    else:
        assert isinstance(program, str)  # for mypy
        all_rule = "all : " + ("create_dirs " if is_win else "") + program

    # build the object file rules
    obj_rules = [prefix_bld + "fruit" + _OBJ_EXT + " : fruit.f90", ""] if is_unit_test else []
    for this_code in code:
        this_name = lowercase_map.get(this_code.name, this_code.name)
        this_uses = sorted([lowercase_map.get(x, x) for x in this_code.uses], key=lambda x: x.lower())
        this_depd = _build_dependencies(this_uses)
        this_rule = prefix_bld + this_name + _OBJ_EXT + " : " + prefix_src + this_name + ".f90"
        if this_depd:
            this_rule += " " + " ".join(this_depd)
        if this_name == "run_all_tests":
            this_rule = line_wrap(this_rule, wrap=len_line, indent=8, line_cont="\\")
        obj_rules.append(this_rule)
        obj_rules.append("")
    # remove the extra newline at the very end
    obj_rules.pop()

    # Update the relevant sections of text
    new_lines = []
    for line in orig_lines:
        new_lines.append(line)
        if line == token_src:
            if is_unit_test:
                new_lines.extend(sorted(["       " + x + _OBJ_EXT + " \\" for x in external_sources], key=lambda x: x.lower()))
            else:
                new_lines.extend(
                    sorted(
                        ["       " + prefix_obj + x + _OBJ_EXT + " \\" for x in sources if x != program],
                        key=lambda x: x.lower(),
                    )
                )
        if line == token_run:
            new_lines.append(all_rule)
            new_lines.extend(run_rules)
        if line == token_obj:
            new_lines.extend(obj_rules)

    # write out to disk and print updates
    text = "\n".join(new_lines)
    # Replace any desired string tokens
    if replacements is not None:
        for key, value in replacements.items():
            text = text.replace(key, value)
    print(f'Writing "{makefile}".')
    write_text_file(makefile, text)


# %% Functions - create_fortran_unit_tests
def create_fortran_unit_tests(
    folder: Path,
    *,
    template: Optional[str] = None,
    external_sources: Optional[Iterable[str]] = None,
    header: Optional[str] = None,
) -> None:
    r"""
    Parses the given folder for Fortran unit test files to build programs that will execute them.

    Parameters
    ----------
    folder : class pathlib.Path
        Folder location to look for unit tests
    template : str, optional
        Template to use for the makefile
    external_sources : set of str, optional
        Files that are assumed to already exist and don't need to be built by the makefile
    header : str, optional
        If specified, write this additional header information

    Returns
    -------
    (None) - creates run_*.f90 files in the same folder location

    Examples
    --------
    >>> from dstauffman import create_fortran_unit_tests, get_root_dir
    >>> folder = get_root_dir().parent.parent.joinpath("forsat", "unit_tests")
    >>> create_fortran_unit_tests(folder) # doctest: +SKIP

    """
    # find all the files to process
    files = folder.glob("test*.f90")
    # initialize code information
    all_code = []
    # process each file
    for file in files:
        # parse the source code
        code = _parse_source(file, assert_single=True)
        newfile = folder.joinpath("run_" + file.name)

        # build the individual unit test
        _write_unit_test(newfile, code, header)
        # save this code for information for the makefile
        all_code.append(code)

    # write run_all_tests file
    _write_all_unit_test(folder / "run_all_tests.f90", all_code, header)

    # Re-parse the source code once all the new files have been written
    files = folder.glob("run_*.f90")
    for file in files:
        code = _parse_source(file, assert_single=True)
        all_code.append(code)

    # Sort the parsed code
    all_code.sort(key=lambda x: x.name)

    # write the master Makefile
    if template is not None:
        makefile = folder / "unit_tests.make"
        _write_makefile(makefile, code=all_code, template=template, external_sources=external_sources)


# %% create_fortran_makefile
def create_fortran_makefile(
    folder: Path,
    makefile: Path,
    program: str,
    sources: List[str],
    *,
    compiler: str = "gfortran",
    is_debug: bool = True,
    template: Optional[str] = None,
    replacements: Optional[Dict[str, str]] = None,
) -> None:
    r"""
    Parses the given folder for Fortran source files to build a makefile.

    Parameters
    ----------
    folder : class pathlib.Path
        Folder location to look for unit tests
    makefile : class pathlib.Path
        Location of the makefile to create
    template : str
        Template to use for the makefile
    program : str
        Name of the Fortran main program
    sources : list of str
        List of names of the source files
    compiler : str, optional
        Name of the compiler to build with, default is gfortran
    is_debug : bool, optional
        Whether to build the debug version of the executable, default is True

    Returns
    -------
    (None) - creates the specified make file

    Examples
    --------
    >>> from dstauffman import create_fortran_makefile, get_root_dir
    >>> folder = get_root_dir().parent.parent.joinpath("forsat", "source")
    >>> makefile = folder.parent.joinpath("Makefile")
    >>> template = "" # TODO: fill this in
    >>> program = "forsat"
    >>> sources = [] # TODO: populate this
    >>> create_fortran_makefile(folder, makefile, template, program, sources) # doctest: +SKIP

    """
    # initialize code information
    all_code = []
    # process each file
    for file in sources:
        code = _parse_source(folder.joinpath(file + ".f90"), assert_single=True)
        all_code.append(code)

    # write makefile
    _write_makefile(
        makefile,
        all_code,
        program=program,
        compiler=compiler,
        is_debug=is_debug,
        template=template,
        sources=sources,
        replacements=replacements,
    )


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_fortran", exit=False)
    doctest.testmod(verbose=False)
