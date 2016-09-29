# dstauffman

The "dstauffman" module is a generic Python code library of functions that I (David C. Stauffer) have found useful.

Written by David C. Stauffer in March 2015.

## Library dependencies

This code tends to push on the leading edge of the Python development cycle.  For example, I do a lot of linear algebra and numerical analysis, and I love the infix matrix multiplier (@) that was introduced in v3.5.  There is no way to make that a backwards compatible, so you must be on Python v3.5 or newer.

The new f-strings in Python v3.6 look amazing, so I plan to upgrade to those shortly after the official release of v3.6.

I do lots of plotting with matplotlib, and additionally use the PyQt5 library for GUIs and adding buttons (like next plot, previous plot and close all) to the standard MPL toolbar.  This code is backend specific, and Qt is much more powerful than Tk/Tcl, so I use PyQt5 instead of the core tkinter.  I currently maintain backwards compatibility with PyQt4 using the newer signal and slots methods, but plan on dropping that when PyQt5 is more stable in WinPython or Anaconda.  I'm satisfied with PyQt5 on Ubuntu 16.04, even though it's still more difficult to install (as of September 2016).

### Built-in libraries

The following built-in Python libraries are used within the dstauffman library.

* collections
* contextlib
* copy
* datetime
* doctest
* enum
* getpass
* glob
* inspect
* io
* logging
* math
* nose
* os
* platform
* pickle
* random
* shutil
* sys
* timeit
* types
* unittest

### Additional libraries

The following non-standard, but for the most part very well known libraries, are also used by the dstauffman library.  Of all of these, PyQt5 is by far the one most likely to cause you issues.  (It uses `python configure.py` instead of `python setup.py`, so it fails under pip or easy_install.  It also depends on SIP, which has the same limitations, and once install matplotlib has to be told to use the same matching backend.

* h5py
* matplotlib
* mpl_toolkits<sup>[1](#myfootnote1)</sup>
* numpy
* pandas
* PIL<sup>[1](#myfootnote1)</sup>
* PyQt5 (or PyQt4)
* pprofile<sup>[1](#myfootnote1)</sup>
* scipy.linalg

<a name="myfootnote1">1</a>: Not required to do `import dstauffman`, but used by some of the optional submodules.

### Additional libraries only used if trying to compile to C with Cython
* pyximport
* distutils.core
* Cython.Build

## Installing on Windows with WinPython
WinPython comes with almost all the libraries you will need.  I believe the only one it doesn't currently include is pprofile.  This library is pure python and easily install with pip.

```
pip install pprofile
```

## Installing on Windows with Anaconda (Anaconda is also available on Unix/Mac)
Ananconda also comes with almost all the libraries you need, plus it has virtual environment support built-in, so that's nice.  The only difference here is you need to use conda instead of pip for new libraries.

```
conda install pprofile
```

## Installing within a virtual environment in Ubuntu 16.04

TODO: finish writing this

## Configuring matplotlib to use PyQt5 (or PyQt4)

When matplotlib is imported, it has to use a graphics backend, and once set, it usually is not possible to change on the fly.  So you have to either do `import matplotlib; matplotlib.use('Qt5Agg')` as the very first import every single time, or the much better solution is to configure the matplotlibrc file to use Qt.

First, find the matplotlibrc file location, or if it doesn't exist, then create one from a template found online.  On a Windows installation of WinPython, this might be somewhere like:

`C:\Programs\WinPython-64bit-3.4.2.4\python-3.4.2.amd64\Lib\site-packages\matplotlib\mpl-data\matplotlibrc`

Change the line with the backend option to:

`backend : Qt5Agg`

Add a line with:

`backend.qt5: PyQt5`

If you are using PyQt4 instead, then these lines should be:

```
backend      : Qt4Agg
backend.qt4  : PyQt4
```
