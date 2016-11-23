# dstauffman

The "dstauffman" module is a generic Python code library of functions that I (David C. Stauffer) have found useful.

Written by David C. Stauffer in March 2015.

## Library dependencies

This code tends to push on the leading edge of the Python development cycle.  For example, I do a lot of linear algebra and numerical analysis, and I love the infix matrix multiplier (@) that was introduced in v3.5.  There is no way to make that a backwards compatible, so you must be on Python v3.5 or newer.

The new f-strings in Python v3.6 look amazing, so I plan to upgrade to those shortly after the official release of v3.6.

I do lots of plotting with matplotlib, and additionally use the PyQt5 library for GUIs and adding buttons (like next plot, previous plot and close all) to the standard MPL toolbar.  This code is backend specific, and Qt is much more powerful than Tk/Tcl, so I use PyQt5 instead of the core tkinter.

### Built-in libraries

The following built-in Python libraries are used within the dstauffman library.

* collections
* contextlib
* copy
* datetime
* doctest
* enum
* inspect
* io
* os
* platform
* pickle
* shutil
* sys
* types
* unittest

### Additional libraries

The following non-standard, but for the most part very well known libraries, are also used by the dstauffman library.  Of all of these, PyQt5 is by far the one most likely to cause you issues.  (It uses `python configure.py` instead of `python setup.py`, so it fails under pip or easy_install.  It also depends on SIP, which has the same limitations, and once install matplotlib has to be told to use the same matching backend.

* h5py
* matplotlib
* numpy
* pandas
* PyQt5
* scipy.linalg

## Installing on Windows with WinPython or Anaconda.
WinPython or Anaconda come with all the libraries you will need.  I highly recommend using one of them.

## Installing within a virtual environment in Ubuntu 16.04

TODO: finish writing this

## Configuring matplotlib to use PyQt5

When matplotlib is imported, it has to use a graphics backend, and once set, it usually is not possible to change on the fly.  So you have to either do `import matplotlib; matplotlib.use('Qt5Agg')` as the very first import every single time, or the much better solution is to configure the matplotlibrc file to use Qt.

First, find the matplotlibrc file location, or if it doesn't exist, then create one from a template found online.  On a Windows installation of WinPython, this might be somewhere like:

`C:\Programs\WinPython-64bit-3.5.2.3Qt5\python-3.5.2.amd64\Lib\site-packages\matplotlib\mpl-data\matplotlibrc`

Change the line with the backend option to:

`backend : Qt5Agg`

Add a line with:

`backend.qt5: PyQt5`
