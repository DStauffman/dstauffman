##########
dstauffman
##########

The "dstauffman" module is a generic Python code library of functions that I (David C. Stauffer) have found useful.

Written by David C. Stauffer in March 2015.


********************
Library dependencies
********************

This code relies on newer features within the Python language.  These include enums from v3.4, the '@' operator for matrix multiplication from v3.5, ordered dictionaries and f-strings from v3.6, and assignment expressions ":=" from v3.8, and eventually newer typing methods from v3.9.  As such, it currently requires at least v3.8 of Python and will not run on v2.7.

I do lots of plotting with matplotlib, and additionally use the PyQt5 library for GUIs and adding buttons (like next plot, previous plot and close all) to the standard MPL toolbar.  This code is backend specific, and Qt is much more powerful than Tk/Tcl, so I use PyQt5 instead of the core tkinter.

Built-in libraries
******************

The following built-in Python libraries are used within the dstauffman library.

* argparse
* collections
* contextlib
* copy
* datetime
* doctest
* enum
* gc
* glob
* inspect
* io
* logging
* os
* platform
* pickle
* re
* shlex
* subprocess
* sys
* time
* types
* unittest
* warnings

Additional libraries
********************

The following non-standard, but for the most part very well known libraries, are also used by the dstauffman library.  Of all of these, PyQt5 is by far the one most likely to cause you issues.  (It uses ``python configure.py`` instead of ``python setup.py``, so it fails under pip or easy_install.  It also depends on SIP, which has the same limitations, and once installed matplotlib has to be told to use the same matching backend.

* h5py
* matplotlib
* numpy
* pandas (minimal usage)
* PyQt5
* pytest
* scipy (minimal usage)
* tblib (minimal usage)


************
Installation
************

Installing on Windows with WinPython or Anaconda
************************************************

WinPython or Anaconda come with all the libraries you will need.  I highly recommend using one of them.  Just download and run their respective installers.


*************
Configuration
*************

Configuring matplotlib to use PyQt5
***********************************

When matplotlib is imported, it has to use a graphics backend, and once set, it usually is not possible to change on the fly without restarting the application.  So you have to either do `import matplotlib; matplotlib.use('Qt5Agg')` as the very first import every single time, or the much better solution is to configure the matplotlibrc file to use Qt.  Newer versions of Anaconda often default to this, and if you are using the built-in Spyder IDE, then there is a menu based preference to choose it.

First, find the matplotlibrc file location, or if it doesn't exist, then create one from a template found online.

.. code-block:: python

    import matplotlib as mpl
    mpl.matplotlib_fname()

Change the line with the backend option to:

``backend : Qt5Agg``

Preparing Python
****************

The code is designed to be imported as a library. In order for that to happen, the "dstauffman" folder must be on either your system path or python path.

The recommended method is to modify your user "PYTHONPATH" variable. On Windows 10, you can do this by hitting start, and then starting to type "environment" and choosing the "Edit environment variables for your account" within the control panel.  Then if the user variable for "PYTHONPATH" (one word, all caps) doesn't exist, create a new one. If it does, append to it. On Windows use a semi-colon (;) to separate folders, and on Unix, use a colon (:) and don't put any spaces between folders. Add the folder location that contains the "dstauffman subfolder" folder. In my case, that's the GitHub folder where I keep my local copy of the repository.

Running the Code
****************

At least one example script should be available in the ./dstauffman/scripts folder. This script can be run via a command prompt:

.. code-block:: python

    python script_name.py

If you are on Windows and installed Anaconda as described earlier, then python may not be on your system path, and you'll likely need to launch the Anaconda Prompt instead.

If you want to be able to interact with the results or the plots, then the better way to run the script is by opening it within Spyder and running it in that application using the IPython console.


**********************
Command Line Interface
**********************

In addition to import the code as a library, some functionality is available through the command line, via a script called "dcs".  (In reality, it still just imports the library under the hood and passes the argument on).

For any of the given commands, you can get more information with a '-h' or '--help' option.

The following commands are available:

* coverage
* enforce
* help
* make_init
* tests
