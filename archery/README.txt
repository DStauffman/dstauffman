# archery
The "archery" module is a collection of code to support tournament scoring, bracket generation and field layout.

The module is broken up into multiple files.
.__init__.py  includes the relative import statements to be executed when someone does "import archery"
.bales.py     includes the logic to assign bales based on the list of registered archers.
.brackets.py  includes the logic to generate brackets based on individual seed order.
.constants.py includes global constants that don't change during execution.
.main.py      includes nothing right now.  It might later include more of a wrapper function.
.pretty.py    includes the functions that create files for user output in a "pretty" format.
.utils.py     includes general utilities that are used by the different pieces of the code.


The master data is defined in an excel spreadsheet.  Details TBD...

Originally developed in January 2015 by David C. Stauffer for use by Stanford University.  It was then continued
as a fun coding project and a way to learn more about Python programming.  This code is intended to be open
source and  freely distributed.  Hopefully it will grow to be useful for other people and other tournaments.
