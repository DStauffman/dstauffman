@echo off

REM %~dp0 returns the directory of this batch file. We use this to anchor our paths to the location of the script
SET scripts_home1=%~dp0
SET scripts_home=%scripts_home1%

REM Remove the final slash if it exists to avoid double slashes in the added paths
IF #%scripts_home1:~-1%# == #\# SET scripts_home=%scripts_home1:~0,-1%

REM Save the original paths
SET PATH_ORIG=%PATH%
SET PYTHONPATH_ORIG=%PYTHONPATH%

REM Modify as needed, currently only modifying the python path
SET PYTHONPATH=%scripts_home%;%PYTHONPATH%
REM TODO: if you wanted to add a custom python path or include one within dcs, then do it here, otherwise assume it exists

"python.exe" %*

:cleanup

REM Set the path back to orignals
SET PATH=%PATH_ORIG%
SET PYTHONPATH=%PYHONPATH_ORIG%
