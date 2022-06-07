@echo off

REM %~dp0 returns the directory of this batch file. We use this to anchor our paths to the location of the script
SET tests_home1=%~dp0
SET tests_home=%tests_home1%

REM Remove the final slash if it exists to avoid double slashes in the added paths
IF #%tests_home1:~-1%# == #\# SET tests_home=%tests_home1:~0,-1%

REM Save the original paths
SET PYTHONPATH_ORIG=%PYTHONPATH%

REM Modify as needed, currently only modifying the python path
SET PYTHONPATH=%tests_home%\..\..;%PYTHONPATH%

REM Run the coverage tool and generate the report
REM TODO: this first option doesn't collect the statistics correctly
REM call coverage run -m dstauffman tests
call coverage run %tests_home%\run_all_tests.py  --rcfile %tests_home%\..\..\pyproject.toml
call coverage html --rcfile %tests_home%\..\..\pyproject.toml

REM Set the path back to orignals
SET PYTHONPATH=%PYHONPATH_ORIG%

REM Open the report
start %tests_home%\coverage_html_report\index.html

echo Press any key to continue . . .
pause >nul
