@echo off

REM %~dp0 returns the directory of this batch file. We use this to anchor our paths to the location of the script
SET tests_home1=%~dp0
SET tests_home=%tests_home1%
SET envs_home=C:\Users\%username%\Documents\venvs

REM Remove the final slash if it exists to avoid double slashes in the added paths
IF #%tests_home1:~-1%# == #\# SET tests_home=%tests_home1:~0,-1%

REM Run the full coverage (without numba/datashader)
call %envs_home%\static\Scripts\activate
call dcs tests --coverage --cov_file %tests_home%\.coverage.no_numba
call deactivate

REM Run with numba/datashader
call %envs_home%\everything\Scripts\activate
call dcs tests --coverage --cov_file %tests_home%\.coverage.full
call deactivate

REM Run with only core Python
call %envs_home%\core_only\Scripts\activate
call dcs tests --coverage --cov_file %tests_home%\.coverage.core

REM Re-run the command help tests to make sure this gets code coverage
SET COVERAGE_FILE=%tests_home%\.coverage.commands
SET COVERAGE_RCFILE=%tests_home%\.coveragerc
SET PYTHONPATH=%tests_home%\..\..;%PYTHONPATH%
call coverage run %tests_home%\test_commands_help.py

cd %tests_home%
SET COVERAGE_FILE=.coverage
SET COVERAGE_RCFILE=.coveragerc
call coverage combine --keep
call coverage html
call coverage xml
call deactivate

REM Open the report
start coverage_html_report\index.html

echo Press any key to continue . . .
pause >nul
