@echo off

REM %~dp0 returns the directory of this batch file. We use this to anchor our paths to the location of the script
SET tests_home1=%~dp0
SET tests_home=%tests_home1%
SET envs_home=C:\Users\%username%\Documents\venvs

REM Remove the final slash if it exists to avoid double slashes in the added paths
IF #%tests_home1:~-1%# == #\# SET tests_home=%tests_home1:~0,-1%

REM Run with only core Python
CALL %envs_home%\core_only\Scripts\activate
CALL dcs tests --coverage --cov_file %tests_home%\.coverage.core
CALL deactivate

REM Run without numba/datashader (as JIT can confuse things)
CALL %envs_home%\static\Scripts\activate
CALL dcs tests --coverage --cov_file %tests_home%\.coverage.no_numba
CALL deactivate

REM Run with numba/datashader
CALL %envs_home%\everything\Scripts\activate
CALL dcs tests --coverage --cov_file %tests_home%\.coverage.full
CALL deactivate

REM Re-run some tests to measure root import statements
SET COVERAGE_FILE=%tests_home%\.coverage.imports
SET COVERAGE_RCFILE=%tests_home%\..\..\pyproject.toml
SET PYTHONPATH=%tests_home%\..\..;%PYTHONPATH%
CALL %envs_home%\static\Scripts\activate
CALL coverage run %tests_home%\test_version.py
CALL deactivate
CALL %envs_home%\everything\Scripts\activate
CALL coverage run --append %tests_home%\test_version.py
CALL deactivate
CALL %envs_home%\core_only\Scripts\activate
CALL coverage run --append %tests_home%\test_commands_help.py

REM Combine results
cd %tests_home%
SET COVERAGE_FILE=%tests_home%\.coverage
CALL coverage combine --keep
CALL coverage html
CALL coverage xml
CALL deactivate

REM Open the report
start %tests_home%\coverage_html_report\index.html

echo Press any key to continue . . .
pause >nul
