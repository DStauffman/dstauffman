@echo off

REM %~dp0 returns the directory of this batch file. We use this to anchor our paths to the location of the script
SET scripts_home1=%~dp0
SET scripts_home=%scripts_home1%

REM Remove the final slash if it exists to avoid double slashes in the added paths
IF #%scripts_home1:~-1%# == #\# SET scripts_home=%scripts_home1:~0,-1%

REM Set other paths
SET tests_home=%scripts_home%\..\dstauffman\tests
SET envs_home=C:\Users\%username%\Documents\venvs

REM Run with only core Python
CALL %envs_home%\core_only\Scripts\activate
CALL dcs tests --coverage --cov_file %scripts_home%\.coverage.core
CALL dcs tests --library %scripts_home%\..\nubs --coverage --cov_file %scripts_home%\.coverage.nubs_core
CALL dcs tests --library %scripts_home%\..\..\slog --coverage --cov_file %scripts_home%\.coverage.slog_core
CALL deactivate

REM Run without numba/datashader (as JIT can confuse things)
CALL %envs_home%\static\Scripts\activate
CALL dcs tests --coverage --cov_file %scripts_home%\.coverage.no_numba
REM CALL dcs tests --library %scripts_home%\..\nubs --coverage --cov_file %scripts_home%\.coverage.nubs_static
CALL deactivate

REM Run with numba/datashader
CALL %envs_home%\everything\Scripts\activate
CALL dcs tests --coverage --cov_file %scripts_home%\.coverage.full
CALL dcs tests --library %scripts_home%\..\nubs --coverage --cov_file %scripts_home%\.coverage.nubs_full
CALL deactivate

REM Re-run some tests to measure root import statements
SET COVERAGE_FILE=%scripts_home%\.coverage.imports
SET COVERAGE_RCFILE=%scripts_home%\..\pyproject.toml
SET PYTHONPATH=%scripts_home%\..;%scripts_home%\..\..\slog;%PYTHONPATH%
CALL %envs_home%\static\Scripts\activate
CALL coverage run %tests_home%\test_version.py
CALL deactivate
CALL %envs_home%\everything\Scripts\activate
CALL coverage run --append %tests_home%\test_version.py
CALL deactivate
CALL %envs_home%\core_only\Scripts\activate
CALL coverage run --append %tests_home%\test_commands_help.py

REM Combine results
cd %scripts_home%
SET COVERAGE_FILE=%scripts_home%\.coverage
CALL coverage combine --keep
CALL coverage html
CALL coverage xml
CALL deactivate

REM Open the report
start %scripts_home%\coverage_html_report\index.html

echo Press any key to continue . . .
pause >nul
