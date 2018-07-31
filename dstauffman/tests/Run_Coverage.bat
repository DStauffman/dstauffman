cd %~dp0
coverage run run_all_tests.py
coverage html
start coverage_html_report\index.html

pause
