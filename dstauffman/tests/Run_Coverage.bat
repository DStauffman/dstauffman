cd %~dp0
coverage run run_all_tests.py
coverage html
start coverage_html_report\index.html

echo 'Press any key to continue'
pause
