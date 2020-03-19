cd %~dp0
coverage run -m dstauffman tests
coverage html
start coverage_html_report\index.html

pause
