set mod_name=dstauffman
set test_name=knight

cd C:\Users\%username%\Documents\GitHub\%mod_name%\games

coverage run test_%test_name%.py
coverage html
start C:\Users\%username%\Documents\GitHub\%mod_name%\games\coverage_html_report\index.html

REM echo 'Press any key to continue'
REM pause
