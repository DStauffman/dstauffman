set mod_name=dstauffman
set test_name=knight

cd C:\Users\%username%\Documents\GitHub\%mod_name%\games

coverage run --rcfile=C:\Users\%username%\Documents\GitHub\%mod_name%\tests\.coveragerc test_%test_name%.py
coverage html --rcfile=C:\Users\%username%\Documents\GitHub\%mod_name%\tests\.coveragerc test_%test_name%.py
start C:\Users\%username%\Documents\GitHub\%mod_name%\games\coverage_html_report\index.html

echo 'Press any key to continue'
pause
