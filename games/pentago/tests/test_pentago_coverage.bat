cd C:\Users\%username%\Documents\GitHub\dstauffman\games\pentago\tests

coverage run  --rcfile=C:\Users\%username%\Documents\GitHub\dstauffman\tests\.coveragerc run_all_tests.py
coverage html --rcfile=C:\Users\%username%\Documents\GitHub\dstauffman\tests\.coveragerc
start C:\Users\%username%\Documents\GitHub\dstauffman\games\pentago\tests\coverage_html_report\index.html

echo 'Press any key to continue'
pause
