cd C:\Users\%username%\Documents\GitHub\dstauffman\apps\bac_gui

coverage run --rcfile=C:\Users\%username%\Documents\GitHub\dstauffman\tests\.coveragerc test_bac_gui.py
coverage html --rcfile=C:\Users\%username%\Documents\GitHub\dstauffman\tests\.coveragerc 
start C:\Users\%username%\Documents\GitHub\dstauffman\apps\bac_gui\coverage_html_report\index.html

echo 'Press any key to continue'
pause
