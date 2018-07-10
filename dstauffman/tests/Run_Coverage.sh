# Get the directory where this script is located
dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
# change to the script directory
cd $dir
# run the coverage tests
coverage run run_all_tests.py
# generate the coverage report
coverage html
# open the report
open coverage_html_report/index.html
# pause execution so the user can see the results
read -n1 -r -p "Press any key to continue..."
