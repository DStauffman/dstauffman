# Get the directory where this script is located
dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
# change to the script directory
cd $dir
# save the original paths
PYTHONPATH_ORIG=$PYTHONPATH
# modify as needed
PYTHONPATH=$dir/../..:$PYTHONPATH
# run the coverage tests
#coverage run -m dstauffman tests
coverage run $dir/run_all_tests.py
# restore the paths
PYTHONPATH=$PYTHONPATH_ORIG
# generate the coverage report
coverage html
# open the report
xdg-open ./coverage_html_report/index.html
# pause execution so the user can see the results
# read -n1 -r -p "Press any key to continue..."
