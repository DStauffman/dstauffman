#!/bin/bash

# Get the directory that contains this script
THISDIR=${BASH_SOURCE%/*}
#THISDIR=${THISDIR:-"$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"}

if grep -i -q windows <<< "$OS"; then
    # Windows - (presumably via Git-Bash)
    # Pass through to dcspython.bat
    $THISDIR/dcspython.bat ${@}
else
    # Presumably Linux
    # Setup PYTHONPATH
    # Remember the original python path
    PYTHONPATH_ORIG=$PYTHONPATH

    # Prepend THISDIR to PYTHONPATH
    export PYTHONPATH="$THISDIR:$PYTHONPATH"
    
    if command -v python3 &>/dev/null; then
        # Try using python3
        exec python3 ${@}
    else
        # If that didn't work, then throw error (as Python v2 is not supported?)
        echo "ERROR: dstauffman requires python3"
        # exec python ${@}
    fi
    
    # CLEANUP
    export PYTHONPATH=$PYTHONPATH_ORIG
fi
