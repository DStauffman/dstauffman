#!/bin/bash

# Get the directory that contains this script
THISDIR=${BASH_SOURCE%/*}
#THISDIR=${THISDIR:-"$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"}

if grep -i -q windows <<< "$OS"; then
    # Windows - (presumably via Git-Bash)
    # Pass through to dcspython.bat
    $THISDIR/dcsipython.bat ${@}
else
    # Presumably Linux
    # Setup PYTHONPATH, first remember the original python path
    PYTHONPATH_ORIG=$PYTHONPATH

    # Add this location
    export PYTHONPATH="$THISDIR/..:$THISDIR/../../nubs:$THISDIR/../../slog:$THISDIR/../../dstauffman2:$PYTHONPATH"

    if command -v ipython3 &>/dev/null; then
        # Try using python3
        exec ipython3 ${@}
    else
        # If that didn't work, then try just python
        exec ipython ${@}
    fi

    # CLEANUP
    export PYTHONPATH=$PYTHONPATH_ORIG
fi
