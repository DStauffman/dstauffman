#!/bin/bash

# Get the directory that contains this script
THISDIR=${BASH_SOURCE%/*}
#THISDIR=${THISDIR:-"$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"}

if grep -i -q windows <<< "$OS"; then
    # Windows - (presumably via Git-Bash)
    # Pass through to dcspypy.bat
    $THISDIR/dcspypy.bat ${@}
else
    # Presumably Linux
    # Setup PYTHONPATH, first remember the original python path
    PYTHONPATH_ORIG=$PYTHONPATH

    # Add this location
    export PYTHONPATH="$THISDIR/..:$THISDIR/../../nubs:$THISDIR/../../slog:$THISDIR/../../dstauffman2:$PYTHONPATH"

    if command -v pypy3 &>/dev/null; then
        # Try using pypy3
        exec pypy3 ${@}
    else
        # If that didn't work, then try just pypy
        exec pypy ${@}
    fi

    # CLEANUP
    export PYTHONPATH=$PYTHONPATH_ORIG
fi
