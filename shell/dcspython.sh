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
    # Setup PYTHONPATH, first remember the original python path
    PYTHONPATH_ORIG=$PYTHONPATH

    # Add this location
    export PYTHONPATH="$THISDIR/..:$THISDIR/../../slog:$THISDIR/../../dstauffman2:$PYTHONPATH"

    if command -v python3 &>/dev/null; then
        # Try using python3
        exec python3 ${@}
    else
        # If that didn't work, then try just python
        exec python ${@}
    fi

    # CLEANUP
    export PYTHONPATH=$PYTHONPATH_ORIG
fi
