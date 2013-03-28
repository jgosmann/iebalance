#!/bin/zsh
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <file to run> <label> [additional arguments ...]" >&2
    exit -1
fi

mainfile=$1
label=$2
timestamp=`date +%Y%m%d-%H%M%S`
prefix=${mainfile:t:r}

reasonfile=`mktemp reason.XXXXXX`
${EDITOR:-vi} "$reasonfile"
reason=`cat "$reasonfile"`
rm "$reasonfile"

if [[ -z "$reason" ]]; then
    echo "Aborting because of empty 'reason' message." >&2
    exit -1
fi

python='python'
if [[ -f '.python.cmd' ]]; then
    python=`cat .python.cmd`
fi

shift 2
smt configure -l "$label"
smt run -r "$reason" -e "$python" \
    -l "${prefix}-${label}-${timestamp}" -m "$mainfile" -- $*
