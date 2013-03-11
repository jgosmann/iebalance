#!/bin/zsh
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <file to run> [additional arguments ...]" >&2
    exit -1
fi

mainfile=$1
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

shift
smt run -r "$reason" \
    -l "${prefix}-${timestamp}" -m "$mainfile" -- $*
