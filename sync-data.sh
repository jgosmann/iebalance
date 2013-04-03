#!/bin/zsh

p="${0:h}"
remote="$p/remote"
archive="$p/Archive"
data="$p/Data"

if [[ -z `ls $remote` ]]; then
    sshfs compute3:/extra/gosmann "$remote"
fi

rsync -rtuvz "$remote/iebalance/Archive" "$p"
for file in `ls $archive`; do
    tar -xvz -f "$archive/$file" --keep-newer-files --strip-components 1 -C "$data"
done
