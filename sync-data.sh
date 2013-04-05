#!/bin/zsh

p="${0:h}"
remote="$p/remote"
archive="$p/Archive"
data="$p/Data"

if [[ -z `ls $remote` ]]; then
    sshfs compute3:/extra/gosmann "$remote"
fi

files=`rsync -rtuvz "$remote/iebalance/Archive" "$p"`
for file in "(${(ps:\n:)${files}})"; do
    if [[ -f "$file" ]]; then
        tar -xvz -f "$file" --keep-newer-files --strip-components 1 -C "$data"
    fi
done
