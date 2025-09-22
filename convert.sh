#!/usr/bin/env sh
set -x

for gct_file in gct_*_; do
  if [ -f "$gct_file" ]; then
    base_name=$(echo "$gct_file" | sed 's/gct_//' | sed 's/_$//')
    sfd_name="GCT$(echo "$base_name" | awk -F_ '{for(i=1;i<=NF;i++) printf "%s", toupper(substr($i,1,1)) substr($i,2)}')"
    ./convert_gct.py "$gct_file" "${sfd_name}.sfd"
    fontforge -lang=ff -c 'Open($1); Generate($2)' "${sfd_name}.sfd" "${sfd_name}.ttf"
  fi
done
