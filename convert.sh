#!/usr/bin/env sh
set -x
./convert_gct.py
fontforge -lang=ff -c 'Open($1); Generate($2)' GCTGothicEnglish.sfd GCTGothicEnglish.ttf
