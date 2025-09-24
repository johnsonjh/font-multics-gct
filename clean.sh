#!/usr/bin/env sh

if [ "Linux" = "$(uname -s 2> /dev/null || :)" ]; then
  # shellcheck disable=SC2015
  command -v flock > /dev/null 2>&1 && {
    # shellcheck disable=SC2015,SC2154
    [ "${FLOCKER}" != "$0" ] && exec env FLOCKER="$0" flock -E 14 -en "$0" "$0" "$@" || :
  } || :
fi

rm -f ./a.out ./errnum ./*.sfd ./*.ttf ./*.log ./core-* ./*.core ./core ./clean ./convert
