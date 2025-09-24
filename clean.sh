#!/usr/bin/env sh
# SPDX-License-Identifier: Multics or MIT-0
# Copyright (c) 2025 Jeffrey H. Johnson
# Copyright (c) 2025 The DPS8M Development Team
# scspell-id: 58d56a76-9924-11f0-9b08-80ee73e9b8e7

if [ "Linux" = "$(uname -s 2> /dev/null || :)" ]; then
  # shellcheck disable=SC2015
  command -v flock > /dev/null 2>&1 && {
    # shellcheck disable=SC2015,SC2154
    [ "${FLOCKER}" != "$0" ] && exec env FLOCKER="$0" flock -E 14 -en "$0" "$0" "$@" || :
  } || :
fi

rm -f ./a.out ./errnum ./*.sfd ./*.ttf ./*.log ./core-* ./*.core ./core
