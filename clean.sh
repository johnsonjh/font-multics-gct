#!/usr/bin/env sh
# SPDX-License-Identifier: Multics or MIT-0
# Copyright (c) 2025 Jeffrey H. Johnson
# Copyright (c) 2025 The DPS8M Development Team
# scspell-id: 58d56a76-9924-11f0-9b08-80ee73e9b8e7

# shellcheck disable=SC2006,SC2046,SC2065,SC2116
test _`echo asdf 2>/dev/null` != _asdf >/dev/null &&\
  printf '%s\n' "FATAL: Using csh as sh is not supported." &&\
  exit 1

"${RM:-rm}" -f ./a.out ./errnum ./*.sfd ./*.ttf ./*.log ./core-* ./*.core ./core
