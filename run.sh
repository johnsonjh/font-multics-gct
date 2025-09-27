#!/usr/bin/env sh
# SPDX-License-Identifier: Multics or MIT-0
# Copyright (c) 2025 Jeffrey H. Johnson
# Copyright (c) 2025 The DPS8M Development Team
# scspell-id: aa0756ca-9924-11f0-bb6f-80ee73e9b8e7

# shellcheck disable=SC2006,SC2046,SC2065,SC2116
test _`echo asdf 2>/dev/null` != _asdf >/dev/null &&\
  printf '%s\n' "FATAL: Using csh as sh is not supported." &&\
  exit 1

export POSIXLY_CORRECT=1
export TZ=UTC

# shellcheck disable=SC3045
ulimit -c 0 > /dev/null 2>&1

check_for()
{
  for program; do
    if ! command -v "${program:?}" > /dev/null 2>&1; then
      printf 'FATAL: %s\n' "${program:?} not found" >&2
      exit 1
    fi
  done
}

check_for "${AWK:-awk}" "${CC:-cc}" "${DATE:-date}" "${GREP:-grep}" "${RM:-rm}" "${SED:-sed}" "${FONTFORGE:-fontforge}" "python3"

if [ "$1" != "ttf" ]; then
  test -x ./clean.sh && {
    ./clean.sh || :
  }
fi

test -x ./errnum || {
  "${CC:-cc}" -o errnum errnum.c || {
    printf 'FATAL: %s\n' "errnum.c compilation failed" >&2
    exit 1
  }
}

if [ "$1" = "ttf" ]; then
  for sfd_file in GCT*.sfd; do
    case "${sfd_file:?}" in
      *Stroke*) continue ;;
      "GCT*.sfd") { printf 'FATAL: %s\n' "No valid 'GCT*.sfd' files found" >&2; exit 1; } ;;
      *) : ;;
    esac
    sfd_name="${sfd_file%.*}"
    printf 'Converting %s: \t%s' "${sfd_name:?}" "TrueType: "; printf '%s\n' "" "TrueType: $(date 2> /dev/null || :)" \
      "================================================================================" "" >> "${sfd_name:?}.log"
    # shellcheck disable=SC2016
    "${FONTFORGE:-fontforge}" -lang=ff -c 'Open($1); Generate($2)' "${sfd_name:?}.sfd" "${sfd_name:?}.ttf" >> "${sfd_name:?}.log" 2>&1; _E=$?
    ./errnum "${_E:?}" >> "${sfd_name:?}.log" 2>&1; printf '%s' "$(./errnum "${_E:?}" || :)"
    printf '%s\n' ""
  done
  exit 0
fi

trap '' SEGV > /dev/null 2>&1
trap '' BUS  > /dev/null 2>&1

for gct_file in gct_*_; do
  if [ -f "${gct_file:?}" ]; then
    base_name=$(printf '%s\n' "${gct_file:?}" | "${SED:-sed}" 's/gct_//' | "${SED:-sed}" 's/_$//')
    # shellcheck disable=SC2016
    sfd_name="GCT$(printf '%s\n' "${base_name:?}" \
      | "${AWK:-awk}" -F_ ' { for (i=1; i <= NF; i++) printf "%s", toupper (substr ($i, 1, 1)) substr ($i, 2) }')"
    printf 'Converting %s:' "${sfd_name:?}"

    printf '\t %s' " Stroke: "; printf '%s\n' "STROKE: $(date 2> /dev/null || :)" \
      "================================================================================" "" >> "${sfd_name:?}.log"
    ./convert_gct_stroke.py "${gct_file:?}" "${sfd_name:?}_Stroke.sfd" >> "${sfd_name:?}.log" 2>&1; _E=$?
    ./errnum "${_E:?}" >> "${sfd_name:?}.log" 2>&1; printf '%s' "$(./errnum "${_E:?}" || :)"

    printf '\t %s' "Outline: "; printf '%s\n' "" "OUTLINE: $(date 2> /dev/null || :)" \
      "================================================================================" "" >> "${sfd_name:?}.log"
    ./convert_gct_outline.py "${gct_file:?}" "${sfd_name:?}.sfd" >> "${sfd_name:?}.log" 2>&1; _E=$?
    ./errnum "${_E:?}" >> "${sfd_name:?}.log" 2>&1; printf '%s' "$(./errnum "${_E:?}" || :)"

    printf '%s\n' ""
  fi
done

_E=$(
  for log in ./*.log; do
    # shellcheck disable=SC2016
    "${GREP:-grep}" "Could not parse coordinates in" "${log:?}" 2> /dev/null \
      | "${AWK:-awk}" '{ print "*** "$0 }' || :
  done || : 2> /dev/null

  for log in ./*.log; do
    # shellcheck disable=SC2016
    "${GREP:-grep}" "^Glyph processing error" "${log:?}" 2> /dev/null \
      | "${AWK:-awk}" '{ print "*** "$0 }' || :
  done || : 2> /dev/null
)

test -z "${_E}" || {
  printf '\n%s\n' "${_E:-}" >&2
} || :

# shellcheck disable=SC2016
_itot=$("${GREP:-grep}" -E 'Processing [0-9]+ glyphs' ./*.log 2> /dev/null \
  | "${AWK:-awk}" '{ for (i = 1; i <= NF; i++)
                       if ($i == "Processing")
                         sum += $(i + 1)
                   } END { print (sum == "" ? 0 : sum) }') 2> /dev/null

# shellcheck disable=SC2016
_otot=$("${GREP:-grep}" -E 'Font saved to .* with [0-9]+ glyphs' ./*.log 2> /dev/null \
  | "${AWK:-awk}" '{ for (i = 1; i <= NF; i++)
                       if ($i == "with")
                         sum += $(i + 1)
                   } END { print (sum == "" ? 0 : sum) }') 2> /dev/null

printf '\nSUMMARY: %s\n' "${_itot:-0} total glyphs in, ${_otot:-0} total glyphs out."
