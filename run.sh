#!/usr/bin/env sh

# shellcheck disable=SC2006,SC2046,SC2065,SC2116
test _`echo asdf 2>/dev/null` != _asdf >/dev/null &&\
  printf '%s\n' "FATAL: Using csh as sh is not supported." &&\
  exit 1

if [ "Linux" = "$(uname -s 2> /dev/null || :)" ]; then
  # shellcheck disable=SC2015
  command -v flock > /dev/null 2>&1 && {
    # shellcheck disable=SC2015,SC2154
    [ "${FLOCKER}" != "$0" ] && exec env FLOCKER="$0" flock -E 14 -en "$0" "$0" "$@" || :
  } || :
fi

export POSIXLY_CORRECT=1

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

check_for "${AWK:-awk}" "${CC:-cc}" "${GREP:-grep}" "${SED:-sed}" "fontforge" "python3"

test -x ./clean.sh && {
  ./clean.sh || :
}

test -x ./errnum || {
  "${CC:-cc}" -o errnum errnum.c || {
    printf 'FATAL: %s\n' "errnum.c compilation failed" >&2
    exit 1
  }
}

trap '' SEGV > /dev/null 2>&1
trap '' BUS  > /dev/null 2>&1

for gct_file in gct_*_; do
  if [ -f "${gct_file:?}" ]; then
    base_name=$(printf '%s\n' "${gct_file:?}" | "${SED:-sed}" 's/gct_//' | "${SED:-sed}" 's/_$//')
    # shellcheck disable=SC2016
    sfd_name="GCT$(printf '%s\n' "${base_name:?}" \
      | "${AWK:-awk}" -F_ ' { for (i=1; i <= NF; i++) printf "%s", toupper (substr ($i, 1, 1)) substr ($i, 2) }')"
    printf 'Converting %s:' "${sfd_name:?}"

    printf '\t %s' " Stroke: "; printf '%s\n' "STROKE" "======" "" >> "${sfd_name:?}.log"
    ./convert_gct_stroke.py "${gct_file:?}" "${sfd_name:?}_Stroke.sfd" >> "${sfd_name:?}.log" 2>&1; _E=$?
    ./errnum "${_E:?}" >> "${sfd_name:?}.log" 2>&1; printf '%s' "$(./errnum "${_E:?}" || :)"

    printf '\t %s' "Outline: "; printf '%s\n' "" "OUTLINE" "=======" "" >> "${sfd_name:?}.log"
    ./convert_gct_outline.py "${gct_file:?}" "${sfd_name:?}.sfd" >> "${sfd_name:?}.log" 2>&1; _E=$?
    ./errnum "${_E:?}" >> "${sfd_name:?}.log" 2>&1; printf '%s' "$(./errnum "${_E:?}" || :)"

    test -f "${sfd_name:?}.sfd" && {
      printf '\t%s' "TrueType: "; printf '%s\n' "" "TrueType" "========" "" >> "${sfd_name:?}.log"
      # shellcheck disable=SC2016
      fontforge -lang=ff -c 'Open($1); Generate($2)' "${sfd_name:?}.sfd" "${sfd_name:?}.ttf" >> "${sfd_name:?}.log" 2>&1; _E=$?
      ./errnum "${_E:?}" >> "${sfd_name:?}.log" 2>&1; printf '%s' "$(./errnum "${_E:?}" || :)"
    }

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
