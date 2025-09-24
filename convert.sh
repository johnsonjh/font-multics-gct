#!/usr/bin/env sh

cc -o errnum errnum.c

for gct_file in gct_*_; do
  if [ -f "${gct_file:?}" ]; then
    base_name=$(printf '%s\n' "${gct_file:?}" | sed 's/gct_//' | sed 's/_$//')

    sfd_name="GCT$(printf '%s\n' "${base_name:?}" \
      | awk -F_ ' { for (i=1; i <= NF; i++) printf "%s", toupper (substr ($i, 1, 1)) substr ($i, 2) }')"
    printf 'Converting %s:\t' "${sfd_name:?}"

    printf '%s' " Stroke: "
    printf '%s\n' "STROKE" "======" "" >> "${sfd_name:?}.log"
    ./convert_gct_stroke.py "${gct_file:?}" "${sfd_name:?}_Stroke.sfd" >> "${sfd_name:?}.log" 2>&1
    _E=$?
    ./errnum ${_E:?} >> "${sfd_name:?}.log" 2>&1
    printf '%s' "$(./errnum "${_E:?}")"

    printf '\t %s' "Outline: "
    printf '%s\n' "" "OUTLINE" "=======" "" >> "${sfd_name:?}.log"
    ./convert_gct_outline.py "${gct_file:?}" "${sfd_name:?}.sfd" >> "${sfd_name:?}.log" 2>&1
    _E=$?
    ./errnum ${_E:?} >> "${sfd_name:?}.log" 2>&1
    printf '%s' "$(./errnum "${_E:?}")"

    test -f "${sfd_name:?}.sfd" && {
      printf '\t%s' "TrueType: "
      printf '%s\n' "" "TrueType" "========" "" >> "${sfd_name:?}.log"
      # shellcheck disable=SC2016
      fontforge -lang=ff -c 'Open($1); Generate($2)' "${sfd_name:?}.sfd" "${sfd_name:?}.ttf" >> "${sfd_name:?}.log" 2>&1
      _E=$?
      ./errnum ${_E:?} >> "${sfd_name:?}.log" 2>&1
      printf '%s' "$(./errnum "${_E:?}")"
    }

    printf '%s\n' ""
  fi
done
