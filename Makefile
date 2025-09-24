# SPDX-License-Identifier: Multics or MIT-0
# Copyright (c) 2025 Jeffrey H. Johnson
# Copyright (c) 2025 The DPS8M Development Team
# scspell-id: 9f257480-9924-11f0-84b8-80ee73e9b8e7

.PHONY: all clean
all run:
	@./run.sh; status=$$?; \
	if [ "$${status:-}" -eq 14 ]; then \
	{ printf '%s\n' "flock(1) error - is another instance running?" >&2; exit 1; }; fi

errnum: errnum.c

clean:
	@./clean.sh; status=$$?; \
	if [ "$${status:-}" -eq 14 ]; then \
	{ printf '%s\n' "flock(1) error - is another instance running?" >&2; exit 1; }; fi

.NOT_PARALLEL:
