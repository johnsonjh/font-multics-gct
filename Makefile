# SPDX-License-Identifier: Multics or MIT-0
# Copyright (c) 2025 Jeffrey H. Johnson
# Copyright (c) 2025 The DPS8M Development Team
# scspell-id: 9f257480-9924-11f0-84b8-80ee73e9b8e7

.PHONY: all clean
all run:
	@./run.sh

ttf:
	@./run.sh "ttf"

errnum: errnum.c

clean:
	@./clean.sh

lint:
	@rm -f errnum
	scan-build --status-bugs make errnum
	@rm -f errnum
	cppcheck --quiet --force --check-level=exhaustive *.c
	shellcheck -o any,all *.sh
	black --check *.py
	codespell -L Groupe .
	reuse lint -q || reuse lint

.NOT_PARALLEL:
