.PHONY: all clean
all:
	@./run.sh; \
	  status=$$?; if [ "$${status:-}" -eq 14 ]; then { printf '%s\n' "flock(1) error - is another instance running?"; exit 1; }; fi

clean:
	@./clean.sh; \
	  status=$$?; if [ "$${status:-}" -eq 14 ]; then { printf '%s\n' "flock(1) error - is another instance running?"; exit 1; }; fi
