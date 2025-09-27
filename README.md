<!-- SPDX-License-Identifier: Multics or MIT-0 -->
<!-- Copyright (c) 2025 Jeffrey H. Johnson -->
<!-- scspell-id: 73135104-9b3c-11f0-b48f-80ee73e9b8e7 -->
# font-multics-gct

## Build

* The build process has been tested on **AIX**, **FreeBSD**, and **Linux**.

### Requirements

* A [C99 compiler](https://gcc.gnu.org/), [Python 3](https://www.python.org/), and [Fontforge](https://fontforge.org/) (with Python support).
* A POSIX **shell** environment with the `awk`, `date`, `grep`, `make`, `rm`, and `sed` tools.
* If your Python interpreter is not `python3`, minor modifications will be needed (*see* `grep 'python3' *`).
[]()

[]()
* To run the `lint` target (`make lint`), some additional tools are required:
  * [`reuse`](https://github.com/fsfe/reuse-tool), [`codespell`](https://github.com/codespell-project/codespell), [Black](https://github.com/psf/black), [ShellCheck](https://www.shellcheck.net/), [Cppcheck](https://www.cppcheck.com/), and [Clang Analyzer](https://clang-analyzer.llvm.org/).

## Fontforge issues

You need a recent version of **Fontforge** to successfully build the fonts.

* It is *recommend* to use Fontforge **based on sources from 2025-01-01 or later**.
  * Fontforge from **Fedora 42** and **43** is known to work.
  * Fontforge from **Ubuntu 25.04** and **25.10** is known to work (and **24.04** *seems* to work).
[]()

[]()
* Older versions of Fontforge are ***not*** sufficient and are *known to fail*.
  * Fontforge from **Ubuntu 18.04**, **20.04**, and **22.04** is known to fail.
  * Fontforge currently distributed by **Homebrew** is *very* old and known to fail.
[]()

[]()
* If you see warning messages such as the following, you have a **deficient** version of Fontforge:
  \
  `*** Glyph processing errors for gct_gothic_german_: 5 glyphs failed (z, l_brace, vert_bar, r_brace, tilde)`
[]()

[]()
*  You can check your Fontforge version from the command-line with `fontforge -v`.
