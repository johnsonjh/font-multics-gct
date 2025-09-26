/*
 * vim: filetype=c:tabstop=2:softtabstop=2:shiftwidth=2:ai:expandtab
 * SPDX-License-Identifier: Multics or MIT-0
 * Copyright (c) 2025 Jeffrey H. Johnson
 * scspell-id: 3a9b1772-98e3-11f0-b710-80ee73e9b8e7
 */

#if defined (_POSIX_C_SOURCE)
# undef _POSIX_C_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L
#if !defined (_GNU_SOURCE)
# define _GNU_SOURCE
#endif
#if !defined (_NETBSD_SOURCE)
# define _NETBSD_SOURCE
#endif
#if !defined (_OPENBSD_SOURCE)
# define _OPENBSD_SOURCE
#endif
#if !defined (__BSD_VISIBLE)
# define __BSD_VISIBLE 1
#endif
#if !defined (__EXTENSIONS__)
# define __EXTENSIONS__
#endif

#include <errno.h>
#include <limits.h>
#include <locale.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined (__APPLE__)
# include <xlocale.h>
#endif

#if defined (__FreeBSD__) || defined (__OpenBSD__) || defined (__NetBSD__) || defined (__illumos__) || \
  ((defined (__sun) || defined (__sun__)) && (defined (__SVR4) || defined (__svr4__)))
# include <sys/signal.h>
#endif

#if !defined (NSIG)
# if defined (_NSIG)
#  define NSIG _NSIG
# endif
#endif

#if !defined (NSIG)
# error NSIG undefined
#endif

#if NSIG < 128
# undef NSIG
# define NSIG 128
#endif

#define DEF_EMAXLEN 32767

static const char *
xstrerror_l (int errnum)
{
  int saved = errno, n_buf;
  const char * ret = NULL;
  static char buf [DEF_EMAXLEN];

#if defined (__APPLE__) || defined (_AIX) || defined (__MINGW32__) || defined (__MINGW64__)
# if defined (__MINGW32__) || defined (__MINGW64__)
  (0 == strerror_s (buf, sizeof buf, errnum)) ? (ret = buf) : (void)0;
# else
  (0 == strerror_r (errnum, buf, sizeof buf)) ? (ret = buf) : (void)0;
# endif
#else
# if defined (__NetBSD__)
  locale_t loc = LC_GLOBAL_LOCALE;
# else
  locale_t loc = uselocale ((locale_t)0);
# endif
  locale_t copy = (loc == LC_GLOBAL_LOCALE ? duplocale (loc) : loc);
  ret = copy ? (ret = strerror_l (errnum, copy), (loc == LC_GLOBAL_LOCALE ? (freelocale (copy), 0) : 0), ret) : ret;
#endif
  ret = ret ? ret : ((n_buf = snprintf (buf, sizeof buf, "Unknown error %d", errnum)) < 0 || (size_t)n_buf >= sizeof buf
            ? (fprintf (stderr, "FATAL: snprintf buffer overflow at %s[%s:%d]\n", __func__, __FILE__, __LINE__),
               exit (EXIT_FAILURE), (const char *)0) : buf);

  errno = saved;

  return ret;
}

static const char *
xstrsignal (int sig)
{
#if defined (__MINGW32__) || defined (__MINGW64__)
  static char buf [DEF_EMAXLEN];
  (void)snprintf (buf, sizeof buf, "Signal %d", sig);
  return buf;
#else
  int saved = errno, n_buf;
  const char * ret = strsignal (sig);
  static char buf [DEF_EMAXLEN];

  ret = ret ? ret : (((n_buf = snprintf (buf, sizeof buf, "Unknown signal %d", sig)) < 0 || (size_t)n_buf >= sizeof buf)
            ? (fprintf (stderr, "FATAL: snprintf buffer overflow at %s[%s:%d]\n", __func__, __FILE__, __LINE__),
               exit (EXIT_FAILURE), (const char *)0) : buf);

  errno = saved;

  return ret;
#endif
}

int
main (int argc, char * * argv)
{
  long errnum;
  int errint;
  char * eptr;

  (void)setlocale(LC_ALL, "");

  errno = 0;

  errnum = (2 == argc) ? strtol (argv [1], & eptr, 10) : 0;

  const char *fatal =
    (2 != argc)                            ? "Usage: errnum <errno>"         :
    (ERANGE == errno)                      ? "FATAL: Out of range."          :
    (argv [1] == eptr || '\0' != * eptr)   ? "FATAL: Invalid number."        :
    (errnum < INT_MIN || errnum > INT_MAX) ? "FATAL: Number exceeds limits." : NULL;

  if (fatal)
    return (fprintf (stderr, "%s\n", fatal), EXIT_FAILURE);

  errint = (int)errnum;

  int xysh = (256 + 128 + 1 <= errint && errint < 256 + 128 + NSIG);
  int xksh = (! xysh && 256 + 1 <= errint && errint < 256 + NSIG);
  int xsig = (128 + 1 <= errint && errint < 128 + NSIG) || xksh || xysh;
  int errx = xsig ? (xysh ? errint - 256 - 128 : (xksh ? errint - 256 : errint - 128)) : errint;

  const char *(* msgfn)(int) = xsig ? xstrsignal : xstrerror_l;
  const char * lbl = xsig ? (xysh ? "yash signal" : xksh ? "ksh93 signal" : "signal") : "Error";
  const char * msg = msgfn (errx);

  return fprintf (stdout, errnum ? "%s (%s %d)\n" : "Success\n", errnum ? msg : 0, lbl, errx), EXIT_SUCCESS;
}
