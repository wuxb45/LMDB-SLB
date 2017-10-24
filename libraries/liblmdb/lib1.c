/*
 * Copyright (c) 2016  Wu, Xingbo <wuxb45@gmail.com>
 *
 * All rights reserved. No warranty, explicit or implicit, provided.
 */
#define _GNU_SOURCE

// headers {{{
#include "lib1.h"
#include <assert.h>
#include <math.h>
#include <execinfo.h>
#include <signal.h>
#include <stdatomic.h>
#include <sys/socket.h>
#include <netdb.h>
// }}} headers

// atomic {{{
/* C11 atomic types */
typedef atomic_bool             abool;

typedef atomic_uint_least8_t    au8;
typedef atomic_uint_least16_t   au16;
typedef atomic_uint_least32_t   au32;
typedef atomic_uint_least64_t   au64;

typedef atomic_int_least8_t     as8;
typedef atomic_int_least16_t    as16;
typedef atomic_int_least32_t    as32;
typedef atomic_int_least64_t    as64;
// }}} atomic

// locking {{{
  inline void
spinlock_init(spinlock * const lock)
{
  pthread_spin_init(&(lock->lock), PTHREAD_PROCESS_SHARED);
}

  inline void
spinlock_lock(spinlock * const lock)
{
  //lock->padding[7]++;
  const int r = pthread_spin_lock(&(lock->lock));
  if (r == EDEADLK) exit(0);
}

  inline bool
spinlock_trylock(spinlock * const lock)
{
  //lock->padding[7]++;
  const int r = pthread_spin_trylock(&(lock->lock));
  if (r == 0) return true;
  else if (r == EBUSY) return false;
  else exit(0);
}

  inline void
spinlock_unlock(spinlock * const lock)
{
  pthread_spin_unlock(&(lock->lock));
}

  inline void
mutexlock_init(mutexlock * const lock)
{
  pthread_mutex_init(&(lock->lock), NULL);
}

  inline void
mutexlock_lock(mutexlock * const lock)
{
  do {
    const int r = pthread_mutex_lock(&(lock->lock));
    if (r == 0) return;
    else if (r != EAGAIN) exit(0);
  } while (true);
}

  inline bool
mutexlock_trylock(mutexlock * const lock)
{
  do {
    const int r = pthread_mutex_trylock(&(lock->lock));
    if (r == 0) return true;
    else if (r == EBUSY) return false;
    else if (r != EAGAIN) exit(0);
  } while (true);
}

  inline void
mutexlock_unlock(mutexlock * const lock)
{
  do {
    const int r = pthread_mutex_unlock(&(lock->lock));
    if (r == 0) return;
    else if ((r != EAGAIN)) exit(0);
  } while (true);
}

#define RWLOCK_WBIT ((UINT64_C(0x8000000000000000)))
  inline void
rwlock_init(rwlock * const lock)
{
  au64 * const pvar = (typeof(pvar))(&(lock->var));
  atomic_store(pvar, 0);
}

  inline bool
rwlock_trylock_read(rwlock * const lock)
{
  au64 * const pvar = (typeof(pvar))(&(lock->var));
  u64 v0 = *pvar;
  if ((v0 & RWLOCK_WBIT) == UINT64_C(0)) {
    const u64 v1 = v0 + 1;
    const bool r = atomic_compare_exchange_strong(pvar, &v0, v1);
    return r;
  }
  return false;
}

  inline void
rwlock_lock_read(rwlock * const lock)
{
  while (rwlock_trylock_read(lock) == false);
}

  inline void
rwlock_unlock_read(rwlock * const lock)
{
  au64 * const pvar = (typeof(pvar))(&(lock->var));
  bool r = false;
  do {
    u64 v0 = *pvar;
    const u64 v1 = v0 - 1;
    r = atomic_compare_exchange_strong(pvar, &v0, v1);
  } while (r == false);
}

  inline bool
rwlock_trylock_write(rwlock * const lock)
{
  au64 * const pvar = (typeof(pvar))(&(lock->var));
  u64 v0 = *pvar;
  if ((v0 & RWLOCK_WBIT) == UINT64_C(0)) {
    const u64 v1 = v0 | RWLOCK_WBIT;
    const bool r = atomic_compare_exchange_strong(pvar, &v0, v1);
    if (r) {
      while (atomic_load(pvar) != RWLOCK_WBIT);
    }
    return r;
  }
  return false;
}

  inline void
rwlock_lock_write(rwlock * const lock)
{
  while (rwlock_trylock_write(lock) == false);
}

  inline void
rwlock_unlock_write(rwlock * const lock)
{
  au64 * const pvar = (typeof(pvar))(&(lock->var));
  atomic_store(pvar, UINT64_C(0));
}
#undef RWLOCK_WBIT
// }}} locking

// timing {{{
  inline u64
time_nsec(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return ts.tv_sec * UINT64_C(1000000000) + ts.tv_nsec;
}

  inline double
time_sec(void)
{
  const u64 nsec = time_nsec();
  return ((double)nsec) / 1000000000.0;
}

  inline u64
time_diff_nsec(const u64 last)
{
  return time_nsec() - last;
}

  inline double
time_diff_sec(const double last)
{
  return time_sec() - last;
}

  inline u64
timespec_diff(const struct timespec t0, const struct timespec t1)
{
  return UINT64_C(1000000000) * (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec);
}
// }}} timing

// debug {{{
  void
debug_backtrace(void)
{
  void *array[100];
  const int size = backtrace(array, 100);
  dprintf(2, "Backtrace (%d):\n", size);
  backtrace_symbols_fd(array, size, 2);
}

static u64 * __ptr_watch_u64 = NULL;

  static void
__signal_handler_watch_u64(const int sig)
{
  (void)sig;
  const u64 v = __ptr_watch_u64 ? (*__ptr_watch_u64) : 0;
  fprintf(stderr, "[USR1] %" PRIu64 " (0x%" PRIx64 ")\n", v, v);
}

  void
watch_u64_usr1(u64 * const ptr)
{
  __ptr_watch_u64 = ptr;
  struct sigaction sa = {};
  sa.sa_handler = __signal_handler_watch_u64;
  sigemptyset(&(sa.sa_mask));
  sa.sa_flags = SA_RESTART;
  if (sigaction(SIGUSR1, &sa, NULL) == -1) {
    fprintf(stderr, "Failed to set signal handler for SIGUSR1\n");
  } else {
    fprintf(stderr, "to watch> kill -s SIGUSR1 %d\n", getpid());
  }
}

  void
debug_wait_gdb(void)
{
  debug_backtrace();
  bool wait = true;
  volatile bool * const v = &wait;
  *v = true;

  time_t now;
  time(&now);
  struct tm nowtm;
  localtime_r(&now, &nowtm);
  char timestamp[64] = {};
  strftime(timestamp, 64, "%F %T %Z (%z)", &nowtm);

  char hostname[256] = {};
  gethostname(hostname, 256);
  char threadname[256];
  pthread_getname_np(pthread_self(), threadname, 256);

  const char * const pattern = "[Waiting GDB] %s %s @ %s\n    Attach me:   sudo -Hi gdb -p %d\n";
  fprintf(stderr, pattern, timestamp, threadname, hostname, getpid());
  fflush(stderr);
  // to continue: gdb> set var *v = 0
  while (*v) {
    sleep(1);
  }
}

#ifndef NDEBUG
  inline void
debug_assert(const bool v)
{
  if (!v) debug_wait_gdb();
}
#endif

  static void
__signal_handler_wait_gdb(const int sig, siginfo_t * const info, void * const context)
{
  (void)info;
  (void)context;
  printf("[SIGNAL] %s\n", strsignal(sig));
  debug_wait_gdb();
}

__attribute__((constructor))
  static void
debug_catch_fatals(void)
{
  struct sigaction sa = {};
  sa.sa_sigaction = __signal_handler_wait_gdb;
  sigemptyset(&(sa.sa_mask));
  sa.sa_flags = SA_SIGINFO;
  const int fatals[] = {SIGSEGV, SIGFPE, SIGILL, SIGBUS, 0};
  for (int i = 0; fatals[i]; i++) {
    if (sigaction(fatals[i], &sa, NULL) == -1) {
      fprintf(stderr, "Failed to set signal handler for %s\n", strsignal(fatals[i]));
      fflush(stderr);
    }
  }
}

  void
debug_dump_maps(FILE * const out)
{
  FILE * const in = fopen("/proc/self/smaps", "r");
  char * line0 = malloc(1024);
  size_t size0 = 1024;
  while (!feof(in)) {
    const ssize_t r1 = getline(&line0, &size0, in);
    if (r1 < 0) break;
    fprintf(out, "%s", line0);
  }
  fflush(out);
  fclose(in);
}

static pid_t __perf_pid = 0;

  void
debug_perf_start(void)
{
  const pid_t self = getpid();
  const pid_t perfpid = fork();
  if (perfpid < 0) {
    fprintf(stderr, "%s fork() failed\n", __func__);
  } else if (perfpid == 0) {
    // call perf
    char buf[10];
    sprintf(buf, "%d", self);
    char * args[10] = {};
    args[0] = "perf";
    args[1] = "record";
    args[2] = "--call-graph";
    args[3] = "lbr";
    args[4] = "--switch-output";
    args[5] = "-s";
    args[6] = "-p";
    args[7] = buf;
    args[8] = NULL;
    execvp(args[0], args);
    fprintf(stderr, "%s execvp() failed\n", __func__);
    fflush(stderr);
    exit(0);
  } else {
    __perf_pid = perfpid;
  }
}

  void
debug_perf_switch(void)
{
  if (__perf_pid > 0) {
    kill(__perf_pid, SIGUSR2);
  }
}

  void
debug_perf_stop(void)
{
  if (__perf_pid > 0) {
    kill(__perf_pid, SIGINT);
  }
}
// }}} debug

// bits {{{
  inline u32
bits_reverse_u32(const u32 v)
{
  const u32 v1 = (v >> 16) | (v << 16);
  const u32 v2 = ((v1 & UINT32_C(0xff00ff00)) >> 8) | ((v1 & UINT32_C(0x00ff00ff)) << 8);
  const u32 v3 = ((v2 & UINT32_C(0xf0f0f0f0)) >> 4) | ((v2 & UINT32_C(0x0f0f0f0f)) << 4);
  const u32 v4 = ((v3 & UINT32_C(0xcccccccc)) >> 2) | ((v3 & UINT32_C(0x33333333)) << 2);
  const u32 v5 = ((v4 & UINT32_C(0xaaaaaaaa)) >> 1) | ((v4 & UINT32_C(0x55555555)) << 1);
  return v5;
}

  inline u64
bits_reverse_u64(const u64 v)
{
  const u64 v0 = (v >> 32) | (v << 32);
  const u64 v1 = ((v0 & UINT64_C(0xffff0000ffff0000)) >> 16) | ((v0 & UINT64_C(0x0000ffff0000ffff)) << 16);
  const u64 v2 = ((v1 & UINT64_C(0xff00ff00ff00ff00)) >>  8) | ((v1 & UINT64_C(0x00ff00ff00ff00ff)) <<  8);
  const u64 v3 = ((v2 & UINT64_C(0xf0f0f0f0f0f0f0f0)) >>  4) | ((v2 & UINT64_C(0x0f0f0f0f0f0f0f0f)) <<  4);
  const u64 v4 = ((v3 & UINT64_C(0xcccccccccccccccc)) >>  2) | ((v3 & UINT64_C(0x3333333333333333)) <<  2);
  const u64 v5 = ((v4 & UINT64_C(0xaaaaaaaaaaaaaaaa)) >>  1) | ((v4 & UINT64_C(0x5555555555555555)) <<  1);
  return v5;
}

  inline u64
bits_rotl_u64(const u64 v, const u64 n)
{
  const u64 sh = n & 0x3f;
  return (v << sh) | (v >> (64 - sh));
}

  inline u64
bits_rotr_u64(const u64 v, const u64 n)
{
  const u64 sh = n & 0x3f;
  return (v >> sh) | (v << (64 - sh));
}

  inline u32
bits_rotl_u32(const u32 v, const u64 n)
{
  const u64 sh = n & 0x1f;
  return (v << sh) | (v >> (32 - sh));
}

  inline u32
bits_rotr_u32(const u32 v, const u64 n)
{
  const u64 sh = n & 0x1f;
  return (v >> sh) | (v << (32 - sh));
}
// }}} bits

// bitmap {{{
struct bitmap {
  u64 bits;
  au64 ones;
  u64 bm[];
};

  inline struct bitmap *
bitmap_create(const u64 bits)
{
  struct bitmap * const bm = (typeof(bm))calloc(1, sizeof(*bm) + (sizeof(u64) * ((bits + 63) >> 6)));
  bm->bits = bits;
  atomic_store(&(bm->ones), 0);
  return bm;
}

  inline bool
bitmap_test(const struct bitmap * const bm, const u64 idx)
{
  return ((idx < bm->bits) && (bm->bm[idx >> 6] & (UINT64_C(1) << (idx & UINT64_C(0x3f))))) ? true : false;
}

  inline bool
bitmap_test_all1(struct bitmap * const bm)
{
  return atomic_load(&(bm->ones)) == bm->bits ? true : false;
}

  inline bool
bitmap_test_all0(struct bitmap * const bm)
{
  return atomic_load(&(bm->ones)) == 0 ? true : false;
}

  inline void
bitmap_set1(struct bitmap * const bm, const u64 idx)
{
  if (idx < bm->bits && bitmap_test(bm, idx) == false) {
    bm->bm[idx >> 6] |= (UINT64_C(1) << (idx & UINT64_C(0x3f)));
    (void)atomic_fetch_add(&(bm->ones), 1);
  }
}

  inline void
bitmap_set0(struct bitmap * const bm, const u64 idx)
{
  if (idx < bm->bits && bitmap_test(bm, idx) == true) {
    bm->bm[idx >> 6] &= ~(UINT64_C(1) << (idx & UINT64_C(0x3f)));
    (void)atomic_fetch_sub(&(bm->ones), 1);
  }
}

  inline u64
bitmap_count(struct bitmap * const bm)
{
  return atomic_load(&(bm->ones));
}

  inline void
bitmap_set_all1(struct bitmap * const bm)
{
  memset(bm->bm, 0xff, (sizeof(u64) * ((bm->bits + 63) >> 6)));
  atomic_store(&(bm->ones), bm->bits);
}

  inline void
bitmap_set_all0(struct bitmap * const bm)
{
  memset(bm->bm, 0, (sizeof(u64) * ((bm->bits + 63) >> 6)));
  atomic_store(&(bm->ones), 0);
}

  inline void
bitmap_static_init(struct bitmap * const bm, const u64 bits)
{
  bm->bits = bits;
  bitmap_set_all0(bm);
}
// }}} bitmap

// bloom filter {{{
struct bloomfilter {
  u64 nr_probe;
  struct bitmap * bm;
};

  inline struct bloomfilter *
bf_create(const u64 bpk, const u64 capacity)
{
  struct bitmap * const bm = bitmap_create(bpk * capacity);
  if (bm) {
    struct bloomfilter * const bf = malloc(sizeof(*bf));
    bf->nr_probe = log(2.0) * (double)bpk;
    bf->bm = bm;
    return bf;
  }
  return NULL;
}

  inline void
bf_mark(struct bloomfilter * const bf, u64 hash64)
{
  u64 t = hash64;
  const u64 inc = bits_rotl_u64(hash64, 31);
  struct bitmap * const bm = bf->bm;
  for (u64 i = 0; i < bf->nr_probe; i++) {
    bitmap_set1(bm, t % bm->bits);
    t += inc;
  }
}

  inline bool
bf_test(const struct bloomfilter * const bf, u64 hash64)
{
  u64 t = hash64;
  const u64 inc = bits_rotl_u64(hash64, 31);
  const struct bitmap * const bm = bf->bm;
  for (u64 i = 0; i < bf->nr_probe; i++) {
    if (bitmap_test(bm, t % bm->bits) == false) return false;
    t += inc;
  }
  return true;
}

  inline void
bf_destroy(struct bloomfilter * const bf)
{
  free(bf->bm);
  free(bf);
}
// }}} bloom filter

// process/thread {{{
  u64
process_get_rss(void)
{
  u64 size;
  u64 rss;
  FILE * const fp = fopen("/proc/self/statm", "r");
  if (fp == NULL) return 0;
  if (fscanf(fp, "%" PRIu64 "%" PRIu64, &size, &rss) != 2) {
    fclose(fp);
    return 0;
  }
  fclose(fp);
  return rss * (u64)sysconf(_SC_PAGESIZE);
}

  static inline cpu_set_t *
__cpu_set_alloc(size_t * const size_out)
{
  const int ncpu = sysconf(_SC_NPROCESSORS_CONF);
  const size_t s1 = CPU_ALLOC_SIZE(ncpu);
  const size_t s2 = sizeof(cpu_set_t);
  const size_t size = s1 > s2 ? s1 : s2;
  *size_out = size;
  cpu_set_t * const set = malloc(size);
  return set;
}

  u64
process_affinity_core_count(void)
{
  size_t xsize = 0;
  cpu_set_t * const set = __cpu_set_alloc(&xsize);
  if (sched_getaffinity(0, xsize, set) != 0) {
    return sysconf(_SC_NPROCESSORS_CONF);
  }
  const int nr = CPU_COUNT_S(xsize, set);
  free(set);
  return (nr > 0) ? nr : sysconf(_SC_NPROCESSORS_CONF);
}

  u64
process_affinity_core_list(const u64 max, u64 * const cores)
{
  memset(cores, 0, max * sizeof(cores[0]));
  size_t xsize = 0;
  cpu_set_t * const set = __cpu_set_alloc(&xsize);
  if (sched_getaffinity(0, xsize, set) != 0) return 0;

  const u64 ncpu = sysconf(_SC_NPROCESSORS_CONF);
  const u64 nr_affinity = CPU_COUNT_S(xsize, set);
  const u64 nr = nr_affinity < max ? nr_affinity : max;
  u64 j = 0;
  for (u64 i = 0; i < ncpu; i++) {
    if (CPU_ISSET_S((int)i, xsize, set)) {
      cores[j++] = i;
    }
    if (j >= nr) break;
  }
  free(set);
  return j;
}

  u64
process_cpu_time_usec(void)
{
  struct rusage r;
  getrusage(RUSAGE_SELF, &r);
  const u64 usr = (r.ru_utime.tv_sec * UINT64_C(1000000)) + r.ru_utime.tv_usec;
  const u64 sys = (r.ru_stime.tv_sec * UINT64_C(1000000)) + r.ru_stime.tv_usec;
  return usr + sys;
}

  void
thread_set_affinity(const u64 cpu)
{
  const u64 ncpu = sysconf(_SC_NPROCESSORS_CONF);
  size_t xsize = 0;
  cpu_set_t * const set = __cpu_set_alloc(&xsize);

  CPU_ZERO_S(xsize, set);
  CPU_SET_S(cpu % ncpu, xsize, set);
  sched_setaffinity(0, xsize, set);
  free(set);
}

  double
thread_fork_join_private(const u64 nr, void *(*func) (void *), void * const * const argv)
{
  if (nr == 0) return 0.0;
  const u64 ncpu = sysconf(_SC_NPROCESSORS_CONF);
  int cores[ncpu];
  size_t xsize = 0;
  cpu_set_t * const set = __cpu_set_alloc(&xsize);

  const bool force_use = (sched_getaffinity(0, xsize, set) != 0) ? true : false;
  u64 j = 0;
  for (u64 i = 0; i < ncpu; i++) {
    if (force_use || CPU_ISSET_S((int)i, xsize, set)) {
      cores[j++] = i;
    }
  }
  const u64 ncores = j;

  const u64 nr_threads = nr ? nr : ncores;
  pthread_t tids[nr_threads];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  char thname[32];
  const double t0 = time_sec();
  for (u64 i = 0; i < nr_threads; i++) {
    CPU_ZERO_S(xsize, set);
    CPU_SET_S(cores[i % ncores], xsize, set);
    pthread_attr_setaffinity_np(&attr, xsize, set);
    const int r = pthread_create(&(tids[i]), &attr, func, argv[i]);
    if (r != 0) {
      tids[i] = 0;
    } else {
      sprintf(thname, "fork_join_%"PRIu64, i);
      pthread_setname_np(tids[i], thname);
    }
  }
  for (u64 i = 0; i < nr_threads; i++) {
    if (tids[i]) pthread_join(tids[i], NULL);
  }
  const double dt = time_diff_sec(t0);
  pthread_attr_destroy(&attr);
  free(set);
  return dt;
}

  inline double
thread_fork_join(const u64 nr, void *(*func) (void *), void * const arg)
{
  const u64 nthreads = nr ? nr : process_affinity_core_count();
  void * argv[nthreads];
  for (u64 i = 0; i < nthreads; i++) {
    argv[i] = arg;
  }
  return thread_fork_join_private(nthreads, func, argv);
}

  inline int
thread_create_at(const u64 cpu, pthread_t * const thread, void *(*start_routine) (void *), void * const arg)
{
  const u64 ncpu = sysconf(_SC_NPROCESSORS_CONF);
  const u64 cpu_id = cpu % ncpu;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  size_t xsize = 0;
  cpu_set_t * const set = __cpu_set_alloc(&xsize);

  CPU_ZERO_S(xsize, set);
  CPU_SET_S(cpu_id, xsize, set);
  pthread_attr_setaffinity_np(&attr, xsize, set);
  const int r = pthread_create(thread, &attr, start_routine, arg);
  pthread_attr_destroy(&attr);
  free(set);
  return r;
}
// }}} process/thread

// mm {{{
  inline void *
xalloc(const u64 align, const u64 size)
{
  void * p;
  const int r = posix_memalign(&p, align, size);
  if (r == 0) return p;
  else return NULL;
}

  inline void
pages_unmap(void * const ptr, const size_t size)
{
#ifndef HEAPCHECKING
  munmap(ptr, size);
#else
  (void)size;
  free(ptr);
#endif
}

  static inline void *
__pages_alloc(const size_t size, const int flags)
{
  void * const p = mmap(NULL, size, PROT_READ | PROT_WRITE, flags | MAP_LOCKED, -1, 0);
  if (p == MAP_FAILED) {
    return NULL;
  }
  return p;
}

  inline void *
pages_alloc_1gb(const size_t nr_1gb)
{
  const u64 sz = nr_1gb << 30;
#ifndef HEAPCHECKING
  return __pages_alloc(sz, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (30 << MAP_HUGE_SHIFT));
#else
  void * const p = xalloc(UINT64_C(1) << 30, sz);
  if (p) memset(p, 0, sz);
  return p;
#endif
}

  inline void *
pages_alloc_2mb(const size_t nr_2mb)
{
  const u64 sz = nr_2mb << 21;
#ifndef HEAPCHECKING
  return __pages_alloc(sz, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (21 << MAP_HUGE_SHIFT));
#else
  void * const p = xalloc(UINT64_C(1) << 21, sz);
  if (p) memset(p, 0, sz);
  return p;
#endif
}

  inline void *
pages_alloc_4kb(const size_t nr_4kb)
{
  const size_t sz = nr_4kb << 12;
#ifndef HEAPCHECKING
  return __pages_alloc(sz, MAP_PRIVATE | MAP_ANONYMOUS);
#else
  void * const p = xalloc(UINT64_C(1) << 12, sz);
  if (p) memset(p, 0, sz);
  return p;
#endif
}

  void *
pages_alloc_best(const size_t size, const bool try_1gb, u64 * const size_out)
{
  if (try_1gb) {
    const size_t nr_1gb = (size + ((UINT64_C(1) << 30) - UINT64_C(1))) >> 30;
    // 1gb super huge page: waste < 1/16 or 6.25%
    if (((nr_1gb << 30) - size) < (size >> 4)) {
      void * const p1 = pages_alloc_1gb(nr_1gb);
      if (p1) {
        *size_out = nr_1gb << 30;
        return p1;
      }
    }
  }

  // 2mb huge page: at least 1MB
  if (size >= (UINT64_C(1) << 20)) {
    const size_t nr_2mb = (size + ((UINT64_C(1) << 21) - UINT64_C(1))) >> 21;
    void * const p2 = pages_alloc_2mb(nr_2mb);
    if (p2) {
      *size_out = nr_2mb << 21;
      return p2;
    }
  }

  const size_t nr_4kb = (size + ((UINT64_C(1) << 12) - UINT64_C(1))) >> 12;
  void * const p3 = pages_alloc_4kb(nr_4kb);
  if (p3) {
    *size_out = nr_4kb << 12;
  }
  return p3;
}
// }}} mm

// oalloc {{{
#define OALLOC_UNIT_POWER ((18))
#define OALLOC_UNIT_SIZE  ((UINT64_C(1) << OALLOC_UNIT_POWER))
#define OALLOC_PAGE_POWER ((30))
#define OALLOC_PAGE_SIZE  ((UINT64_C(1) << OALLOC_PAGE_POWER))
#define OALLOC_NR_UNIT    ((OALLOC_PAGE_SIZE / OALLOC_UNIT_SIZE))
#define OALLOC_NR_PAGE    ((1000))
#define OALLOC_NR_SHARDS  ((1024))
struct oalloc_lp { // large pool
  spinlock lock;
  u64 page_id;
  u64 unit_id;
  u8 * pages[OALLOC_NR_PAGE];
};

struct oalloc_sp { // small pool
  spinlock lock;
  u8 * unit;
  u64 offset;
};

struct oalloc {
  struct oalloc_lp lp;
  struct oalloc_sp sps[OALLOC_NR_SHARDS];
};

  struct oalloc *
oalloc_create(void)
{
  struct oalloc * const oa = (typeof(oa))calloc(1, sizeof(*oa));
  spinlock_init(&(oa->lp.lock));
  for (u64 i = 0; i < OALLOC_NR_SHARDS; i++) {
    spinlock_init(&(oa->sps[i].lock));
  }
  return oa;
}

  static u8 *
oalloc_lp_alloc(struct oalloc_lp * const lp)
{
  spinlock_lock(&(lp->lock));
  if ((lp->unit_id) >= OALLOC_NR_UNIT) {
    lp->page_id++;
    lp->unit_id = 0;
  }

  if (lp->page_id >= OALLOC_NR_PAGE) {
    spinlock_unlock(&(lp->lock));
    return NULL;
  }
  const u64 pid = lp->page_id;
  const u64 uid = lp->unit_id;
  if (lp->pages[pid] == NULL) {
    u64 size = 0;
    lp->pages[pid] = pages_alloc_best(OALLOC_PAGE_SIZE, true, &size);
    debug_assert(size == OALLOC_PAGE_SIZE);
  }

  u8 * const ptr = &(lp->pages[pid][uid << OALLOC_UNIT_POWER]);
  lp->unit_id++;
  spinlock_unlock(&(lp->lock));
  return ptr;
}

  void *
oalloc_alloc(const u64 size, struct oalloc * const oa)
{
  const u64 s = random_u64() % OALLOC_NR_SHARDS;
  struct oalloc_sp * const sp = &(oa->sps[s]);
  const u64 asize = (size + 7) & (~ UINT64_C(7));
  spinlock_lock(&(sp->lock));
  if (sp->unit == NULL || ((sp->offset + asize) > OALLOC_UNIT_SIZE)) {
    sp->unit = oalloc_lp_alloc(&(oa->lp));
    sp->offset = 0;
    if (sp->unit == NULL) {
      spinlock_unlock(&(sp->lock));
      return NULL;
    }
  }
  void * const ptr = &(sp->unit[sp->offset]);
  sp->offset += asize;
  spinlock_unlock(&(sp->lock));
  return ptr;
}

  void
oalloc_destroy(struct oalloc * const oa)
{
  for (u64 i = 0; i < OALLOC_NR_PAGE; i++) {
    if (oa->lp.pages[i]) {
      pages_unmap(oa->lp.pages[i], OALLOC_PAGE_SIZE);
    }
  }
  free(oa);
}
// }}} oalloc

// gcache {{{
struct gcache_object {
  struct gcache_object * next;
};

#define GALLOC_SHARDS ((32))
struct gcache_class {
  struct gcache_object * heads[GALLOC_SHARDS];
};

struct gcache {
  spinlock locks[GALLOC_SHARDS];
  u64 nr;
  u64 inc;
  struct gcache_class cs[];
};

  struct gcache *
gcache_create(const u64 nr_classes, const u64 inc)
{
  struct gcache * const g = (typeof(g))calloc(1, sizeof(*g) + (nr_classes * sizeof(g->cs[0])));
  for (u64 i = 0; i < GALLOC_SHARDS; i++) {
    spinlock_init(&(g->locks[i]));
  }
  g->nr = nr_classes;
  g->inc = inc;
  return g;
}

  static void *
gcache_pull_gc(struct gcache * const g, struct gcache_class * const gc)
{
  const u64 r = random_u64();
  for (u64 i = 0; i < GALLOC_SHARDS; i++) {
    const u64 idx = (i + r) % GALLOC_SHARDS;
    if (gc->heads[idx]) {
      spinlock_lock(&(g->locks[idx]));
      if (gc->heads[idx]) {
        struct gcache_object * const curr = gc->heads[idx];
        gc->heads[idx] = curr->next;
        spinlock_unlock(&(g->locks[idx]));
        return (void *)curr;
      }
      spinlock_unlock(&(g->locks[idx]));
    }
  }
  return NULL;
}

  void *
gcache_pull(struct gcache * const g, const u64 size)
{
  const u64 idx = (size + g->inc - 1) / g->inc;
  if (idx >= g->nr || idx == 0) return NULL;
  struct gcache_class * const gc = &(g->cs[idx]);
  return gcache_pull_gc(g, gc);
}

  static void
gcache_push_gc(struct gcache * const g, struct gcache_class * const gc, struct gcache_object * const go)
{
  const u64 idx = random_u64() % GALLOC_SHARDS;
  spinlock_lock(&(g->locks[idx]));
  go->next = gc->heads[idx];
  gc->heads[idx] = go;
  spinlock_unlock(&(g->locks[idx]));
}

  bool
gcache_push(struct gcache * const g, const u64 size, void * ptr)
{
  struct gcache_object * const go = (typeof(go))ptr;
  const u64 idx = (size + g->inc - 1) / g->inc;
  if (idx >= g->nr || idx == 0) return false;
  struct gcache_class * const gc = &(g->cs[idx]);
  gcache_push_gc(g, gc, go);
  return true;
}

  void
gcache_clean(struct gcache * const g)
{
  for (u64 i = 0; i < GALLOC_SHARDS; i++) {
    spinlock_lock(&(g->locks[i]));
  }
  for (u64 i = 1; i < g->nr; i++) {
    memset(&(g->cs[i]), 0, sizeof(g->cs[i]));
  }
  for (u64 i = 0; i < GALLOC_SHARDS; i++) {
    spinlock_unlock(&(g->locks[i]));
  }
}

  void
gcache_destroy(struct gcache * const g)
{
  free(g);
}

struct gcache_iter {
  struct gcache * g;
  u64 cid;
  u64 sid;
  struct gcache_object * curr;
};

  struct gcache_iter *
gcache_iter_create(struct gcache * const g)
{
  struct gcache_iter * const gi = calloc(1, sizeof(*gi));
  gi->g = g;
  gi->cid = 1;
  gi->curr = g->cs[1].heads[0];
  return gi;
}

  void *
gcache_iter_next(struct gcache_iter * const gi)
{
  if (gi->cid >= gi->g->nr) return NULL;
  while (gi->curr == NULL) {
    gi->sid++;
    if (gi->sid >= GALLOC_SHARDS) {
      gi->cid++;
      gi->sid = 0;
    }
    if (gi->cid >= gi->g->nr) return NULL;
    gi->curr = gi->g->cs[gi->cid].heads[gi->sid];
  }
  struct gcache_object * const curr = gi->curr;
  gi->curr = curr->next;
  return curr;
}

  void
gcache_iter_destroy(struct gcache_iter * const gi)
{
  free(gi);
}
// }}} gcache

// cpucache {{{
  inline void
cpu_clflush1(void * const ptr)
{
  __builtin_ia32_clflush(ptr);
}

  inline void
cpu_clflush(void * const ptr, const size_t size)
{
  u8 * first = (typeof(first))(((u64)ptr) & (~ UINT64_C(0x3f)));
  u8 * const last  = (typeof(last))(((u64)(((u8*)ptr) + (size - 1u))) & (~ UINT64_C(0x3f)));
  while ((last - first) >= 256) {
    __builtin_ia32_clflush(first);
    __builtin_ia32_clflush(first + 64);
    __builtin_ia32_clflush(first + 128);
    __builtin_ia32_clflush(first + 192);
    first += 256;
  }
  for (u8 * p = first; p <= last; p += 64u) {
    __builtin_ia32_clflush(p);
  }
}

  inline void
cpu_mfence(void)
{
  __builtin_ia32_mfence();
}
// }}} cpucache

// qsort {{{
  static int
__compare_u16(const void * const p1, const void * const p2)
{
  const u16 v1 = *((const u16 *)p1);
  const u16 v2 = *((const u16 *)p2);
  if (v1 < v2) return -1;
  else if (v1 > v2) return 1;
  else return 0;
}

  inline void
qsort_u16(u16 * const array, const size_t nr)
{
  qsort(array, nr, sizeof(array[0]), __compare_u16);
}

  static int
__compare_u32(const void * const p1, const void * const p2)
{
  const u32 v1 = *((const u32 *)p1);
  const u32 v2 = *((const u32 *)p2);
  if (v1 < v2) return -1;
  else if (v1 > v2) return 1;
  else return 0;
}

  inline void
qsort_u32(u32 * const array, const size_t nr)
{
  qsort(array, nr, sizeof(array[0]), __compare_u32);
}

  static int
__compare_u64(const void * const p1, const void * const p2)
{
  const u64 v1 = *((const u64 *)p1);
  const u64 v2 = *((const u64 *)p2);
  if (v1 < v2) return -1;
  else if (v1 > v2) return 1;
  else return 0;
}

  inline void
qsort_u64(u64 * const array, const size_t nr)
{
  qsort(array, nr, sizeof(array[0]), __compare_u64);
}

  static int
__compare_double(const void * const p1, const void * const p2)
{
  const double v1 = *((const double *)p1);
  const double v2 = *((const double *)p2);
  if (v1 < v2) return -1;
  else if (v1 > v2) return 1;
  else return 0;
}

  inline void
qsort_double(double * const array, const size_t nr)
{
  qsort(array, nr, sizeof(array[0]), __compare_double);
}

  void
qsort_u64_sample(const u64 * const array0, const u64 nr, const u64 res, FILE * const out)
{
  const u64 datasize = nr * sizeof(array0[0]);
  u64 * const array = (typeof(array))malloc(datasize);
  debug_assert(array);
  memcpy(array, array0, datasize);
  qsort_u64(array, nr);

  const double sized = (double)nr;
  const u64 srate = res ? res : 64;
  const u64 xstep = ({u64 step = nr / srate; step ? step : 1; });
  const u64 ystep = ({u64 step = (array[nr - 1] - array[0]) / srate; step ? step : 1; });
  u64 i = 0;
  fprintf(out, "%lu %06.2lf %lu\n", i, ((double)(i + 1lu)) * 100.0 / sized, array[i]);
  for (u64 j = 1; j < nr; j++) {
    if (((j - i) >= xstep) || (array[j] - array[i]) >= ystep) {
      i = j;
      fprintf(out, "%lu %06.2lf %lu\n", i, ((double)(i + 1lu)) * 100.0 / sized, array[i]);
    }
  }
  if (i != (nr - 1)) {
    i = nr - 1;
    fprintf(out, "%lu %06.2lf %lu\n", i, ((double)(i + 1lu)) * 100.0 / sized, array[i]);
  }
  free(array);
}

  void
qsort_double_sample(const double * const array0, const u64 nr, const u64 res, FILE * const out)
{
  const u64 datasize = nr * sizeof(double);
  double * const array = (typeof(array))malloc(datasize);
  debug_assert(array);
  memcpy(array, array0, datasize);
  qsort_double(array, nr);

  const u64 srate = res ? res : 64;
  const double srate_d = (double)srate;
  const u64 sized = (double)nr;
  const u64 xstep = ({u64 step = nr / srate; step ? step : 1; });
  const double ystep = ({ double step = fabs((array[nr - 1] - array[0]) / srate_d); step ? step : 1.0; });
  u64 i = 0;
  fprintf(out, "%lu %06.2lf %020.9lf\n", i, ((double)(i + 1lu)) * 100.0 / sized, array[i]);
  for (u64 j = 1; j < nr; j++) {
    if (((j - i) >= xstep) || (array[j] - array[i]) >= ystep) {
      i = j;
      fprintf(out, "%lu %06.2lf %020.9lf\n", i, ((double)(i + 1lu)) * 100.0 / sized, array[i]);
    }
  }
  if (i != (nr - 1)) {
    i = nr - 1;
    fprintf(out, "%lu %06.2lf %020.9lf\n", i, ((double)(i + 1lu)) * 100.0 / sized, array[i]);
  }
  free(array);
}
// }}} qsort

// hash {{{
#if defined(HAVE_sse4_2_crc32qi) && defined(HAVE_sse4_2_crc32hi) && defined(HAVE_sse4_2_crc32si) && defined(HAVE_sse4_2_crc32di)
  inline u32
crc32(const void * const ptr, const size_t size)
{
#define ALIGN_SIZE  ((UINT32_C(0x08)))
#define ALIGN_MASK  ((ALIGN_SIZE - 1))
  union {
    const u8 * p1;
    const u16 * p2;
    const u32 * p4;
    const u64 * p8;
    const void * p;
    u64 v;
  } iter = {.p = ptr};

  u32 crc = UINT32_C(0xffffffff);
  size_t todo = size;
  // align to 8 one by one
  while (todo && (iter.v & ALIGN_MASK)) {
    crc = __builtin_ia32_crc32qi(crc, *(iter.p1)); iter.v++; todo--;
  }
  // 8
  while (todo >= sizeof(*(iter.p8))) {
    crc = __builtin_ia32_crc32di(crc, *(iter.p8)); iter.v += 8; todo -= 8;
  }
  // 4
  while (todo >= sizeof(*(iter.p4))) {
    crc = __builtin_ia32_crc32si(crc, *(iter.p4)); iter.v += 4; todo -= 4;
  }
  // 2
  while (todo >= sizeof(*(iter.p2))) {
    crc = __builtin_ia32_crc32hi(crc, *(iter.p2)); iter.v += 2; todo -= 2;
  }
  // 1
  while (todo >= sizeof(*(iter.p1))) {
    crc = __builtin_ia32_crc32qi(crc, *(iter.p1)); iter.v++; todo--;
  }
  return crc ^ UINT32_C(0xffffffff);
#undef ALIGN_SIZE
#undef ALIGN_MASK
}
#else
static const u32 __crc32_tab[] = {
  0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
  0xe963a535, 0x9e6495a3,	0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
  0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
  0xf3b97148, 0x84be41de,	0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
  0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec,	0x14015c4f, 0x63066cd9,
  0xfa0f3d63, 0x8d080df5,	0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
  0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b,	0x35b5a8fa, 0x42b2986c,
  0xdbbbc9d6, 0xacbcf940,	0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
  0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
  0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
  0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d,	0x76dc4190, 0x01db7106,
  0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
  0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
  0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
  0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
  0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
  0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
  0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
  0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
  0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
  0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
  0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
  0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
  0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
  0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
  0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
  0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
  0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
  0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
  0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
  0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
  0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
  0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
  0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
  0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
  0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
  0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
  0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
  0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
  0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
  0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
  0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
  0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d,
};

  inline u32
crc32(const void * const ptr, const size_t size)
{
  const uint8_t * p = (typeof(p))ptr;
  size_t sz = size;

  u32 crc = UINT32_C(0xffffffff);

  while (sz--)
    crc = __crc32_tab[(crc ^ *p++) & 0xFF] ^ (crc >> 8);

  return crc ^ UINT32_C(0xffffffff);
}
#endif

#define XXH_PRIVATE_API
#include "xxhash.h"
#undef XXH_PRIVATE_API

  inline u32
xxhash32(const void * const ptr, const size_t size)
{
  return XXH32(ptr, size, 0);
}

  inline u64
xxhash64(const void * const ptr, const size_t size)
{
  return XXH64(ptr, size, 0);
}
// }}} hash

// xlog {{{
  struct xlog *
xlog_create(const u64 nr_init, const u64 unit_size)
{
  struct xlog * const xlog = (typeof(xlog))xalloc(64, sizeof(*xlog));
  debug_assert(xlog);
  xlog->nr_rec = 0;
  xlog->nr_cap = nr_init ? nr_init : 4096;
  debug_assert(unit_size);
  xlog->unit_size = unit_size;

  xlog->ptr = malloc(xlog->unit_size * xlog->nr_cap);
  debug_assert(xlog->ptr);
  return xlog;
}

  static inline void
xlog_enlarge(struct xlog * const xlog)
{
  const u64 new_cap = (xlog->nr_cap < (1lu<<20)) ? (xlog->nr_cap * 2lu) : (xlog->nr_cap + (1lu<<20));
  void * const new_ptr = realloc(xlog->ptr, xlog->unit_size * new_cap);
  debug_assert(new_ptr);
  xlog->ptr = new_ptr;
  xlog->nr_cap = new_cap;
}

  inline void
xlog_append(struct xlog * const xlog, const void * const rec)
{
  if (xlog->nr_rec == xlog->nr_cap) {
    xlog_enlarge(xlog);
  }
  u8 * const ptr = xlog->ptr + (xlog->nr_rec * xlog->unit_size);
  memcpy(ptr, rec, xlog->unit_size);
  xlog->nr_rec++;
}

  inline void
xlog_append_cycle(struct xlog * const xlog, const void * const rec)
{
  if (xlog->nr_rec == xlog->nr_cap) {
    xlog->nr_rec = 0;
  }
  xlog_append(xlog, rec);
}
  inline void
xlog_reset(struct xlog * const xlog)
{
  xlog->nr_rec = 0;
}

  void
xlog_dump(struct xlog * const xlog, FILE * const out)
{
  const size_t nd = fwrite(xlog->ptr, xlog->unit_size, xlog->nr_rec, out);
  (void)nd;
  debug_assert(nd == xlog->nr_rec);
}

  inline void
xlog_destroy(struct xlog * const xlog)
{
  free(xlog->ptr);
  free(xlog);
}

struct xlog_iter {
  const struct xlog * xlog;
  u64 next_id;
};

  inline struct xlog_iter *
xlog_iter_create(const struct xlog * const xlog)
{
  struct xlog_iter * const iter = (typeof(iter))malloc(sizeof(*iter));
  iter->xlog = xlog;
  iter->next_id = 0;
  return iter;
}

  inline bool
xlog_iter_next(struct xlog_iter * const iter, void * const out)
{
  const struct xlog * const xlog = iter->xlog;
  if (iter->next_id < xlog->nr_rec) {
    void * const ptr = xlog->ptr + (xlog->unit_size * iter->next_id);
    memcpy(out, ptr, xlog->unit_size);
    iter->next_id++;
    return true;
  } else {
    return false;
  }
}
// }}} xlog

// string {{{
static u16 __conv_table_10[100];

__attribute__((constructor))
  static void
__conv_table_10_init(void)
{
  for (u8 i = 0; i < 100; i++) {
    const u8 hi = (typeof(hi))('0' + (i / 10));
    const u8 lo = (typeof(lo))('0' + (i % 10));
    __conv_table_10[i] = (lo << 8) | hi;
  }
}

// output 10 bytes
  void
str10_u32(void * const out, const u32 v)
{
  u32 vv = v;
  u16 * const ptr = (typeof(ptr))out;
  for (int i = 4; i >= 0; i--) {
    ptr[i] = __conv_table_10[vv % 100];
    vv /= 100u;
  }
}

// output 20 bytes
  void
str10_u64(void * const out, const u64 v)
{
  u64 vv = v;
  u16 * const ptr = (typeof(ptr))out;
  for (int i = 9; i >= 0; i--) {
    ptr[i] = __conv_table_10[vv % 100];
    vv /= 100;
  }
}

static const u8 __conv_table_16[16] = {
  '0','1','2','3','4','5','6','7',
  '8','9','a','b','c','d','e','f',
};

// output 8 bytes
  void
str16_u32(void * const out, const u32 v)
{
  u8 * const ptr = (typeof(ptr))out;
  for (int i = 0; i < 8; i++) {
    ptr[i] = __conv_table_16[(v >> (28 - (i << 2))) & 0xf];
  }
}

// output 16 bytes
  void
str16_u64(void * const out, const u64 v)
{
  u8 * const ptr = (typeof(ptr))out;
  for (int i = 0; i < 16; i++) {
    ptr[i] = __conv_table_16[(v >> (60 - (i << 2))) & 0xf];
  }
}

// returns a NULL-terminated list of string tokens (char **).
// After use free the returned pointer (tokens).
  char **
string_tokens(const char * const str, const char * const delim)
{
  if (str == NULL) return NULL;
  size_t aptr = 32;
  char ** tokens = (typeof(tokens))malloc(sizeof(tokens[0]) * aptr);
  if (tokens == NULL) return NULL;
  size_t i = 0;
  char * const dup = strdup(str);
  char * internal = NULL;
  const size_t bufsize = strlen(dup) + 1;
  char * tok = strtok_r(dup, delim, &internal);
  while (tok) {
    if (i >= aptr) {
      char ** const r = (typeof(tokens))realloc(tokens, sizeof(tokens[0]) * (aptr + 32));
      if (r == NULL) {
        free(tokens);
        free(dup);
        return NULL;
      }
      tokens = r;
      aptr += 32;
    }
    tokens[i] = tok;
    i++;
    tok = strtok_r(NULL, delim, &internal);
  }
  tokens[i] = tok;
  const size_t nptr = i + 1;
  const size_t rsize = (sizeof(tokens[0]) * nptr) + bufsize;
  char ** const r = (typeof(tokens))realloc(tokens, rsize);
  if (r == NULL) {
    free(tokens);
    free(dup);
    return NULL;
  }
  tokens = r;
  char * const content = (char *)(&(tokens[nptr]));
  memcpy(content, dup, bufsize);
  for (u64 j = 0; j < i; j++) {
    tokens[j] += (content - dup);
  }
  free(dup);
  return tokens;
}
// }}} string

// damp {{{
struct damp {
  u64 cap;
  u64 used;
  u64 next;
  u64 events;
  double dshort;
  double dlong;
  double hist[];
};

  struct damp *
damp_create(const u64 cap, const double dshort, const double dlong)
{
  struct damp * const d = malloc(sizeof(*d) + (sizeof(d->hist[0]) * cap));
  d->cap = cap;
  d->used = 0;
  d->next = 0;
  d->events = 0;
  d->dshort = dshort;
  d->dlong = dlong;
  return d;
}

  double
damp_average(const struct damp * const d)
{
  if (d->used == 0) return 0.0;
  const u64 start = d->next - d->used;
  double sum = 0.0;
  for (u64 i = 0; i < d->used; i++) {
    const u64 idx = (start + i) % d->cap;
    sum += d->hist[idx];
  }
  const double avg = sum / ((double)d->used);
  return avg;
}

  double
damp_min(const struct damp * const d)
{
  if (d->used == 0) return 0.0;
  const u64 start = d->next - d->used;
  double min = d->hist[start % d->cap];
  for (u64 i = 1; i < d->used; i++) {
    const u64 idx = (start + i) % d->cap;
    const double v = d->hist[idx];
    if (v < min) min = v;
  }
  return min;
}

  double
damp_max(const struct damp * const d)
{
  if (d->used == 0) return 0.0;
  const u64 start = d->next - d->used;
  double max = d->hist[start % d->cap];
  for (u64 i = 1; i < d->used; i++) {
    const u64 idx = (start + i) % d->cap;
    const double v = d->hist[idx];
    if (v > max) max = v;
  }
  return max;
}

  bool
damp_add_test(struct damp * const d, const double v)
{
  d->hist[d->next] = v;
  d->next = (d->next + 1) % d->cap;
  if (d->used < d->cap) d->used++;
  d->events++;

  // short-distance history
  const u64 end = d->next - 1;
  if (d->used >= 3) {
    const double v0 = d->hist[(end - 0) % d->cap];
    const double v1 = d->hist[(end - 1) % d->cap];
    const double v2 = d->hist[(end - 2) % d->cap];
    const double dd = v0 * d->dshort;
    const double d01 = fabs(v1 - v0);
    const double d02 = fabs(v2 - v0);
    if (d01 < dd && d02 < dd) {
      return true;
    }
  }

  // full-distance history
  const double avg = damp_average(d);
  const double dev = avg * d->dlong;
  if (d->used == d->cap) {
    double min = d->hist[0];
    double max = min;
    for (u64 i = 1; i < d->cap; i++) {
      if (d->hist[i] < min) min = d->hist[i];
      if (d->hist[i] > max) max = d->hist[i];
    }
    if (fabs(max - min) < dev) {
      return true;
    }
  }

  if (d->events >= (d->cap * 2)) {
    return true;
  }
  return false;
}

  void
damp_clean(struct damp * const d)
{
  d->used = 0;
  d->next = 0;
  d->events = 0;
}

  void
damp_destroy(struct damp * const d)
{
  free(d);
}
// }}} damp

// vctr {{{
struct vctr {
  u64 nr;
  union {
    u64 v;
    au64 av;
  } u[];
};

  struct vctr *
vctr_create(const u64 nr)
{
  struct vctr * const v = (typeof(v))calloc(1, sizeof(*v) + (sizeof(v->u[0]) * nr));
  v->nr = nr;
  return v;
}

  inline u64
vctr_size(struct vctr * const v)
{
  return v->nr;
}

  inline void
vctr_add(struct vctr * const v, const u64 i, const u64 n)
{
  if (i < v->nr) {
    v->u[i].v += n;
  }
}

  inline void
vctr_add1(struct vctr * const v, const u64 i)
{
  if (i < v->nr) {
    v->u[i].v++;
  }
}

  inline void
vctr_add_atomic(struct vctr * const v, const u64 i, const u64 n)
{
  if (i < v->nr) {
    (void)atomic_fetch_add(&(v->u[i].av), n);
  }
}

  inline void
vctr_add1_atomic(struct vctr * const v, const u64 i)
{
  if (i < v->nr) {
    (void)atomic_fetch_add(&(v->u[i].av), UINT64_C(1));
  }
}

  inline void
vctr_set(struct vctr * const v, const u64 i, const u64 n)
{
  if (i < v->nr) {
    v->u[i].v = n;
  }
}

  u64
vctr_get(struct vctr * const v, const u64 i)
{
  if (i < v->nr) {
    return v->u[i].v;
  }
  return 0;
}

  void
vctr_merge(struct vctr * const to, const struct vctr * const from)
{
  const u64 nr = to->nr < from->nr ? to->nr : from->nr;
  for (u64 i = 0; i < nr; i++) {
    to->u[i].v += from->u[i].v;
  }
}

  void
vctr_reset(struct vctr * const v)
{
  for (u64 i = 0; i < v->nr; i++) {
    v->u[i].v = 0;
  }
}

  void
vctr_destroy(struct vctr * const v)
{
  free(v);
}
// }}} vctr

// rgen {{{
enum rgen_type {
  GEN_CONSTANT = 0, // always a constant number
  GEN_COUNTER,      // +1 each fetch
  GEN_COUNTER_UNSAFE, // +1 each fetch
  GEN_SKIPINC,      // +n each fetch
  GEN_SKIPINC_UNSAFE, // +n each fetch
  GEN_REDUCER,      // -1 each fetch
  GEN_REDUCER_UNSAFE,      // -1 each fetch
  GEN_EXPONENTIAL,  // exponential
  GEN_ZIPFIAN,      // Zipfian, 0 is the most popular.
  GEN_XZIPFIAN,     // ScrambledZipfian. scatters the "popular" items across the itemspace.
  GEN_UNIZIPF,      // Uniform + Zipfian
  GEN_UNIFORM,      // Uniformly distributed in an interval [a,b]
  GEN_TRACE32,      // Read from a trace file with unit of u32.
  GEN_TRACE64,      // Read from a trace file with unit of u64.
};

struct rgen_constant {
  u64 constant;
};

struct rgen_counter {
  union {
    au64 counter;
    u64 counter_unsafe;
  };
  u64 base;
  u64 modulus;
};

struct rgen_skipinc {
  union {
    au64 counter;
    u64 counter_unsafe;
  };
  u64 base;
  u64 modulus;
  u64 inc;
};

struct rgen_reducer {
  union {
    au64 counter;
    u64 counter_unsafe;
  };
  u64 base;
  u64 modulus;
};

struct rgen_exponential {
  double gamma;
};

struct rgen_trace32 {
  FILE * fin;
  u64 idx;
  u64 max;
  u32 buf[32];
};

#define ZIPFIAN_CONSTANT ((0.99))
struct rgen_zipfian {
  u64 nr_items;
  u64 base;
  u64 base1;
  double quick1;
  double nr_items_d;
  double zetan;
  double alpha;
  double quick2;
  double eta;
  double theta;
  u64 min;
  u64 max;
};

struct rgen_uniform {
  u64 min;
  u64 max;
  double interval;
};

struct rgen_unizipf {
  struct rgen_zipfian zipfian;
  u64 ufactor;
  u64 min;
  u64 max;
};

#define RGEN_ASYNC_BUFFER_NR ((UINT64_C(4)))
struct rgen_worker_info {
  union {
    u64 * buffers_u64[RGEN_ASYNC_BUFFER_NR];
    u32 * buffers_u32[RGEN_ASYNC_BUFFER_NR];
  };
  u8 * mem;
  u64 alloc_size;
  u64 padding10[8];
  abool avail[RGEN_ASYNC_BUFFER_NR];
  u64 padding20[8];
  u64 (*real_next)(struct rgen * const);
  u64 buffer_nr;
  u64 cpu;
  pthread_t thread;
  abool running;
  u64 padding30[8];
  u64 reader_id;
  u64 i;
};

struct rgen {
  u64 (*next_wait)(struct rgen * const);
  u64 (*next_nowait)(struct rgen * const);
  enum rgen_type type;
  bool unit_u64;
  bool async_worker;
  union {
    struct rgen_constant     constant;
    struct rgen_counter      counter;
    struct rgen_skipinc      skipinc;
    struct rgen_reducer      reducer;
    struct rgen_exponential  exponential;
    struct rgen_trace32      trace32;
    struct rgen_zipfian      zipfian;
    struct rgen_uniform      uniform;
    struct rgen_unizipf      unizipf;
  } gen;
  u64 padding[8];
  struct rgen_worker_info wi;
};

  inline u64
xorshift(const u64 seed)
{
  u64 x = seed ? seed : time_nsec();
  x ^= x >> 12; // a
  x ^= x << 25; // b
  x ^= x >> 27; // c
  return x * UINT64_C(2685821657736338717);
}

static __thread u64 __random_seed_u64 = 0;

  inline u64
random_u64(void)
{
  __random_seed_u64 = xorshift(__random_seed_u64);
  return __random_seed_u64;
}

  inline u64
srandom_u64(const u64 seed)
{
  __random_seed_u64 = xorshift(seed);
  return __random_seed_u64;
}

#define RAND64_MAX   ((UINT64_C(0xffffffffffffffff)))
#define RAND64_MAX_D ((double)(RAND64_MAX))

  inline double
random_double(void)
{
  // random between 0.0 - 1.0
  const u64 r = random_u64();
  return ((double)r) / RAND64_MAX_D;
}

  static u64
gen_exponential(struct rgen * const gi)
{
  const double d = - log(random_double()) / gi->gen.exponential.gamma;
  return (u64)d;
}

  struct rgen *
rgen_new_exponential(const double percentile, const double range)
{
  struct rgen * const gi = (typeof(gi))calloc(1, sizeof(*gi));
  gi->unit_u64 = true;
  gi->gen.exponential.gamma = - log(1.0 - (percentile/100.0)) / range;

  gi->type = GEN_EXPONENTIAL;
  gi->next_wait = gen_exponential;
  gi->next_nowait = gen_exponential;
  return gi;
}

  static u64
gen_constant(struct rgen * const gi)
{
  return gi->gen.constant.constant;
}

  struct rgen *
rgen_new_constant(const u64 constant)
{
  struct rgen * const gi = (typeof(gi))calloc(1, sizeof(*gi));
  gi->unit_u64 = constant > UINT32_MAX ? true : false;
  gi->gen.constant.constant = constant;

  gi->type = GEN_CONSTANT;
  gi->next_wait = gen_constant;
  gi->next_nowait = gen_constant;
  return gi;
}

  static u64
gen_counter(struct rgen * const gi)
{
  const u64 v = atomic_fetch_add(&(gi->gen.counter.counter), 1lu);
  return gi->gen.counter.base + (v % gi->gen.counter.modulus);
}

  struct rgen *
rgen_new_counter(const u64 min, const u64 max)
{
  struct rgen * const gi = (typeof(gi))calloc(1, sizeof(*gi));
  gi->unit_u64 = max > UINT32_MAX ? true : false;
  gi->gen.counter.counter = 0;
  gi->gen.counter.base = min;
  gi->gen.counter.modulus = max - min + 1;

  gi->type = GEN_COUNTER;
  gi->next_wait = gen_counter;
  gi->next_nowait = gen_counter;
  return gi;
}

  static u64
gen_counter_unsafe(struct rgen * const gi)
{
  const u64 v = gi->gen.counter.counter_unsafe;
  gi->gen.counter.counter++;
  return gi->gen.counter.base + (v % gi->gen.counter.modulus);
}

  struct rgen *
rgen_new_counter_unsafe(const u64 min, const u64 max)
{
  struct rgen * const gi = (typeof(gi))calloc(1, sizeof(*gi));
  gi->unit_u64 = max > UINT32_MAX ? true : false;
  gi->gen.counter.counter_unsafe = 0;
  gi->gen.counter.base = min;
  gi->gen.counter.modulus = max - min + 1;

  gi->type = GEN_COUNTER_UNSAFE;
  gi->next_wait = gen_counter_unsafe;
  gi->next_nowait = gen_counter_unsafe;
  return gi;
}

  static u64
gen_skipinc(struct rgen * const gi)
{
  const u64 v = atomic_fetch_add(&(gi->gen.skipinc.counter), gi->gen.skipinc.inc);
  return gi->gen.skipinc.base + (v % gi->gen.skipinc.modulus);
}

  struct rgen *
rgen_new_skipinc(const u64 min, const u64 max, const u64 inc)
{
  struct rgen * const gi = (typeof(gi))calloc(1, sizeof(*gi));
  gi->unit_u64 = max > UINT32_MAX ? true : false;
  gi->gen.skipinc.counter = 0;
  gi->gen.skipinc.base = min;
  gi->gen.skipinc.modulus = max - min + 1;
  gi->gen.skipinc.inc = inc;

  gi->type = GEN_SKIPINC;
  gi->next_wait = gen_skipinc;
  gi->next_nowait = gen_skipinc;
  return gi;
}

  static u64
gen_skipinc_unsafe(struct rgen * const gi)
{
  const u64 v = gi->gen.skipinc.counter_unsafe;
  gi->gen.skipinc.counter_unsafe += gi->gen.skipinc.inc;
  return gi->gen.skipinc.base + (v % gi->gen.skipinc.modulus);
}

  struct rgen *
rgen_new_skipinc_unsafe(const u64 min, const u64 max, const u64 inc)
{
  struct rgen * const gi = (typeof(gi))calloc(1, sizeof(*gi));
  gi->unit_u64 = max > UINT32_MAX ? true : false;
  gi->gen.skipinc.counter_unsafe = 0;
  gi->gen.skipinc.base = min;
  gi->gen.skipinc.modulus = max - min + 1;
  gi->gen.skipinc.inc = inc;

  gi->type = GEN_SKIPINC_UNSAFE;
  gi->next_wait = gen_skipinc_unsafe;
  gi->next_nowait = gen_skipinc_unsafe;
  return gi;
}

  static u64
gen_reducer(struct rgen * const gi)
{
  const u64 v = atomic_fetch_add(&(gi->gen.reducer.counter), 1lu);
  return gi->gen.reducer.base - (v % gi->gen.reducer.modulus);
}

  struct rgen *
rgen_new_reducer(const u64 min, const u64 max)
{
  struct rgen * const gi = (typeof(gi))calloc(1, sizeof(*gi));
  gi->unit_u64 = max > UINT32_MAX ? true : false;
  gi->gen.reducer.counter = 0;
  gi->gen.reducer.base = max;
  gi->gen.reducer.modulus = max - min + 1;

  gi->type = GEN_REDUCER;
  gi->next_wait = gen_reducer;
  gi->next_nowait = gen_reducer;
  return gi;
}

  static u64
gen_reducer_unsafe(struct rgen * const gi)
{
  const u64 v = gi->gen.reducer.counter_unsafe;
  gi->gen.reducer.counter_unsafe--;
  return gi->gen.reducer.base - (v % gi->gen.reducer.modulus);
}

  struct rgen *
rgen_new_reducer_unsafe(const u64 min, const u64 max)
{
  struct rgen * const gi = (typeof(gi))calloc(1, sizeof(*gi));
  gi->unit_u64 = max > UINT32_MAX ? true : false;
  gi->gen.reducer.counter_unsafe = 0;
  gi->gen.reducer.base = max;
  gi->gen.reducer.modulus = max - min + 1;

  gi->type = GEN_REDUCER_UNSAFE;
  gi->next_wait = gen_reducer_unsafe;
  gi->next_nowait = gen_reducer_unsafe;
  return gi;
}

  static u64
gen_zipfian(struct rgen * const gi)
{
  // simplified: no increamental update
  const struct rgen_zipfian * const gz = &(gi->gen.zipfian);
  const double u = random_double();
  const double uz = u * gz->zetan;
  if (uz < 1.0) {
    return gz->base;
  } else if (uz < gz->quick1) {
    return gz->base1;
  }
  //const double x = gz->nr_items_d * pow(gz->eta * (u - 1.0) + 1.0, gz->alpha);
  const double x = gz->nr_items_d * pow((gz->eta * u) + gz->quick2, gz->alpha);
  const u64 ret = gz->base + (u64)x;
  return ret;
}

struct zeta_range_info {
  au64 seq;
  u64 nth;
  u64 start;
  u64 count;
  double theta;

  double sum;
  spinlock lock;
  u64 pad[8];
};

  static void *
zeta_range_worker(void * const ptr)
{
  struct zeta_range_info * const zi = (typeof(zi))ptr;
  const u64 seq = atomic_fetch_add(&(zi->seq), 1);
  const u64 start = zi->start;
  const double theta = zi->theta;
  const u64 count = zi->count;
  const u64 nth = zi->nth;
  double local_sum = 0.0;
  for (u64 i = seq; i < count; i += nth) {
    local_sum += (1.0 / pow((double)(start + i + 1lu), theta));
  }
  spinlock_lock(&(zi->lock));
  zi->sum += local_sum;
  spinlock_unlock(&(zi->lock));
  return NULL;
}

  static double
zeta_range(const u64 start, const u64 count, const double theta)
{
  if (count < 0x10000) {
    // sequential
    double sum = 0.0;
    for (u64 i = 0lu; i < count; i++) {
      sum += (1.0 / pow((double)(start + i + 1lu), theta));
    }
    return sum;
  }

  const u64 nth = process_affinity_core_count();
  debug_assert(nth > 0);
  struct zeta_range_info zi;
  atomic_init(&(zi.seq), 0);
  zi.nth = nth;
  zi.start = start;
  zi.count = count;
  zi.theta = theta;
  zi.sum = 0.0;
  spinlock_init(&(zi.lock));
  thread_fork_join(nth, zeta_range_worker, &zi);
  return zi.sum;
}

static const u64 zetalist_u64[] = {0,
  UINT64_C(0x4040437dd948c1d9), UINT64_C(0x4040b8f8009bce85),
  UINT64_C(0x4040fe1121e564d6), UINT64_C(0x40412f435698cdf5),
  UINT64_C(0x404155852507a510), UINT64_C(0x404174d7818477a7),
  UINT64_C(0x40418f5e593bd5a9), UINT64_C(0x4041a6614fb930fd),
  UINT64_C(0x4041bab40ad5ec98), UINT64_C(0x4041cce73d363e24),
  UINT64_C(0x4041dd6239ebabc3), UINT64_C(0x4041ec715f5c47be),
  UINT64_C(0x4041fa4eba083897), UINT64_C(0x4042072772fe12bd),
  UINT64_C(0x4042131f5e380b72), UINT64_C(0x40421e53630da013),
};

static const double * zetalist_double = (typeof(zetalist_double))zetalist_u64;
static const u64 zetalist_step = UINT64_C(0x10000000000);
static const u64 zetalist_count = 16;
//static const double zetalist_theta = 0.99;

  static double
zeta(const u64 n, const double theta)
{
  //assert(theta == 0.99);
  const u64 zlid0 = n / zetalist_step;
  const u64 zlid = (zlid0 > zetalist_count) ? zetalist_count : zlid0;
  const double sum0 = zetalist_double[zlid];
  const u64 start = zlid * zetalist_step;
  const u64 count = n - start;
  const double sum1 = zeta_range(start, count, theta);
  return sum0 + sum1;
}

  struct rgen *
rgen_new_zipfian(const u64 min, const u64 max)
{
  struct rgen * const gi = (typeof(gi))calloc(1, sizeof(*gi));
  gi->unit_u64 = max > UINT32_MAX ? true : false;
  struct rgen_zipfian * const gz = &(gi->gen.zipfian);

  const u64 items = max - min + 1;
  gz->nr_items = items;
  gz->nr_items_d = (double)items;
  gz->base = min;
  gz->base1 = min + 1lu;
  gz->theta = ZIPFIAN_CONSTANT;
  gz->quick1 = 1.0 + pow(0.5, gz->theta);
  const double zeta2theta = zeta(2, ZIPFIAN_CONSTANT);
  gz->alpha = 1.0 / (1.0 - ZIPFIAN_CONSTANT);
  const double zetan = zeta(items, ZIPFIAN_CONSTANT);
  gz->zetan = zetan;
  gz->eta = (1.0 - pow(2.0 / (double)items, 1.0 - ZIPFIAN_CONSTANT)) / (1.0 - (zeta2theta / zetan));
  gz->quick2 = 1.0 - gz->eta;
  gz->min = min;
  gz->max = max;

  gi->type = GEN_ZIPFIAN;
  gi->next_wait = gen_zipfian;
  gi->next_nowait = gen_zipfian;
  return gi;
}

  static u64
gen_xzipfian(struct rgen * const gi)
{
  const u64 z = gen_zipfian(gi);
  const u64 xz = gi->gen.zipfian.min + (xxhash64(&z, sizeof(z)) % gi->gen.zipfian.nr_items);
  return xz;
}

  struct rgen *
rgen_new_xzipfian(const u64 min, const u64 max)
{
  struct rgen * gi = rgen_new_zipfian(min, max);
  gi->unit_u64 = max > UINT32_MAX ? true : false;
  gi->type = GEN_XZIPFIAN;
  gi->next_wait = gen_xzipfian;
  gi->next_nowait = gen_xzipfian;
  return gi;
}

  static u64
gen_unizipf(struct rgen * const gi)
{
  const u64 z = gen_zipfian(gi) * gi->gen.unizipf.ufactor;
  const u64 u = random_u64() % gi->gen.unizipf.ufactor;
  return gi->gen.unizipf.min + z + u;
}

  struct rgen *
rgen_new_unizipf(const u64 min, const u64 max, const u64 ufactor)
{
  const u64 nr = max - min + 1;
  if (ufactor == 1lu) { // covers both special gens
    return rgen_new_zipfian(min, max);
  } else if ((ufactor == 0) || ((nr / ufactor) <= 1lu)) {
    return rgen_new_uniform(min, max);
  }
  const u64 znr = nr / ufactor;
  struct rgen * gi = rgen_new_zipfian(0, znr - 1);
  gi->unit_u64 = max > UINT32_MAX ? true : false;
  gi->gen.unizipf.ufactor = ufactor;
  gi->gen.unizipf.min = min;
  gi->gen.unizipf.max = max;
  gi->next_wait = gen_unizipf;
  gi->next_nowait = gen_unizipf;
  return gi;
}

  static u64
gen_uniform(struct rgen * const gi)
{
  const u64 off = (u64)(random_double() * gi->gen.uniform.interval);
  return gi->gen.uniform.min + off;
}

  struct rgen *
rgen_new_uniform(const u64 min, const u64 max)
{
  struct rgen * const gi = (typeof(gi))calloc(1, sizeof(*gi));
  gi->unit_u64 = max > UINT32_MAX ? true : false;
  gi->gen.uniform.min = min;
  gi->gen.uniform.max = max;
  gi->gen.uniform.interval = (double)(max - min);

  gi->type = GEN_UNIFORM;
  gi->next_wait = gen_uniform;
  gi->next_nowait = gen_uniform;
  return gi;
}

  static u64
gen_trace32(struct rgen * const gi)
{
  if (gi->gen.trace32.idx >= gi->gen.trace32.max) {
    if (feof(gi->gen.trace32.fin)) rewind(gi->gen.trace32.fin);
    gi->gen.trace32.idx = 0;
    gi->gen.trace32.max = fread(gi->gen.trace32.buf, sizeof(u32), 32, gi->gen.trace32.fin);
  }
  const u64 r = gi->gen.trace32.buf[gi->gen.trace32.idx];
  gi->gen.trace32.idx++;
  return r;
}

  struct rgen *
rgen_new_trace32(const char * const filename)
{
  struct rgen * const gi = (typeof(gi))calloc(1, sizeof(*gi));
  gi->gen.trace32.fin = fopen(filename, "rb");
  if (gi->gen.trace32.fin == NULL) {
    free(gi);
    return NULL;
  }
  gi->gen.trace32.idx = 0;
  gi->gen.trace32.max = fread(gi->gen.trace32.buf, sizeof(u32), 32, gi->gen.trace32.fin);
  if (gi->gen.trace32.max == 0) {
    free(gi);
    return NULL;
  }
  posix_fadvise(fileno(gi->gen.trace32.fin), 0, 0, POSIX_FADV_SEQUENTIAL);
  gi->type = GEN_TRACE32;
  gi->next_wait = gen_trace32;
  gi->next_nowait = gen_trace32;
  return gi;
}

  inline u64
rgen_next_wait(struct rgen * const geninfo)
{
  return geninfo->next_wait(geninfo);
}

  inline u64
rgen_next_nowait(struct rgen * const geninfo)
{
  return geninfo->next_nowait(geninfo);
}

  static inline void
rgen_clean_async_buffers(struct rgen_worker_info * const info)
{
  if (info->mem == NULL) return;
  pages_unmap(info->mem, info->alloc_size);
  info->mem = NULL;
  for (u64 j = 0; j < RGEN_ASYNC_BUFFER_NR; j++) {
    info->buffers_u64[j] = NULL;
  }
}

  void
rgen_destroy(struct rgen * const gen)
{
  if (gen == NULL) return;
  if (gen->async_worker) {
    struct rgen_worker_info * const info = &(gen->wi);
    atomic_store(&(info->running), false);
    pthread_join(info->thread, NULL);
    rgen_clean_async_buffers(info);
  }
  if (gen->type == GEN_TRACE32) {
    fclose(gen->gen.trace32.fin);
  }
  free(gen);
}

  void
rgen_helper_message(void)
{
  fprintf(stderr, "%s Usage: rgen <type> ...\n", __func__);
  fprintf(stderr, "%s example: rgen unizipf <min> <max> <ufactor>\n", __func__);
  fprintf(stderr, "%s example: rgen uniform <min> <max>\n", __func__);
  fprintf(stderr, "%s example: rgen zipfian <min> <max>\n", __func__);
  fprintf(stderr, "%s example: rgen counter <min> <max>\n", __func__);
  fprintf(stderr, "%s example: rgen reducer <min> <max>\n", __func__);
  fprintf(stderr, "%s example: rgen skipinc <min> <max> <inc>\n", __func__);
  fprintf(stderr, "%s example: rgen trace32 <filename>\n", __func__);
}

  int
rgen_helper(const int argc, char ** const argv, struct rgen ** const gen_out)
{
  *gen_out = NULL;
  if ((argc < 1) || (strcmp("rgen", argv[0]) != 0)) {
    return -1;
  }
  if (0 == strcmp(argv[1], "unizipf")) {
    if (argc < 5) return -1;
    const u64 min = strtoull(argv[2], NULL, 10);
    const u64 max = strtoull(argv[3], NULL, 10);
    const u64 uf = strtoull(argv[4], NULL, 10);
    struct rgen * const gen = rgen_new_unizipf(min, max, uf);
    if (gen == NULL) return -1;
    *gen_out = gen;
    return 5;
  } else if (0 == strcmp(argv[1], "uniform")) {
    if (argc < 4) return -1;
    const u64 min = strtoull(argv[2], NULL, 10);
    const u64 max = strtoull(argv[3], NULL, 10);
    struct rgen * const gen = rgen_new_uniform(min, max);
    if (gen == NULL) return -1;
    *gen_out = gen;
    return 4;
  } else if (0 == strcmp(argv[1], "zipfian")) {
    if (argc < 4) return -1;
    const u64 min = strtoull(argv[2], NULL, 10);
    const u64 max = strtoull(argv[3], NULL, 10);
    struct rgen * const gen = rgen_new_zipfian(min, max);
    if (gen == NULL) return -1;
    *gen_out = gen;
    return 4;
  } else if (0 == strcmp(argv[1], "counter")) {
    if (argc < 4) return -1;
    const u64 min = strtoull(argv[2], NULL, 10);
    const u64 max = strtoull(argv[3], NULL, 10);
    struct rgen * const gen = rgen_new_counter(min, max);
    if (gen == NULL) return -1;
    *gen_out = gen;
    return 4;
  } else if (0 == strcmp(argv[1], "reducer")) {
    if (argc < 4) return -1;
    const u64 min = strtoull(argv[2], NULL, 10);
    const u64 max = strtoull(argv[3], NULL, 10);
    struct rgen * const gen = rgen_new_reducer(min, max);
    if (gen == NULL) return -1;
    *gen_out = gen;
    return 4;
  } else if (0 == strcmp(argv[1], "skipinc")) {
    if (argc < 5) return -1;
    const u64 min = strtoull(argv[2], NULL, 10);
    const u64 max = strtoull(argv[3], NULL, 10);
    const u64 inc = strtoull(argv[4], NULL, 10);
    struct rgen * const gen = rgen_new_skipinc(min, max, inc);
    if (gen == NULL) return -1;
    *gen_out = gen;
    return 5;
  } else if (0 == strcmp(argv[1], "trace32")) {
    if (argc < 3) return -1;
    struct rgen * const gen = rgen_new_trace32(argv[2]);
    if (gen == NULL) return -1;
    *gen_out = gen;
    return 3;
  }
  return -1;
}

  static void *
rgen_worker(void * const ptr)
{
  struct rgen * const gen = (typeof(gen))ptr;
  struct rgen_worker_info * const info = &(gen->wi);
  const u64 cpu = info->cpu;
  srandom_u64((cpu + 3) * 97);
  const u64 nr = info->buffer_nr;
  while (true) {
    for (u64 i = 0; i < RGEN_ASYNC_BUFFER_NR; i++) {
      while (atomic_load(&(info->avail[i])) == true) {
        usleep(10);
        if (atomic_load(&(info->running)) == false) {
          return NULL;
        }
      }
      if (gen->unit_u64) {
        u64 * const buf64 = info->buffers_u64[i];
        for (u64 j = 0; j < nr; j++) {
          buf64[j] = info->real_next(gen);
        }
      } else {
        u32 * const buf32 = info->buffers_u32[i];
        for (u64 j = 0; j < nr; j++) {
          buf32[j] = (u32)(info->real_next(gen));
        }
      }
      atomic_thread_fence(memory_order_seq_cst);
      atomic_store(&(info->avail[i]), true);
    }
  }
  return NULL;
}

  static inline void
rgen_async_wait_at(struct rgen * const gen, const u64 id)
{
  abool * const pavail = &(gen->wi.avail[id]);
  while (atomic_load(pavail) == false) usleep(1);

}

  inline void
rgen_async_wait(struct rgen * const gen)
{
  if (gen->async_worker == false) return;
  rgen_async_wait_at(gen, gen->wi.reader_id);
}

  inline void
rgen_async_wait_all(struct rgen * const gen)
{
  if (gen->async_worker == false) return;
  for (u64 i = 0; i < RGEN_ASYNC_BUFFER_NR; i++) {
    rgen_async_wait_at(gen, i);
  }
}

  static u64
rgen_async_next_wait(struct rgen * const gen)
{
  struct rgen_worker_info * const info = &(gen->wi);
  u64 i = info->i;
  if (i == 0) rgen_async_wait(gen);

  const u64 id0 = info->reader_id;
  const u64 r = gen->unit_u64 ? info->buffers_u64[id0][i] : info->buffers_u32[id0][i];
  i++;
  if (i >= info->buffer_nr) {
    atomic_store(&(info->avail[id0]), false);
    info->reader_id = (id0 + 1) % RGEN_ASYNC_BUFFER_NR;
    i = 0;
  }
  info->i = i;
  return r;
}

  static u64
rgen_async_next_nowait(struct rgen * const gen)
{
  struct rgen_worker_info * const info = &(gen->wi);
  u64 i = info->i;

  const u64 id0 = info->reader_id;
  const u64 r = gen->unit_u64 ? info->buffers_u64[id0][i] : info->buffers_u32[id0][i];
  i++;
  if (i >= info->buffer_nr) {
    info->reader_id = (id0 + 1) % RGEN_ASYNC_BUFFER_NR;
    i = 0;
  }
  info->i = i;
  return r;
}

  struct rgen *
rgen_dup(struct rgen * const gen0)
{
  if (gen0->async_worker) {
    return NULL;
  }
  struct rgen * const gen = (typeof(gen))malloc(sizeof(*gen));
  memcpy(gen, gen0, sizeof(*gen));
  struct rgen_worker_info * const info = &(gen->wi);
  memset(info, 0, sizeof(*info));
  if (gen->type == GEN_TRACE32) {
    FILE * const f2 = fdopen(dup(fileno(gen0->gen.trace32.fin)), "rb");
    posix_fadvise(fileno(f2), 0, 0, POSIX_FADV_SEQUENTIAL);
    gen->gen.trace32.fin = f2;
    gen->gen.trace32.idx = 0;
    gen->gen.trace32.max = 0;
  }
  return gen;
}

  bool
rgen_async_convert(struct rgen * const gen, const u64 cpu)
{
  if (gen->async_worker) {
    return false; // already converted
  }
  struct rgen_worker_info * const info = &(gen->wi);
  memset(info, 0, sizeof(*info));
  info->real_next = gen->next_wait;
  const u64 g = UINT64_C(1) << 30;
  const u64 usize = gen->unit_u64 ? sizeof(u64) : sizeof(u32);
  info->mem = pages_alloc_best(g, true, &(info->alloc_size));
  info->buffer_nr = g / (RGEN_ASYNC_BUFFER_NR * usize);
  info->cpu = cpu;
  atomic_store(&(info->running), true);
  for (u64 j = 0; j < RGEN_ASYNC_BUFFER_NR; j++) {
    info->buffers_u64[j] = (u64 *)(&(info->mem[j * info->buffer_nr * usize]));
    atomic_store(&(info->avail[j]), false);
  }
  gen->next_wait = rgen_async_next_wait;
  gen->next_nowait = rgen_async_next_nowait;
  if (thread_create_at(cpu, &(info->thread), rgen_worker, gen) == 0) {
    char thname[32];
    sprintf(thname, "rgen_async_%"PRIu64, cpu);
    pthread_setname_np(info->thread, thname);
    info->reader_id = 0;
    info->i = 0;
    gen->async_worker = true;
    return true;
  } else {
    rgen_clean_async_buffers(info);
    return false;
  }
}
// }}} rgen

// rcu {{{
// bits 63 -- 16 valud (pointer)
// bits 15       valid
// bits 14 -- 0  count (refcount)
#define RCU_COUNT_MASK   ((UINT64_C(0x7fff)))
#define RCU_VALID_MASK   ((UINT64_C(0x8000)))
#define RCU_VALUE_MASK   ((UINT64_C(0xffffffffffff0000)))
#define RCU_VALUE_SHIFT  ((16))
struct rcu_node {
  au64 x[2];
};

  void
rcu_node_init(struct rcu_node * const node)
{
  atomic_store(&(node->x[0]), RCU_VALID_MASK); // valid null pointer
  atomic_store(&(node->x[1]), RCU_VALUE_MASK); // invalid non-null pointer
}

  struct rcu_node *
rcu_node_create(void)
{
  struct rcu_node * const node = (typeof(node))malloc(sizeof(*node));
  rcu_node_init(node);
  return node;
}

  void *
rcu_read_ref(struct rcu_node * const node)
{
  do {
    for (u64 i = 0; i < 2; i++) {
      u64 x = node->x[i];
      if (x & RCU_VALID_MASK) {
        const bool r = atomic_compare_exchange_strong(&(node->x[i]), &x, x + 1);
        if (r == true) {
          return (void *)(x >> RCU_VALUE_SHIFT);
        }
      }
    }
  } while (true);
}

  void
rcu_read_unref(struct rcu_node * const node, void * ptr)
{
  for (u64 i = 0; i < 2; i++) {
    const u64 x = node->x[i];
    if ((x >> RCU_VALUE_SHIFT) == ((u64)ptr)) {
      bool r = false;
      do {
        u64 xx = node->x[i];
        debug_assert(xx & RCU_COUNT_MASK);
        r = atomic_compare_exchange_strong(&(node->x[i]), &xx, xx - 1);
      } while (r == false);
      return;
    }
  }
  debug_assert(false);
}

  static void
__rcu_gc(au64 * const px)
{
  bool r = false;
  do {
    u64 x0 = *px;
    const u64 x1 = x0 & (~RCU_VALID_MASK);
    r = atomic_compare_exchange_strong(px, &x0, x1);
  } while (r == false);
  atomic_thread_fence(memory_order_seq_cst);
  while (atomic_load(px) & RCU_COUNT_MASK);
}

  void
rcu_update(struct rcu_node * const node, void * ptr)
{
  for (u64 i = 0; i < 2; i++) {
    const u64 x = node->x[i];
    if ((x & RCU_VALID_MASK) == 0) {
      debug_assert((node->x[1-i] >> RCU_VALUE_SHIFT) != ((u64)ptr));
      const u64 xx = (((u64)ptr) << RCU_VALUE_SHIFT) | RCU_VALID_MASK;
      atomic_store(&(node->x[i]), xx);
      atomic_thread_fence(memory_order_seq_cst);
      __rcu_gc(&(node->x[1-i]));
      return;
    }
  }
  debug_assert(false);
}
// }}} rcu

// server {{{
struct server {
  int fd;
  bool running;
  pthread_t pt;
  void *(*worker)(void *);
  void * priv;
};

struct server_wi {
  struct server * server;
  struct stream2 s2;
  void * priv;
};

  static void *
__server_master(void * const ptr)
{
  const u64 nr_cores = process_affinity_core_count();
  u64 * const cores = (typeof(cores))calloc(nr_cores, sizeof(cores[0]));
  const u64 nr_cores1 = process_affinity_core_list(nr_cores, cores);
  u64 next_core_id = 0;

  struct server * const server = (typeof(server))ptr;
  while (server->running) {
    const int cli = accept(server->fd, NULL, NULL);
    if (cli < 0) {
      if (errno == EINTR) continue;
      break;
    }
    pthread_t worker;
    struct server_wi * const wi = (typeof(wi))malloc(sizeof(*wi));
    wi->server = server;
    wi->s2.r = fdopen(dup(cli), "rb");
    wi->s2.w = fdopen(cli, "wb");
    wi->priv = server->priv;
    const int r = thread_create_at(cores[next_core_id], &worker, server->worker, wi);
    if (r != 0) {
      fclose(wi->s2.r);
      fclose(wi->s2.w);
      free(wi);
    } else {
      next_core_id = (next_core_id + 1) % nr_cores1;
    }
  }
  close(server->fd);
  return NULL;
}

  struct server *
server_create(const char * const host, const int port, void*(*worker)(void*), void * const priv)
{
  char portstr[6];
  snprintf(portstr, 6, "%d", port);
  struct addrinfo info0 = {.ai_family = AF_INET, .ai_socktype = SOCK_STREAM};
  struct addrinfo *info = NULL;
  if (getaddrinfo(host, portstr, &info0, &info) != 0) return NULL;
  int fd = -1;

  for (struct addrinfo * p = info; p != NULL; p = p->ai_next) {
    fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd < 0) continue;
    int one = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (bind(fd, p->ai_addr, p->ai_addrlen) != 0) break;
    if (listen(fd, 64) != 0) break;
    struct server * const server = malloc(sizeof(*server));
    server->running = true;
    server->fd = fd;
    server->worker = worker;
    server->priv = priv;
    if (pthread_create(&(server->pt), NULL, __server_master, server) != 0) {
      free(server);
      break;
    }
    freeaddrinfo(info);
    return server;
  }
  if (fd >= 0) {
    close(fd);
  }
  if (info) {
    freeaddrinfo(info);
  }
  return NULL;
}

  void
server_wait(struct server * const server)
{
  pthread_join(server->pt, NULL);
}

  void
server_destroy(struct server * const server)
{
  server->running = false;
  shutdown(server->fd, SHUT_RDWR);
  pthread_join(server->pt, NULL);
  free(server);
}

  struct stream2 *
server_wi_stream2(struct server_wi * const wi)
{
  return &(wi->s2);
}

  void *
server_wi_private(struct server_wi * const wi)
{
  return wi->priv;
}

  void
server_wi_destroy(struct server_wi * const wi)
{
  fclose(wi->s2.w);
  fclose(wi->s2.r);
  free(wi);
}

  struct stream2 *
stream2_create(const char * const host, const int port)
{
  char portstr[6];
  snprintf(portstr, 6, "%d", port);
  struct addrinfo info0 = {.ai_family = AF_INET, .ai_socktype = SOCK_STREAM};
  struct addrinfo *info = NULL;
  if (getaddrinfo(host, portstr, &info0, &info) != 0) return NULL;

  for (struct addrinfo * p = info; p != NULL; p = p->ai_next) {
    const int fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd < 0) continue;
    int one = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (connect(fd, p->ai_addr, p->ai_addrlen) != 0) {
      close(fd);
      continue;
    }
    freeaddrinfo(info);
    struct stream2 * const stream2 = (typeof(stream2))malloc(sizeof(*stream2));
    stream2->w = fdopen(dup(fd), "wb");
    stream2->r= fdopen(fd, "rb");
    return stream2;
  }
  if (info) freeaddrinfo(info);
  return NULL;
}

  void
stream2_destroy(struct stream2 * const stream2)
{
  fclose(stream2->w);
  fclose(stream2->r);
  free(stream2);
}
// }}} server

// kv {{{
  inline size_t
kv_size(const struct kv * const kv)
{
  return sizeof(*kv) + kv->klen + kv->vlen;
}

  inline size_t
kv_size_align(const struct kv * const kv, const u64 align)
{
  debug_assert(align && ((align & (align - 1)) == 0));
  return (sizeof(*kv) + kv->klen + kv->vlen + (align - 1)) & (~(align - 1));
}

  inline size_t
key_size(const struct kv *const key)
{
  return sizeof(*key) + key->klen;
}

  inline size_t
key_size_align(const struct kv *const key, const u64 align)
{
  debug_assert(align && ((align & (align - 1)) == 0));
  return (sizeof(*key) + key->klen + (align - 1)) & (~(align - 1));
}

  inline void
kv_update_hash(struct kv * const kv)
{
  kv->hash = xxhash64((const void *)kv->kv, (const size_t)kv->klen);
}

  inline void
kv_refill(struct kv * const kv, const void * const key, const u32 klen, const void * const value, const u32 vlen)
{
  if (kv) {
    kv->klen = klen;
    kv->vlen = vlen;
    memcpy(&(kv->kv[0]), key, klen);
    memcpy(&(kv->kv[klen]), value, vlen);
    kv_update_hash(kv);
  }
}

  inline void
kv_refill_str(struct kv * const kv, const char * const key, const char * const value)
{
  kv_refill(kv, key, (u32)strlen(key), value, (u32)strlen(value));
}

  inline struct kv *
kv_create(const void * const key, const u32 klen, const void * const value, const u32 vlen)
{
  struct kv * const kv = malloc(sizeof(*kv) + klen + vlen);
  kv_refill(kv, key, klen, value, vlen);
  return kv;
}

  inline struct kv *
kv_create_str(const char * const key, const char * const value)
{
  return kv_create(key, (u32)strlen(key), value, (u32)strlen(value));
}

  inline struct kv *
kv_dup(const struct kv * const kv)
{
  if (kv == NULL) return NULL;

  const size_t sz = kv_size(kv);
  struct kv * const new = (typeof(new))malloc(sz);
  if (new) {
    memcpy(new, kv, sz);
  }
  return new;
}

  inline struct kv *
kv_dup_key(const struct kv * const kv)
{
  if (kv == NULL) return NULL;

  const size_t sz = key_size(kv);
  struct kv * const new = (typeof(new))malloc(sz);
  if (new) {
    memcpy(new, kv, sz);
  }
  return new;
}

  inline struct kv *
kv_dup2(const struct kv * const from, struct kv * const to)
{
  if (from == NULL) return NULL;

  const size_t sz = kv_size(from);
  if (to) {
    memcpy(to, from, sz);
    return to;
  } else {
    struct kv * const new = (typeof(new))malloc(sz);
    if (new) {
      memcpy(new, from, sz);
    }
    return new;
  }
}

  inline struct kv *
kv_dup2_key(const struct kv * const from, struct kv * const to)
{
  if (from == NULL) return NULL;

  const size_t sz = key_size(from);
  if (to) {
    memcpy(to, from, sz);
    return to;
  } else {
    struct kv * const new = (typeof(new))malloc(sz);
    if (new) {
      memcpy(new, from, sz);
    }
    return new;
  }
}

  struct kv *
kv_alloc_malloc(const u64 size, void * const priv)
{
  (void)priv;
  return (struct kv *)malloc(size);
}

  void
kv_retire_free(struct kv * const kv, void * const priv)
{
  (void)priv;
  free(kv);
}

  static inline struct kv *
kv_alloc_dup(const struct kv * const kv, const struct kvmap_mm * const mm)
{
  const u64 sz = kv_size(kv);
  struct kv * const new = mm->af(sz, mm->ap);
  debug_assert(new);
  memcpy(new, kv, sz);
  return new;
}

  inline bool
kv_keymatch(const struct kv * const key1, const struct kv * const key2)
{
  if ((!key1) || (!key2) || (key1->hash != key2->hash) || (key1->klen != key2->klen)) {
    return false;
  }
  const int cmp = memcmp(key1->kv, key2->kv, (size_t)(key1->klen));
  return (cmp == 0) ? true : false;
}

  inline bool
kv_fullmatch(const struct kv * const kv1, const struct kv * const kv2)
{
  if ((!kv1) || (!kv2) || (kv1->hash != kv2->hash) || (kv1->klen != kv2->klen) || (kv1->vlen != kv2->vlen)) {
    return false;
  }
  const int cmp = memcmp(kv1->kv, kv2->kv, (size_t)(kv1->klen + kv1->vlen));
  return (cmp == 0) ? true : false;
}

  int
kv_keycompare(const struct kv * const kv1, const struct kv * const kv2)
{
  if (kv1 == NULL) {
    return (kv2 == NULL) ? 0 : -1;
  } else if (kv2 == NULL) {
    return 1;
  }
  const u32 len = kv1->klen < kv2->klen ? kv1->klen : kv2->klen;
  const int cmp = memcmp(kv1->kv, kv2->kv, (size_t)len);
  if (cmp == 0) {
    if (kv1->klen < kv2->klen) {
      return -1;
    } else if (kv1->klen > kv2->klen) {
      return 1;
    } else {
      return 0;
    }
  } else {
    return cmp;
  }
}

  static int
__kv_compare_pp(const void * const p1, const void * const p2)
{
  const struct kv ** const pp1 = (typeof(pp1))p1;
  const struct kv ** const pp2 = (typeof(pp2))p2;
  return kv_keycompare(*pp1, *pp2);
}

  inline void
kv_qsort(const struct kv ** const kvs, const size_t nr)
{
  qsort(kvs, nr, sizeof(kvs[0]), __kv_compare_pp);
}

  inline void *
kv_value_ptr(struct kv * const kv)
{
  return (void *)(&(kv->kv[kv->klen]));
}

  inline void *
kv_key_ptr(struct kv * const kv)
{
  return (void *)(&(kv->kv[0]));
}

  inline const void *
kv_value_ptr_const(const struct kv * const kv)
{
  return (const void *)(&(kv->kv[kv->klen]));
}

  inline const void *
kv_key_ptr_const(const struct kv * const kv)
{
  return (const void *)(&(kv->kv[0]));
}
// }}} kv

// kvmap {{{
struct __kvmap_entry {
  u64 pkey:16;
  u64 ptr:48;
};

struct __kvmap_slot {
  struct __kvmap_entry e[8];
};

#define PGSZ ((UINT64_C(4096)))
#define KVMAP_BASE_LEVEL ((6))
#define KVMAP_HLOCK_POWER ((10))
#define KVMAP_HLOCK_NR ((UINT64_C(1) << KVMAP_HLOCK_POWER))
#define KVMAP_HLOCK_MASK ((KVMAP_HLOCK_NR - 1))
#define KVMAP_MIN_LEVEL ((KVMAP_BASE_LEVEL + KVMAP_HLOCK_POWER))
#define KVMAP_SLOTS_OF(__lid__) ((UINT64_C(1) << (__lid__)))
#define KVMAP_ENTRIES_OF(__lid__) ((UINT64_C(8) << (__lid__)))
#define KVMAP_PAGES_OF(__lid__) ((UINT64_C(1) << ((__lid__) - UINT64_C(6))))
#define KVMAP_SIZE_OF_LEVEL(__lid__) ((KVMAP_PAGES_OF(__lid__) * PGSZ))

  static inline u64
kvmap_pkey(const u64 hash)
{
  const u64 pkey1 = hash ^ (hash >> 32);
  const u64 pkey2 = pkey1 ^ (pkey1 >> 16);
  return pkey2 & UINT64_C(0xffff);
}

  static inline struct kv *
__u64_to_ptr(const u64 v)
{
  return (struct kv *)v;
}

  static inline u64
__ptr_to_u64(const struct kv * const ptr)
{
  return (u64)ptr;
}

  static inline void
kvmap_put_entry(struct kvmap_mm * const mm, struct __kvmap_entry * const e, const struct kv * const kv)
{
  struct kv * const old = __u64_to_ptr(e->ptr);
  if (old) {
    if (kv && mm->xf) {
      mm->xf(old, kv, mm->xp);
    } else if (mm->rf) {
      mm->rf(old, mm->rp);
    }
  }
  e->ptr = __ptr_to_u64(kv);
  e->pkey = kv ? kvmap_pkey(kv->hash) : 0;
}

  inline void
kvmap_api_destroy(struct kvmap_api * const api)
{
  debug_assert(api);
  api->destroy(api->map);
  free(api);
}

static const struct kvmap_mm __kvmap_mm_default = {
  .af = kv_alloc_malloc,
  .ap = NULL,
  .rf = kv_retire_free,
  .rp = NULL,
  .hf = NULL,
  .hp = NULL,
};

  void
kvmap_api_helper_message(void)
{
  fprintf(stderr, "%s Usage: api <cache-mb> <map-type> <param1> ...\n", __func__);
  fprintf(stderr, "%s example: api 0 kvmap2\n", __func__);
  fprintf(stderr, "%s example: api 0 cuckoo\n", __func__);
  fprintf(stderr, "%s example: api 0 skiplist\n", __func__);
  fprintf(stderr, "%s example: api 0 chainmap\n", __func__);
  fprintf(stderr, "%s example: api 0 bptree\n", __func__);
  fprintf(stderr, "%s example: api 0 mica 20\n", __func__);
  fprintf(stderr, "%s example: api 0 tcpmap <host> <port>\n", __func__);
}

  static int
kvmap_api_helper_map(int argc, char ** const argv, struct kvmap_api ** const out, struct kvmap_mm * const mm)
{
  if (argc < 1) {
    return -1;
  }
  struct kvmap_api * tmp = NULL;
  int used = 0;
  if (strcmp("kvmap2", argv[0]) == 0) {
    tmp = kvmap2_api_create(kvmap2_create(mm));
    used = 1;
  } else if (strcmp("cuckoo", argv[0]) == 0) {
    tmp = cuckoo_api_create(cuckoo_create(mm));
    used = 1;
  } else if (strcmp("skiplist", argv[0]) == 0) {
    tmp = skiplist_api_create(skiplist_create(mm));
    used = 1;
  } else if (strcmp("chainmap", argv[0]) == 0) {
    tmp = chainmap_api_create(chainmap_create(mm));
    used = 1;
  } else if (strcmp("bptree", argv[0]) == 0) {
    tmp = bptree_api_create(bptree_create(mm));
    used = 1;
  } else if (strcmp("mica", argv[0]) == 0) {
    if (argc < 2) return -1;
    tmp = mica_api_create(mica_create(mm, strtoull(argv[1], NULL, 10)));
    used = 2;
  } else if (strcmp("tcpmap", argv[0]) == 0) {
    if (argc < 3) return -1;
    tmp = tcpmap_api_create(tcpmap_create(argv[1], atoi(argv[2])));
    used = 3;
  }
  if (used > 0 && tmp) {
    *out = tmp;
    return used;
  } else {
    return -1;
  }
}

  int
kvmap_api_helper(int argc, char ** const argv, struct kvmap_api ** const out, struct kvmap_mm * const mm, const bool use_ucache)
{
  if (argc < 3 || strcmp(argv[0], "api") != 0) {
    return -1;
  }
  const u64 mb = strtoull(argv[1], NULL, 10);
  if (mb == 0) {
    // icache
    return 2 + kvmap_api_helper_map(argc - 2, argv + 2, out, mm);
  }

  // no icache
  struct icache * const cache = icache_create(NULL, mb);
  if (cache == NULL) {
    fprintf(stderr, "icache_create() failed\n");
    return -1;
  }
  icache_wrap_mm(cache, mm);
  struct kvmap_api * map_api = NULL;
  const int n1 = kvmap_api_helper_map(argc - 2, argv + 2, &map_api, mm);
  if (n1 < 0) {
    icache_destroy(cache);
    return -1;
  }
  struct kvmap_api * const icache_api = use_ucache ? ucache_api_create(cache) : icache_api_create(cache);
  if (icache_api == NULL) {
    kvmap_api_destroy(map_api);
    icache_destroy(cache);
    return -1;
  }
  icache_wrap_kvmap(cache, map_api);
  *out = icache_api;
  return 2 + n1;
}
// }}} kvmap

// kvmap1 {{{
struct kvmap1 {
  struct kvmap_mm mm;
  u64 hash_rotate; // const

  u64 max_level; // current max level
  u64 min_level; // current min level
  u64 first_check;
  u64 padding20[7];
  au64 nr_kv;
  u64 padding30[7];

  spinlock hlocks[KVMAP_HLOCK_NR];
  spinlock split_lock;
  struct __kvmap1_level {
    struct __kvmap_slot * table;
    struct bitmap * bm;
    u64 alloc_size;
    au64 nr_kv;
  } levels[64];
};

  static inline void
kvmap1_init(struct kvmap1 * const map)
{
  const u64 lid = KVMAP_MIN_LEVEL;
  map->max_level = lid;
  map->min_level = lid;
  map->first_check = lid;
  struct __kvmap1_level * const l = &(map->levels[lid]);
  l->table = pages_alloc_best(KVMAP_SIZE_OF_LEVEL(lid), true, &(l->alloc_size));
  l->bm = bitmap_create(KVMAP_SLOTS_OF(lid));
  debug_assert(l->table);
  debug_assert(l->bm);
  bitmap_set_all1(l->bm);
}

  static struct kvmap1 *
kvmap1_create(const struct kvmap_mm * const mm, const u64 hash_rotate)
{
  struct kvmap1 * const map = xalloc(64, sizeof(*map));
  memset(map, 0, sizeof(*map));

  map->mm = mm ? (*mm) : __kvmap_mm_default;
  map->hash_rotate = hash_rotate;

  for (u64 j = 0; j < KVMAP_HLOCK_NR; j++) {
    spinlock_init(&(map->hlocks[j]));
  }
  spinlock_init(&(map->split_lock));

  kvmap1_init(map);
  return map;
}

  static inline u64
kvmap_hlocks_index(const u64 hash)
{
  return (hash >> KVMAP_BASE_LEVEL) & KVMAP_HLOCK_MASK;
}

  static inline void
kvmap1_lock_hash(struct kvmap1 * const map, const struct kv * const key)
{
  const u64 hash = bits_rotl_u64(key->hash, map->hash_rotate);
  spinlock_lock(&(map->hlocks[kvmap_hlocks_index(hash)]));
}

  static inline bool
kvmap1_trylock_hash(struct kvmap1 * const map, const struct kv * const key)
{
  const u64 hash = bits_rotl_u64(key->hash, map->hash_rotate);
  return spinlock_trylock(&(map->hlocks[kvmap_hlocks_index(hash)]));
}

  static inline void
kvmap1_unlock_hash(struct kvmap1 * const map, const struct kv * const key)
{
  const u64 hash = bits_rotl_u64(key->hash, map->hash_rotate);
  spinlock_unlock(&(map->hlocks[kvmap_hlocks_index(hash)]));
}

  static inline void
kvmap1_locate_level(const struct kvmap1 * const map, const u64 hash, u64 * const out_level, u64 * const out_index)
{
  u64 i = map->first_check;
  const u64 check0 = i;
  while (i >= map->min_level) {
    const struct __kvmap1_level * const level = &(map->levels[i]);
    const u64 j = hash & ((UINT64_C(1) << i) - UINT64_C(1));
    if (bitmap_test(level->bm, j)) {
      *out_level = i;
      *out_index = j;
      return;
    } else if (i >= check0) {
      if (i == map->max_level) i = check0 - 1;
      else i++;
    } else {
      i--;
    }
  }
}

  static bool
kvmap1_locate_key(const struct kvmap1 * const map, const struct kv * const key,
    u64 * const out_level, u64 * const out_index, u64 * const out_entry)
{
  const u64 hash = bits_rotl_u64(key->hash, map->hash_rotate);
  const u64 pkey = kvmap_pkey(hash);
  kvmap1_locate_level(map, hash, out_level, out_index);
  const struct __kvmap_slot * const slot = &(map->levels[*out_level].table[*out_index]);
  u64 used = 0;
  for (u64 k = 0; k < 8; k++) {
    if (slot->e[k].ptr) {
      used++;
      if (slot->e[k].pkey == pkey) {
        const struct kv * const curr = __u64_to_ptr(slot->e[k].ptr);
        if (kv_keymatch(key, curr)) {
          *out_entry = k;
          return true;
        }
      }
    }
  }
  *out_entry = used;
  return false;
}

// true: the slot has free entry; false: the slot is full.
  static bool
kvmap1_locate_free_entry(const struct kvmap1 * const map, const struct kv * const key,
    u64 * const out_level, u64 * const out_index, u64 * const out_entry)
{
  const u64 hash = bits_rotl_u64(key->hash, map->hash_rotate);
  kvmap1_locate_level(map, hash, out_level, out_index);
  const struct __kvmap_slot * const slot = &(map->levels[*out_level].table[*out_index]);
  for (u64 k = 0; k < 8; k++) {
    if (slot->e[k].ptr == 0) {
      *out_entry = k;
      return true; // found an empty entry
    }
  }
  return false;
}

  static void
kvmap1_split(struct kvmap1 * const map, const u64 i, const u64 j)
{
  debug_assert(i < 64);
  debug_assert(j < (UINT64_C(1) << i));
  const u64 lid1 = i + 1;
  struct __kvmap1_level * const level0 = &(map->levels[i]);
  struct __kvmap1_level * const level1 = &(map->levels[lid1]);
  debug_assert(level0->table);
  debug_assert(level0->bm);

  // create next level
  if (level1->table == NULL) {
    struct bitmap * const bm1 = bitmap_create(KVMAP_SLOTS_OF(lid1));
    u64 alloc_size = 0;
    struct __kvmap_slot * const table1 = pages_alloc_best(KVMAP_SIZE_OF_LEVEL(lid1), true, &alloc_size);
    debug_assert(bm1);
    debug_assert(table1);
    bool allocs_used = false;

    spinlock_lock(&map->split_lock);
    if (level1->table == NULL) {
      allocs_used = true;
      level1->bm = bm1;
      level1->table = table1;
      level1->alloc_size = alloc_size;
      atomic_thread_fence(memory_order_seq_cst);
      if (map->max_level < lid1) {
        debug_assert((map->max_level + 1) == lid1);
        map->max_level++;
      }
      atomic_thread_fence(memory_order_seq_cst);
    }
    spinlock_unlock(&map->split_lock);

    if (allocs_used == false) {
      free(bm1);
      pages_unmap(table1, alloc_size);
    }
  }

  const u64 j2 = j | (UINT64_C(1) << i);
  bitmap_set1(level1->bm, j);
  bitmap_set1(level1->bm, j2);

  debug_assert(map->max_level >= lid1);

  // mark next level slots inuse
  for (u64 k = 0; k < 8; k++) {
    struct kv * const kv = __u64_to_ptr(level0->table[j].e[k].ptr);
    if (kv) {
      const u64 hash = bits_rotl_u64(kv->hash, map->hash_rotate);
      const u64 jj = hash & ((UINT64_C(1) << (lid1)) - 1);
      level1->table[jj].e[k] = level0->table[j].e[k];
      level0->table[j].e[k].ptr = 0;
      level0->table[j].e[k].pkey = 0;
      atomic_fetch_add(&(level1->nr_kv), 1);
      atomic_fetch_sub(&(level0->nr_kv), 1);
    }
  }

  struct __kvmap_slot * table0 = NULL;

  spinlock_lock(&map->split_lock);
  if (map->first_check == i && (level1->nr_kv >= (level0->nr_kv << 1))) {
    map->first_check ++;
  }
  bitmap_set0(level0->bm, j);
  if (bitmap_test_all0(level0->bm) && (map->min_level == i)) {
    table0 = level0->table;
    level0->table = NULL;
    // TODO: start bg thread to recycle it.
    // Don't free bm. Some getter may check it.
    //free(level0->bm);
    //level0->bm = NULL;
    map->min_level++;
  }
  spinlock_unlock(&map->split_lock);
  if(table0) pages_unmap(table0, level0->alloc_size);
}

  static inline void
kvmap1_put_entry(struct kvmap1 * const map, struct __kvmap_entry * const e, const struct kv * const kv)
{
  struct kv * const old = __u64_to_ptr(e->ptr);
  if (old && map->mm.rf) {
    map->mm.rf(old, map->mm.rp);
  }
  e->ptr = __ptr_to_u64(kv);
  e->pkey = kv ? kvmap_pkey(bits_rotl_u64(kv->hash, map->hash_rotate)) : 0;
}

  static bool
kvmap1_insert(struct kvmap1 * const map, const struct kv * const kv, const u64 max_level)
{
  u64 i = UINT64_MAX;
  u64 j = UINT64_MAX;
  u64 k = UINT64_MAX;
  if (kvmap1_locate_free_entry(map, kv, &i, &j, &k)) {
    // has free entry, just insert
    struct __kvmap_entry * const e = &(map->levels[i].table[j].e[k]);
    kvmap1_put_entry(map, e, kv);
    //(void)atomic_fetch_add(&(map->nr_kv), 1);
    (void)atomic_fetch_add(&(map->levels[i].nr_kv), 1);
    return true;
  } else if (i < max_level) {
    kvmap1_split(map, i, j);
    return kvmap1_insert(map, kv, max_level);
  } else {
    return false;
  }
}

  static void
kvmap1_clean(struct kvmap1 * const map)
{
  for (u64 i = map->max_level; i >= KVMAP_MIN_LEVEL; i--) {
    struct __kvmap1_level * const level = &(map->levels[i]);
    if (level->bm == NULL) continue;
    const u64 nr_slots = KVMAP_SLOTS_OF(i);
    for (u64 j = 0; j < nr_slots; j++) {
      if (bitmap_test(level->bm, j)) {
        struct __kvmap_slot * const slot = &(level->table[j]);
        for (u64 k = 0; k < 8; k++) {
          struct __kvmap_entry * const e = &(slot->e[k]);
          if (e->ptr) {
            kvmap1_put_entry(map, e, NULL);
            //(void)atomic_fetch_sub(&(map->nr_kv), 1);
            (void)atomic_fetch_sub(&(level->nr_kv), 1);
          }
        }
      }
    }
    debug_assert(atomic_load(&(level->nr_kv)) == 0);
    if (level->table) {
      pages_unmap(level->table, level->alloc_size);
      level->table = NULL;
    }
    free(level->bm);
    level->bm = NULL;
  }
  debug_assert(atomic_load(&(map->nr_kv)) == 0);
  kvmap1_init(map);
}

  static void
kvmap1_destroy(struct kvmap1 * const map)
{
  kvmap1_clean(map);
  pages_unmap(map->levels[KVMAP_MIN_LEVEL].table, map->levels[KVMAP_MIN_LEVEL].alloc_size);
  free(map->levels[KVMAP_MIN_LEVEL].bm);
  free(map);
}

  static void
kvmap1_fprint(struct kvmap1 * const map, FILE * const out)
{
  u64 sum_kv = 0;
  for (u64 i = 0; i < 64; i++) {
    struct __kvmap1_level * const level = &(map->levels[i]);
    if (level->table && level->bm) {
      sum_kv += atomic_load(&(level->nr_kv));
    }
  }
  fprintf(out, "KVMAP ROTL %"PRIu64" NR %"PRIu64" ||", map->hash_rotate, sum_kv);
  const double sum100 = ((double)sum_kv) * 0.01;

  u64 sumhit = 0;
  u64 sumtry = 0;
  for (u64 i = 0; i < 64; i++) {
    struct __kvmap1_level * const level = &(map->levels[i]);
    if (level->table && level->bm) {
      const u64 nr_kv = atomic_load(&(level->nr_kv));
      const u64 used = bitmap_count(level->bm);
      const double pslots = ((double)used) / ((double)(KVMAP_SLOTS_OF(i))) * 100.0;
      const char tag = (i == map->first_check) ? '*':' ';
      fprintf(out, " %cL %"PRIu64" SL %"PRIu64" %"PRIu64" %.2lf%% ET %"PRIu64" %"PRIu64" %.2lf%% |",
          tag, i, used, KVMAP_SLOTS_OF(i), pslots, nr_kv, KVMAP_ENTRIES_OF(i), ((double)nr_kv) / sum100);
      sumhit += nr_kv;
      const u64 ntry = i >= map->first_check ? (i - map->first_check + 1) : (map->max_level - i + 1);
      sumtry += nr_kv * ntry;
    }
  }
  fprintf(out, " KVMAP AVG TRY %.2lf\n", ((double)sumtry) / ((double)sumhit));
}

struct kvmap1_iter {
  struct kvmap1 * map;
  u64 i;
  u64 j;
  u64 k;
};

  static struct kvmap1_iter *
kvmap1_iter_create(struct kvmap1 * const map)
{
  struct kvmap1_iter * const iter = (typeof(iter))malloc(sizeof(*iter));
  if (iter == NULL) {
    return NULL;
  }
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_lock(&(map->hlocks[i]));
  }
  spinlock_lock(&(map->split_lock));
  iter->map = map;
  iter->i = map->max_level;
  iter->j = 0;
  iter->k = 0;
  return iter;
}

  static struct kv *
kvmap1_iter_next(struct kvmap1_iter * const iter, struct kv * const out)
{
  u64 i = iter->i;
  u64 j = iter->j;
  u64 k = iter->k;
  struct kv * ret = NULL;
  while (i >= KVMAP_MIN_LEVEL) {
    if (iter->map->levels[i].table == NULL) {
      i--;
      continue;
    }
    struct __kvmap1_level * const level = &(iter->map->levels[i]);
    if (k == 0) { // try skip empty slots at every new j
      while ((j < (1lu << i)) && (bitmap_test(level->bm, j) == false)) j++;
      if (j >= (1lu << i)) {
        j = 0;
        i--;
        continue;
      }
    }
    if (level->table[j].e[k].ptr) {
      struct kv * const kv = __u64_to_ptr(level->table[j].e[k].ptr);
      ret = kv_dup2(kv, out);
    }
    // shift next
    k++;
    if (k >= 8) {
      k = 0;
      j++;
      if (j >= (1lu << i)) {
        j = 0;
        i--;
      }
    }
    if (ret) break;
  }
  iter->i = i;
  iter->j = j;
  iter->k = k;
  return ret;
}

  static inline void
kvmap1_iter_destroy(struct kvmap1_iter * const iter)
{
  struct kvmap1 * const map = iter->map;
  spinlock_unlock(&(map->split_lock));
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_unlock(&(map->hlocks[i]));
  }
  free(iter);
}
// }}} kvmap1

// kvmap2 {{{
struct kvmap2 {
  struct kvmap_mm mm;

  u64 sz_shift;
  struct kvmap1 *maps[2];
};

  struct kvmap2 *
kvmap2_create(const struct kvmap_mm * const mm)
{
  struct kvmap2 * const map2 = calloc(1, sizeof(*map2));
  map2->mm = mm ? (*mm) : __kvmap_mm_default;
  map2->sz_shift = 5;
  map2->maps[0] = kvmap1_create(mm, 0);
  map2->maps[1] = kvmap1_create(mm, 19);
  return map2;
}

  struct kv *
kvmap2_get(struct kvmap2 * const map2, const struct kv * const key, struct kv * const out)
{
  u64 i = UINT64_MAX;
  u64 j = UINT64_MAX;
  u64 k = UINT64_MAX;
  kvmap1_lock_hash(map2->maps[0], key);
  for (u64 x = 0; x < 2; x++) {
    struct kvmap1 * const map = map2->maps[x];
    if (kvmap1_locate_key(map, key, &i, &j, &k)) {
      struct kv * const curr = __u64_to_ptr(map->levels[i].table[j].e[k].ptr);
      if (map2->mm.hf) map2->mm.hf(curr, map2->mm.hp);
      struct kv * const ret = kv_dup2(curr, out);
      kvmap1_unlock_hash(map2->maps[x], key);
      return ret;
    }
    if (x == 0) {
      kvmap1_lock_hash(map2->maps[1], key);
    }
    kvmap1_unlock_hash(map2->maps[x], key);
  }
  return NULL;
}

  bool
kvmap2_probe(struct kvmap2 * const map2, const struct kv * const key)
{
  u64 i = UINT64_MAX;
  u64 j = UINT64_MAX;
  u64 k = UINT64_MAX;
  kvmap1_lock_hash(map2->maps[0], key);
  for (u64 x = 0; x < 2; x++) {
    struct kvmap1 * const map = map2->maps[x];
    if (kvmap1_locate_key(map, key, &i, &j, &k)) {
      struct kv * const curr = __u64_to_ptr(map->levels[i].table[j].e[k].ptr);
      if (map2->mm.hf) map2->mm.hf(curr, map2->mm.hp);
      kvmap1_unlock_hash(map2->maps[x], key);
      return true;
    }
    if (x == 0) {
      kvmap1_lock_hash(map2->maps[1], key);
    }
    kvmap1_unlock_hash(map2->maps[x], key);
  }
  return false;
}

  static bool
kvmap2_try_cuckoo(struct kvmap1 * const target, struct kvmap1 * const origin, const u64 oi, struct __kvmap_entry * const e)
{
  struct kv * const victim = __u64_to_ptr(e->ptr);
  if (kvmap1_trylock_hash(target, victim)) {
    const bool ret = kvmap1_insert(target, victim, target->max_level);
    if (ret) {
      e->ptr = 0;
      e->pkey = 0;
      //(void)atomic_fetch_sub(&(origin->nr_kv), 1);
      (void)atomic_fetch_sub(&(origin->levels[oi].nr_kv), 1);
    }
    kvmap1_unlock_hash(target, victim);
    return ret;
  }
  return false;
}

  static bool
kvmap2_cuckoo(struct kvmap2 * const map2, const u64 x, const struct kv * const kv)
{
  u64 vi = UINT64_MAX;
  u64 vj = UINT64_MAX;
  u64 vk = UINT64_MAX;
  (void)kvmap1_locate_key(map2->maps[x], kv, &vi, &vj, &vk);
  (void)vk;
  struct __kvmap_slot * const slot = &(map2->maps[x]->levels[vi].table[vj]);
  for (u64 k = 0; k < 8; k++) {
    struct __kvmap_entry * const e = &(slot->e[k]);
    if (e->ptr == 0) return true;
    const bool ret = kvmap2_try_cuckoo(map2->maps[1 - x], map2->maps[x], vi, e);
    if (ret) return true;
  }
  return false;
}

  bool
kvmap2_set(struct kvmap2 * const map2, const struct kv * const kv0)
{
  u64 i = UINT64_MAX;
  u64 j = UINT64_MAX;
  u64 k = UINT64_MAX;
  u64 hmax[2];
  u64 hi[2];
  //u64 hj[2];
  u64 hk[2];

  const struct kv * const kv = kv_alloc_dup(kv0, &(map2->mm));
  if (kv == NULL) return false;

  for (u64 x = 0; x < 2; x++) {
    kvmap1_lock_hash(map2->maps[x], kv);
  }
  for (u64 x = 0; x < 2; x++) {
    struct kvmap1 * const map = map2->maps[x];
    if (kvmap1_locate_key(map, kv, &i, &j, &k)) {
      kvmap1_unlock_hash(map2->maps[1 - x], kv);
      struct __kvmap_entry * const e = &(map->levels[i].table[j].e[k]);
      kvmap1_put_entry(map, e, kv);
      kvmap1_unlock_hash(map2->maps[x], kv);
      return true;
    }
    hi[x] = i;
    hk[x] = k;
    hmax[x] = map->max_level;
    const u64 bar0 = hmax[x] + (x * map2->sz_shift);
    if (bar0 < hmax[0]) {
      hmax[x] += (hmax[0] - bar0);
    }
  }

  // insert with no split
  for (u64 x = 0; x < 2; x++) {
    if (hk[x] < 8) {
      kvmap1_unlock_hash(map2->maps[1 - x], kv);
      const bool ret = kvmap1_insert(map2->maps[x], kv, hi[x]);
      (void)ret;
      kvmap1_unlock_hash(map2->maps[x], kv);
      debug_assert(ret);
      return true;
    }
  }

  // insert with limited number of splits
  for (u64 x = 0; x < 2; x++) {
    if (hmax[x] > hi[x]) {
      const bool ret = kvmap1_insert(map2->maps[x], kv, hmax[x]);
      if (ret) {
        for (u64 y = 0; y < 2; y++) {
          kvmap1_unlock_hash(map2->maps[y], kv);
        }
        return true;
      }
    }
  }

  // cuckoo
  for (u64 x = 0; x < 2; x++) {
    if (kvmap2_cuckoo(map2, x, kv)) {
      kvmap1_unlock_hash(map2->maps[1 - x], kv);
      const bool ret = kvmap1_insert(map2->maps[x], kv, hmax[x]);
      (void)ret;
      kvmap1_unlock_hash(map2->maps[x], kv);
      debug_assert(ret);
      return true;
    }
  }

  // bad luck: expand
  kvmap1_unlock_hash(map2->maps[1], kv);
  const bool ret = kvmap1_insert(map2->maps[0], kv, 64);
  (void)ret;
  debug_assert(ret);
  kvmap1_unlock_hash(map2->maps[0], kv);
  return true;
}

  bool
kvmap2_del(struct kvmap2 * const map2, const struct kv * const key)
{
  u64 i = UINT64_MAX;
  u64 j = UINT64_MAX;
  u64 k = UINT64_MAX;
  kvmap1_lock_hash(map2->maps[0], key);
  for (u64 x = 0; x < 2; x++) {
    struct kvmap1 * const map = map2->maps[x];
    if (kvmap1_locate_key(map, key, &i, &j, &k)) {
      struct __kvmap_entry * const e = &(map->levels[i].table[j].e[k]);
      //(void)atomic_fetch_sub(&(map->nr_kv), 1);
      (void)atomic_fetch_sub(&(map->levels[i].nr_kv), 1);
      kvmap1_put_entry(map, e, NULL);
      kvmap1_unlock_hash(map2->maps[x], key);
      return true;
    }
    if (x == 0) {
      kvmap1_lock_hash(map2->maps[1], key);
    }
    kvmap1_unlock_hash(map2->maps[x], key);
  }
  return false;

}

  inline void
kvmap2_clean(struct kvmap2 * const map2)
{
  for (u64 x = 0; x < 2; x++) {
    kvmap1_clean(map2->maps[x]);
  }
}

  inline void
kvmap2_destroy(struct kvmap2 * const map2)
{
  for (u64 x = 0; x < 2; x++) {
    kvmap1_destroy(map2->maps[x]);
  }
  free(map2);
}

  void
kvmap2_fprint(struct kvmap2 * const map2, FILE * const out)
{
  //u64 sum_kv = 0;
  //for (u64 x = 0; x < 2; x++) {
  //  sum_kv += atomic_load(&(map2->maps[x]->nr_kv));
  //}
  //const double sum100 = ((double)sum_kv) * 0.01;

  //fprintf(out, "KVMAP2 sz_shift %"PRIu64" nr_kv %"PRIu64"\n", map2->sz_shift, sum_kv);
  for (u64 x = 0; x < 2; x++) {
    //const u64 nr_kv = atomic_load(&(map2->maps[x]->nr_kv));
    //fprintf(out, "  KVMAP2 %"PRIu64" %.2lf%% ||| ", x, ((double)nr_kv) / sum100);
    kvmap1_fprint(map2->maps[x], out);
  }
}

struct kvmap2_iter {
  struct kvmap2 * map2;
  u64 map_id;
  struct kvmap1_iter * iter;
  struct kvmap1_iter * iters[2];
};

  inline struct kvmap2_iter *
kvmap2_iter_create(struct kvmap2 * const map2)
{
  struct kvmap2_iter * const iter2 = (typeof(iter2))malloc(sizeof(*iter2));
  iter2->map2 = map2;
  iter2->map_id = 0;
  for (u64 x = 0; x < 2; x++) {
    iter2->iters[x] = kvmap1_iter_create(map2->maps[x]);
  }
  iter2->iter = iter2->iters[0];
  return iter2;
}

  struct kv *
kvmap2_iter_next(struct kvmap2_iter * const iter2, struct kv * const out)
{
  struct kv * const ret = kvmap1_iter_next(iter2->iter, out);
  if (ret == NULL) {
    iter2->iter = NULL;
    iter2->map_id++;
    if (iter2->map_id < 2) {
      iter2->iter = iter2->iters[iter2->map_id];
      return kvmap1_iter_next(iter2->iter, out);
    }
  }
  return ret;
}

  inline void
kvmap2_iter_destroy(struct kvmap2_iter * const iter2)
{
  for (u64 x = 0; x < 2; x++) {
    kvmap1_iter_destroy(iter2->iters[x]);
  }
  free(iter2);
}

  struct kvmap_api *
kvmap2_api_create(struct kvmap2 * const map)
{
  if (map == NULL) return NULL;
  const struct kvmap_api api = {
    .map = map,
    .get = (typeof(api.get))kvmap2_get,
    .probe = (typeof(api.probe))kvmap2_probe,
    .set = (typeof(api.set))kvmap2_set,
    .del = (typeof(api.del))kvmap2_del,
    .clean = (typeof(api.clean))kvmap2_clean,
    .destroy = (typeof(api.destroy))kvmap2_destroy,
    .fprint = (typeof(api.fprint))kvmap2_fprint,
    .iter_create = (typeof(api.iter_create))kvmap2_iter_create,
    .iter_next = (typeof(api.iter_next))kvmap2_iter_next,
    .iter_destroy = (typeof(api.iter_destroy))kvmap2_iter_destroy,
  };
  struct kvmap_api * const papi = malloc(sizeof(*papi));
  *papi = api;
  return papi;
}
// }}} kvmap2

// cuckoo {{{
struct cuckoo {
  struct kvmap_mm mm;

  spinlock hlocks[KVMAP_HLOCK_NR];
  mutexlock expand_lock;
  pthread_t expand_thread;
  u64 padding10[7];

  u64 fixed_power;
  struct bitmap * bm; // NULL if expand prohibited

  u64 padding20[7];

  struct __cuckoo_level {
    struct __kvmap_slot * table;
    u64 alloc_size;
    u64 power;
  } levels[2];
};

// low power-bits
  static inline u64
__cuckoo_hidx(const u64 hash, const u64 power)
{
  return hash & ((UINT64_C(1) << power) - UINT64_C(1));
}

// mid bits
  static inline u64
cuckoo_hash(const struct kv * const key, const u64 hash_id)
{
  return hash_id ? bits_rotl_u64(key->hash, hash_id * 19lu) : key->hash;
}

  static void
cuckoo_init(struct cuckoo * const map)
{
  const u64 power = map->fixed_power > KVMAP_MIN_LEVEL ? map->fixed_power : KVMAP_MIN_LEVEL;
  map->levels[0].power = power;
  map->levels[0].table = pages_alloc_best(KVMAP_SIZE_OF_LEVEL(power), true, &(map->levels[0].alloc_size));
  if (map->fixed_power == 0) {
    bitmap_set_all0(map->bm);
  }
  debug_assert(map->levels[0].table);
}

  struct cuckoo *
cuckoo_create(const struct kvmap_mm * const mm)
{
  struct cuckoo * const map = xalloc(64, sizeof(*map));
  memset(map, 0, sizeof(*map));

  map->mm = mm ? (*mm) : __kvmap_mm_default;
  for (u64 j = 0; j < KVMAP_HLOCK_NR; j++) {
    spinlock_init(&(map->hlocks[j]));
  }
  mutexlock_init(&(map->expand_lock));

  map->fixed_power = 0;
  if (map->fixed_power == 0) {
    map->bm = bitmap_create(KVMAP_HLOCK_NR);
  }

  cuckoo_init(map);
  return map;
}

  static inline void
cuckoo_lock(struct cuckoo * const map, const u64 idx)
{
  spinlock_lock(&(map->hlocks[idx]));
}

  static void
cuckoo_lock_two(struct cuckoo * const map, const struct kv * const key)
{
  const u64 hash2[2] = {cuckoo_hash(key, 0), cuckoo_hash(key, 1), };
  const u64 i1 = kvmap_hlocks_index(hash2[0]);
  const u64 i2 = kvmap_hlocks_index(hash2[1]);
  if (i1 < i2) {
    cuckoo_lock(map, i1);
    cuckoo_lock(map, i2);
  } else if (i1 > i2) {
    cuckoo_lock(map, i2);
    cuckoo_lock(map, i1);
  } else { // equal
    cuckoo_lock(map, i1);
  }
}

  static inline bool
cuckoo_trylock(struct cuckoo * const map, const u64 idx)
{
  return spinlock_trylock(&(map->hlocks[idx]));
}

  static inline void
cuckoo_unlock(struct cuckoo * const map, const u64 idx)
{
  spinlock_unlock(&(map->hlocks[idx]));
}

  static void
cuckoo_unlock_two(struct cuckoo * const map, const struct kv * const key)
{
  const u64 hash2[2] = {cuckoo_hash(key, 0), cuckoo_hash(key, 1), };
  const u64 i1 = kvmap_hlocks_index(hash2[0]);
  const u64 i2 = kvmap_hlocks_index(hash2[1]);
  if (i1 < i2) {
    cuckoo_unlock(map, i1);
    cuckoo_unlock(map, i2);
  } else if (i1 > i2) {
    cuckoo_unlock(map, i2);
    cuckoo_unlock(map, i1);
  } else { // equal
    cuckoo_unlock(map, i1);
  }
}

  static inline struct __kvmap_slot *
cuckoo_hash_slot(struct cuckoo * const map, const u64 hash)
{
  const u64 lid = map->bm ? (bitmap_test(map->bm, kvmap_hlocks_index(hash)) ? 1 : 0) : 0;
  struct __cuckoo_level * const level = &(map->levels[lid]);
  const u64 hidx = __cuckoo_hidx(hash, level->power);
  debug_assert(level->table);
  struct __kvmap_slot * const slot = &(level->table[hidx]);
  return slot;
}

  struct kv *
cuckoo_get(struct cuckoo * const map, const struct kv * const key, struct kv * const out)
{
  const u64 hash2[2] = {cuckoo_hash(key, 0), cuckoo_hash(key, 1), };
  const u64 pkey = kvmap_pkey(key->hash);

  cuckoo_lock_two(map, key);
  for (u64 x = 0; x < 2; x++) {
    struct __kvmap_slot * const slot = cuckoo_hash_slot(map, hash2[x]);
    for (u64 k = 0; k < 8; k++) {
      if (slot->e[k].pkey == pkey) {
        const struct kv * const curr = __u64_to_ptr(slot->e[k].ptr);
        if (kv_keymatch(key, curr)) {
          if (map->mm.hf) map->mm.hf(curr, map->mm.hp);
          struct kv * const ret = kv_dup2(curr, out);
          cuckoo_unlock_two(map, key);
          return ret;
        }
      }
    }
  }
  cuckoo_unlock_two(map, key);
  return NULL;
}

  bool
cuckoo_probe(struct cuckoo * const map, const struct kv * const key)
{
  const u64 hash2[2] = {cuckoo_hash(key, 0), cuckoo_hash(key, 1), };
  const u64 pkey = kvmap_pkey(key->hash);

  cuckoo_lock_two(map, key);
  for (u64 x = 0; x < 2; x++) {
    struct __kvmap_slot * const slot = cuckoo_hash_slot(map, hash2[x]);
    for (u64 k = 0; k < 8; k++) {
      if (slot->e[k].pkey == pkey) {
        const struct kv * const curr = __u64_to_ptr(slot->e[k].ptr);
        if (kv_keymatch(key, curr)) {
          if (map->mm.hf) map->mm.hf(curr, map->mm.hp);
          cuckoo_unlock_two(map, key);
          return true;
        }
      }
    }
  }
  cuckoo_unlock_two(map, key);
  return false;
}

struct cuckoo_expand_worker_info {
  struct cuckoo * map;
  au64 uniq;
  struct __cuckoo_level * level0;
  struct __cuckoo_level * level1;
  void (* bm_updater)(struct bitmap *, u64);
};

  static void *
cuckoo_expand_worker(void * const ptr)
{
  struct cuckoo_expand_worker_info * const info = (typeof(info))ptr;
  struct cuckoo * const map = info->map;
  struct __cuckoo_level * const level0 = info->level0;
  struct __cuckoo_level * const level1 = info->level1;
  u64 uniq;
  while ((uniq = atomic_fetch_add(&(info->uniq), 64)) < KVMAP_HLOCK_NR) {
    for (u64 lidx = uniq; lidx < (uniq + 64); lidx++) {
      spinlock_lock(&(map->hlocks[lidx]));
      const u64 power0 = level0->power;
      const u64 hi_nr = UINT64_C(1) << (power0 - KVMAP_MIN_LEVEL);
      const u64 lo_nr = UINT64_C(1) << KVMAP_BASE_LEVEL;
      const u64 mid = lidx << KVMAP_BASE_LEVEL;
      for (u64 hi = 0; hi < hi_nr; hi++) {
        for (u64 lo = 0; lo < lo_nr; lo++) {
          const u64 hidx0 = (hi << KVMAP_MIN_LEVEL) | mid | lo;
          debug_assert(kvmap_hlocks_index(hidx0) == lidx);
          struct __kvmap_slot * const slot = &(level0->table[hidx0]);
          for (u64 k = 0; k < 8; k++) {
            if (slot->e[k].ptr) {
              struct kv * const curr = __u64_to_ptr(slot->e[k].ptr);
              const u64 hash2[2] = {cuckoo_hash(curr, 0), cuckoo_hash(curr, 1), };
              const u64 hash = __cuckoo_hidx(hash2[0], power0) == hidx0 ? hash2[0] : hash2[1];
              debug_assert(__cuckoo_hidx(hash, power0) == hidx0);
              level1->table[__cuckoo_hidx(hash, level1->power)].e[k] = slot->e[k];
              slot->e[k].ptr = 0;
              slot->e[k].pkey = 0;
            }
          }
        }
      }
      info->bm_updater(map->bm, lidx);
      spinlock_unlock(&(map->hlocks[lidx]));
    }
  }
  return NULL;
}

  static void *
cuckoo_expand_thread(void * const ptr)
{
  struct cuckoo * const map = (typeof(map))ptr;
  debug_assert(map->bm);
  struct __cuckoo_level * const level0 = map->levels[0].table ? &(map->levels[0]) : &(map->levels[1]);
  struct __cuckoo_level * const level1 = map->levels[0].table ? &(map->levels[1]) : &(map->levels[0]);
  debug_assert(level0->table);
  debug_assert(level1->table == NULL);
  const u64 power1 = level0->power + 1;
  level1->power = power1;
  level1->table = pages_alloc_best(KVMAP_SIZE_OF_LEVEL(power1), true, &(level1->alloc_size));

  struct cuckoo_expand_worker_info info = {
    .map = map,
    .uniq = 0,
    .level0 = level0,
    .level1 = level1,
    .bm_updater = bitmap_count(map->bm) ? bitmap_set0 : bitmap_set1,
  };
  const double dt = thread_fork_join(0, cuckoo_expand_worker, &info);
  (void)dt;
  //fprintf(stderr, "%s workers time %.2lf\n", __func__, dt);
  debug_assert(atomic_load(&info.uniq) > KVMAP_HLOCK_NR);
  pages_unmap(level0->table, level0->alloc_size);
  level0->table = NULL;
  mutexlock_unlock(&(map->expand_lock));
  return NULL;
}

  static void
cuckoo_expand(struct cuckoo * const map)
{
  if (mutexlock_trylock(&(map->expand_lock)) == false) return;
  pthread_create(&(map->expand_thread), NULL, cuckoo_expand_thread, map);
  pthread_setname_np(map->expand_thread, "cuckoo_expand");
}

// assume lidx of hash locked
  static bool
cuckoo_cuckoo_local(struct cuckoo * const map, const u64 hash, const struct kv * const kv)
{
  struct __kvmap_slot * const slot = cuckoo_hash_slot(map, hash);
  for (u64 k = 0; k < 8; k++) {
    if (slot->e[k].ptr == 0) {
      //cuckoo_put_entry(map, &(slot->e[k]), kv);
      kvmap_put_entry(&(map->mm), &(slot->e[k]), kv);
      return true;
    }
  }
  return false;
}

  static bool
cuckoo_cuckoo_depth(struct cuckoo * const map, const u64 hash, const struct kv * const kv, const u64 max_cuckoo)
{
  if (max_cuckoo == 0) {
    return cuckoo_cuckoo_local(map, hash, kv);
  }

  struct __kvmap_slot * const slot = cuckoo_hash_slot(map, hash);
  const u64 lidx0 = kvmap_hlocks_index(hash);
  for (u64 k = 0; k < 8; k++) {
    if (slot->e[k].ptr == 0) { // some slots may be released by other threads
      //cuckoo_put_entry(map, &(slot->e[k]), kv);
      kvmap_put_entry(&(map->mm), &(slot->e[k]), kv);
      return true;
    }
    const struct kv * const curr = __u64_to_ptr(slot->e[k].ptr);
    const u64 hash2[2] = {cuckoo_hash(curr, 0), cuckoo_hash(curr, 1), };
    for (u64 x = 0; x < 2; x++) {
      const u64 lidx1 = kvmap_hlocks_index(hash2[x]);
      if ((lidx1 != lidx0) && (cuckoo_trylock(map, lidx1))) {
        const bool r = cuckoo_cuckoo_depth(map, hash2[x], curr, max_cuckoo - 1lu);
        cuckoo_unlock(map, lidx1);
        if (r) {
          slot->e[k].ptr = 0;
          slot->e[k].pkey = 0;
          //cuckoo_put_entry(map, &(slot->e[k]), kv);
          kvmap_put_entry(&(map->mm), &(slot->e[k]), kv);
          return true;
        }
      }
    }
  }
  return false;
}

  static bool
cuckoo_try_set(struct cuckoo * const map, const struct kv * const kv)
{
  const u64 hash2[2] = {cuckoo_hash(kv, 0), cuckoo_hash(kv, 1), };
  const u64 pkey = kvmap_pkey(kv->hash);

  cuckoo_lock_two(map, kv);
  // try update
  for (u64 x = 0; x < 2; x++) {
    struct __kvmap_slot * const slot = cuckoo_hash_slot(map, hash2[x]);
    for (u64 k = 0; k < 8; k++) {
      if (slot->e[k].pkey == pkey) {
        const struct kv * const curr = __u64_to_ptr(slot->e[k].ptr);
        if (kv_keymatch(kv, curr)) {
          kvmap_put_entry(&(map->mm), &(slot->e[k]), kv);
          cuckoo_unlock_two(map, kv);
          return true;
        }
      }
    }
  }

  // cuckoo
  for (u64 depth = 0; depth < 3; depth++) {
    for (u64 x = 0; x < 2; x++) {
      if (cuckoo_cuckoo_depth(map, hash2[x], kv, depth)) {
        cuckoo_unlock_two(map, kv);
        return true;
      }
    }
  }
  cuckoo_unlock_two(map, kv);
  if (map->bm) cuckoo_expand(map);
  return false;
}

  bool
cuckoo_set(struct cuckoo * const map, const struct kv * const kv)
{
  const struct kv * const new = kv_alloc_dup(kv, &(map->mm));
  if (new == NULL) return false;
  if (map->fixed_power) {
    for (u64 i = 0; i < 8; i++) {
      if (cuckoo_try_set(map, new)) return true;
    }
    return false;
  } else {
    while (cuckoo_try_set(map, new) == false);
    return true;
  }
}

  bool
cuckoo_del(struct cuckoo * const map, const struct kv * const key)
{
  const u64 hash2[2] = {cuckoo_hash(key, 0), cuckoo_hash(key, 1), };
  const u64 pkey = kvmap_pkey(key->hash);

  cuckoo_lock_two(map, key);
  for (u64 x = 0; x < 2; x++) {
    struct __kvmap_slot * const slot = cuckoo_hash_slot(map, hash2[x]);
    for (u64 k = 0; k < 8; k++) {
      if (slot->e[k].pkey == pkey) {
        const struct kv * const curr = __u64_to_ptr(slot->e[k].ptr);
        if (kv_keymatch(key, curr)) {
          kvmap_put_entry(&(map->mm), &(slot->e[k]), NULL);
          cuckoo_unlock_two(map, key);
          return true;
        }
      }
    }
  }
  cuckoo_unlock_two(map, key);
  return false;
}

  void
cuckoo_clean(struct cuckoo * const map)
{
  mutexlock_lock(&(map->expand_lock));
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_lock(&(map->hlocks[i]));
  }
  for (u64 x = 0; x < 2; x++) {
    if (map->levels[x].table == NULL) continue;
    const u64 nr_slots = UINT64_C(1) << map->levels[x].power;
    for (u64 j = 0; j < nr_slots; j++) {
      struct __kvmap_slot * const slot = &(map->levels[x].table[j]);
      for (u64 k = 0; k < 8; k++) {
        if (slot->e[k].ptr) {
          kvmap_put_entry(&(map->mm), &(slot->e[k]), NULL);
        }
      }
    }
    pages_unmap(map->levels[x].table, map->levels[x].alloc_size);
    map->levels[x].table = NULL;
  }
  cuckoo_init(map);
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_unlock(&(map->hlocks[i]));
  }
  mutexlock_unlock(&(map->expand_lock));
}

  void
cuckoo_destroy(struct cuckoo * const map)
{
  cuckoo_clean(map);
  for (u64 x = 0; x < 2; x++) {
    if (map->levels[x].table) {
      pages_unmap(map->levels[x].table, map->levels[x].alloc_size);
    }
  }
  free(map->bm);
  free(map);
}

  void
cuckoo_fprint(struct cuckoo * const map, FILE * const out)
{
  mutexlock_lock(&(map->expand_lock));
  for (u64 x = 0; x < 2; x++) {
    if (map->levels[x].table == NULL) continue;
    const u64 power = map->levels[x].power;
    const u64 nr_slots = KVMAP_SLOTS_OF(power);
    const u64 nr_entries = nr_slots * 8;
    fprintf(out, "CUCKOO POWER %"PRIu64" ENTRY %"PRIu64"\n", power, nr_entries);
  }
  mutexlock_unlock(&(map->expand_lock));
}

struct cuckoo_iter {
  struct cuckoo * map;
  struct __cuckoo_level * level;
  u64 j;
  u64 k;
};

  struct cuckoo_iter *
cuckoo_iter_create(struct cuckoo * const map)
{
  struct cuckoo_iter * const iter = (typeof(iter))malloc(sizeof(*iter));
  if (iter == NULL) {
    return NULL;
  }
  mutexlock_lock(&(map->expand_lock));
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_lock(&(map->hlocks[i]));
  }
  iter->map = map;
  iter->level = map->levels[0].table ? &map->levels[0] : &map->levels[1];
  iter->j = 0;
  iter->k = 0;
  return iter;
}

  struct kv *
cuckoo_iter_next(struct cuckoo_iter * const iter, struct kv * const out)
{
  u64 j = iter->j;
  u64 k = iter->k;
  struct __cuckoo_level * const level= iter->level;

  const u64 nr = KVMAP_SLOTS_OF(level->power);
  if (j >= nr) return NULL;
  while ((j < nr) && (level->table[j].e[k].ptr == 0)) {
    k++;
    if (k == 8) {
      j++;
      k = 0;
    }
  }
  struct kv * ret = NULL;
  if (j < nr) {
    struct kv * kv = __u64_to_ptr(level->table[j].e[k].ptr);
    ret = kv_dup2(kv, out);
    k++;
    if (k == 8) {
      k = 0;
      j++;
    }
  }
  iter->j = j;
  iter->k = k;
  return ret;
}

  void
cuckoo_iter_destroy(struct cuckoo_iter * const iter)
{
  struct cuckoo * const map = iter->map;
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_unlock(&(map->hlocks[i]));
  }
  mutexlock_unlock(&(map->expand_lock));
  free(iter);
}

  struct kvmap_api *
cuckoo_api_create(struct cuckoo * const map)
{
  if (map == NULL) return NULL;
  const struct kvmap_api api = {
    .map = map,
    .get = (typeof(api.get))cuckoo_get,
    .probe = (typeof(api.probe))cuckoo_probe,
    .set = (typeof(api.set))cuckoo_set,
    .del = (typeof(api.del))cuckoo_del,
    .clean = (typeof(api.clean))cuckoo_clean,
    .destroy = (typeof(api.destroy))cuckoo_destroy,
    .fprint = (typeof(api.fprint))cuckoo_fprint,
    .iter_create = (typeof(api.iter_create))cuckoo_iter_create,
    .iter_next = (typeof(api.iter_next))cuckoo_iter_next,
    .iter_destroy = (typeof(api.iter_destroy))cuckoo_iter_destroy,
  };
  struct kvmap_api * const papi = (typeof(papi))malloc(sizeof(*papi));
  *papi = api;
  return papi;
}
// }}} cuckoo

// skiplist {{{
struct skipnode {
  struct __kvmap_entry e;
  struct skipnode * next[0];
};

struct skiplist {
  struct kvmap_mm mm;
  kv_compare_func comp;
  double factors[64];
  u64 height;
  struct skipnode n0;
  u64 n0extra[64];
};

  struct skiplist *
skiplist_create_f(const struct kvmap_mm * const mm, kv_compare_func comp)
{
  struct skiplist * const head = (typeof(head))malloc(sizeof(*head));
  memset(head, 0, sizeof(*head));
  head->mm = mm ? (*mm) : __kvmap_mm_default;
  head->comp = comp;
  head->height = 1;
  head->factors[0] = 1.0;
  const double factor = 0.25;
  const double mul = fabs(factor) > 0.5 ? 0.5 : fabs(factor);
  for (u64 i = 1; i < 64; i++) {
    head->factors[i] = head->factors[i-1] * mul;
  }
  return head;
}

  struct skiplist *
skiplist_create(const struct kvmap_mm * const mm)
{
  return skiplist_create_f(mm, kv_keycompare);
}

  struct kv *
skiplist_get(struct skiplist * const list, const struct kv * const key, struct kv * const out)
{
  u64 i = list->height - 1;
  struct skipnode * left = &(list->n0);
  struct skipnode * right = NULL;
  while (i) {
    while (left->next[i]) {
      right = left->next[i];
      struct kv * const curr = __u64_to_ptr(right->e.ptr);
      debug_assert(curr);
      const int cmp = list->comp(key, curr);
      if (cmp == 0) {
        if (list->mm.hf) list->mm.hf(curr, list->mm.hp);
        struct kv * const ret = kv_dup2(curr, out);
        return ret;
      } else if (cmp < 0) {
        break;
      } else {
        left = right;
      }
    }
    i--;
  }
  const u64 pkey = kvmap_pkey(key->hash);
  if (left) left = left->next[0];
  while (left && (left != right)) {
    struct kv * const curr = __u64_to_ptr(left->e.ptr);
    if (kvmap_pkey(curr->hash) == pkey) {
      if (kv_keymatch(curr, key)) {
        if (list->mm.hf) list->mm.hf(curr, list->mm.hp);
        struct kv * const ret = kv_dup2(curr, out);
        return ret;
      }
    }
    left = left->next[0];
  }
  return NULL;
}

  bool
skiplist_probe(struct skiplist * const list, const struct kv * const key)
{
  u64 h = list->height - 1;
  struct skipnode * left = &(list->n0);
  struct skipnode * right = NULL;
  while (h) {
    while (left->next[h]) {
      right = left->next[h];
      struct kv * const curr = __u64_to_ptr(right->e.ptr);
      debug_assert(curr);
      const int cmp = list->comp(key, curr);
      if (cmp == 0) {
        if (list->mm.hf) list->mm.hf(curr, list->mm.hp);
        return true;
      } else if (cmp < 0) {
        break;
      } else {
        left = right;
      }
    }
    h--;
  }
  const u64 pkey = kvmap_pkey(key->hash);
  if (left) left = left->next[0];
  while (left && (left != right)) {
    struct kv * const curr = __u64_to_ptr(left->e.ptr);
    if (kvmap_pkey(curr->hash) == pkey) {
      if (kv_keymatch(curr, key)) {
        if (list->mm.hf) list->mm.hf(curr, list->mm.hp);
        return true;
      }
    }
    left = left->next[0];
  }
  return false;
}

  bool
skiplist_set(struct skiplist * const list, const struct kv * const kv)
{
  const struct kv * const newkv = kv_alloc_dup(kv, &(list->mm));
  u64 h = list->height - 1;
  struct skipnode * path[64];
  struct skipnode * left = &(list->n0);
  struct skipnode * right = NULL;
  while (h < 64) { // h >= 0
    while (left->next[h]) {
      right = left->next[h];
      struct kv * const curr = __u64_to_ptr(right->e.ptr);
      debug_assert(curr);
      const int cmp = list->comp(newkv, curr);
      if (cmp == 0) {
        kvmap_put_entry(&(list->mm), &(right->e), newkv);
        return true;
      } else if (cmp < 0) {
        break;
      } else {
        left = right;
      }
    }
    path[h] = left;
    h--;
  }
  // insert
  const double r = random_double();
  h = 0;
  while (r < list->factors[h]) h++;
  const u64 height = h;
  if (height > list->height) {
    for (u64 i = list->height; i < height; i++) {
      path[i] = &(list->n0);
    }
    list->height = height;
  }
  const u64 nodesize = sizeof(list->n0) + (sizeof(list->n0.next[0]) * height);
  struct skipnode * const newnode = (typeof(newnode))malloc(nodesize);
  if (newnode == NULL) {
    if (list->mm.rf) {
      list->mm.rf((struct kv *)newkv, list->mm.rp);
    }
    return false;
  }
  newnode->e.ptr = 0;
  kvmap_put_entry(&(list->mm), &(newnode->e), newkv);
  for (u64 i = 0; i < height; i++) {
    newnode->next[i] = path[i]->next[i];
    path[i]->next[i] = newnode;
  }
  return true;
}

  bool
skiplist_del(struct skiplist * const list, const struct kv * const key)
{
  u64 h = list->height - 1;
  struct skipnode * left = &(list->n0);
  struct skipnode * right = NULL;
  struct skipnode * del = NULL;
  while ((h < 64) && (del == NULL)) {
    while (left->next[h]) {
      right = left->next[h];
      struct kv * const curr = __u64_to_ptr(right->e.ptr);
      debug_assert(curr);
      const int cmp = list->comp(key, curr);
      if (cmp == 0) {
        del = right;
        break;
      } else if (cmp < 0) {
        break;
      } else {
        left = right;
      }
    }
    h--;
  }

  if (del == NULL) {
    return false;
  }
  h++;
  const u64 depth = h+1;
  struct skipnode *path[depth];
  while (h < 64) {
    while (left->next[h] != del) {
      left = left->next[h];
    }
    path[h] = left;
    h--;
  }
  for (u64 i = 0; i < depth; i++) {
    path[i]->next[i] = del->next[i];
  }
  kvmap_put_entry(&(list->mm), &(del->e), NULL);
  free(del);
  return true;
}

  void
skiplist_clean(struct skiplist * const list)
{
  struct skipnode * iter = list->n0.next[0];
  while (iter) {
    struct skipnode * const next = iter->next[0];
    kvmap_put_entry(&(list->mm), &(iter->e), NULL);
    free(iter);
    iter = next;
  }
  for (u64 i = 0; i < 64; i++) {
    list->n0.next[i] = NULL;
  }
}

  void
skiplist_destroy(struct skiplist * const list)
{
  skiplist_clean(list);
  free(list);
}

  void
skiplist_fprint(struct skiplist * const list, FILE * const out)
{
  struct skipnode * iter = list->n0.next[0];
  u64 count = 0;
  while (iter) {
    count++;
    iter = iter->next[0];
  }
  fprintf(out, "%s count %"PRIu64" height %"PRIu64"\n", __func__, count, list->height);
}

  struct kv *
skiplist_head(struct skiplist * const list, struct kv * const out)
{
  if (list->n0.next[0] == NULL) return NULL;
  struct kv * const curr = __u64_to_ptr(list->n0.next[0]->e.ptr);
  struct kv * const ret = kv_dup2(curr, out);
  return ret;
}

  struct kv *
skiplist_tail(struct skiplist * const list, struct kv * const out)
{
  if (list->n0.next[0] == NULL) return NULL;
  u64 i = list->height - 1;
  struct skipnode * iter = &(list->n0);
  while (i) {
    while (iter->next[i]) iter = iter->next[i];
    i--;
  }
  while (iter->next[0]) iter = iter->next[0];
  struct kv * const curr = __u64_to_ptr(iter->e.ptr);
  struct kv * const ret = kv_dup2(curr, out);
  return ret;
}

struct skiplist_iter {
  struct skipnode * iter;
  struct skiplist * list;
};

  struct skiplist_iter *
skiplist_iter_create(struct skiplist * const list)
{
  struct skiplist_iter * const iter = malloc(sizeof(*iter));
  iter->iter = list->n0.next[0];
  iter->list = list;
  return iter;
}

  struct kv *
skiplist_iter_next(struct skiplist_iter * const iter, struct kv * const out)
{
  if (iter->iter == NULL) return NULL;
  struct kv * const curr = __u64_to_ptr(iter->iter->e.ptr);
  struct kv * const ret = kv_dup2(curr, out);
  iter->iter = iter->iter->next[0];
  return ret;
}

  void
skiplist_iter_destroy(struct skiplist_iter * const iter)
{
  free(iter);
}

  struct kvmap_api *
skiplist_api_create(struct skiplist * const map)
{
  if (map == NULL) return NULL;
  const struct kvmap_api api = {
    .map = map,
    .get = (typeof(api.get))skiplist_get,
    .probe = (typeof(api.probe))skiplist_probe,
    .set = (typeof(api.set))skiplist_set,
    .del = (typeof(api.del))skiplist_del,
    .clean = (typeof(api.clean))skiplist_clean,
    .destroy = (typeof(api.destroy))skiplist_destroy,
    .fprint = (typeof(api.fprint))skiplist_fprint,
    .iter_create = (typeof(api.iter_create))skiplist_iter_create,
    .iter_next = (typeof(api.iter_next))skiplist_iter_next,
    .iter_destroy = (typeof(api.iter_destroy))skiplist_iter_destroy,
  };
  struct kvmap_api * const papi = (typeof(papi))malloc(sizeof(*papi));
  *papi = api;
  return papi;
}
// }}} skiplist

// chainmap {{{
struct chainnode {
  struct __kvmap_entry e;
  struct chainnode * next;
};

struct chainmap {
  struct kvmap_mm mm;
  u64 padding10[7];
  au64 nr_kv;
  u64 padding20[7];
  u64 fixed_power;
  u64 power;
  u64 trigger;
  u64 expand_thread;

  struct chainnode **table;
  u64 alloc_size;
  mutexlock expand_lock;
  spinlock hlocks[KVMAP_HLOCK_NR];
};

  struct chainmap *
chainmap_create(const struct kvmap_mm * const mm)
{
  struct chainmap * const map = (typeof(map))malloc(sizeof(*map));
  debug_assert(map);
  memset(map, 0, sizeof(*map));
  map->mm = mm ? (*mm) : __kvmap_mm_default;
  atomic_store(&(map->nr_kv), 0);
  map->fixed_power = 0;
  //if (map->fixed_power && (map->fixed_power < KVMAP_MIN_LEVEL)) {
  //  map->fixed_power = KVMAP_MIN_LEVEL;
  //}
  map->power = map->fixed_power ? map->fixed_power : KVMAP_MIN_LEVEL;
  map->trigger = UINT64_C(2) << map->power;
  const u64 table_size = sizeof(map->table[0]) * (UINT64_C(1) << map->power);
  map->table = (typeof(map->table))pages_alloc_best(table_size, true, &(map->alloc_size));
  mutexlock_init(&(map->expand_lock));
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_init(&(map->hlocks[i]));
  }
  return map;
}

  struct kv *
chainmap_get(struct chainmap * const map, const struct kv * const key, struct kv * const out)
{
  const u64 hash = key->hash;
  const u64 lidx = kvmap_hlocks_index(hash);
  const u64 pkey = kvmap_pkey(hash);
  spinlock_lock(&(map->hlocks[lidx]));
  const u64 cid = (hash >> KVMAP_BASE_LEVEL) & ((UINT64_C(1) << map->power) - UINT64_C(1));
  struct chainnode * iter = map->table[cid];
  while (iter) {
    if (iter->e.pkey == pkey) {
      const struct kv * const curr = __u64_to_ptr(iter->e.ptr);
      if (kv_keymatch(key, curr)) {
        if (map->mm.hf) map->mm.hf(curr, map->mm.hp);
        struct kv * const ret = kv_dup2(curr, out);
        spinlock_unlock(&(map->hlocks[lidx]));
        return ret;
      }
    }
    iter = iter->next;
  }
  spinlock_unlock(&(map->hlocks[lidx]));
  return NULL;
}

  bool
chainmap_probe(struct chainmap * const map, const struct kv * const key)
{
  const u64 hash = key->hash;
  const u64 lidx = kvmap_hlocks_index(hash);
  const u64 pkey = kvmap_pkey(hash);
  spinlock_lock(&(map->hlocks[lidx]));
  const u64 cid = (hash >> KVMAP_BASE_LEVEL) & ((UINT64_C(1) << map->power) - UINT64_C(1));
  struct chainnode * iter = map->table[cid];
  while (iter) {
    if (iter->e.pkey == pkey) {
      const struct kv * const curr = __u64_to_ptr(iter->e.ptr);
      if (kv_keymatch(key, curr)) {
        if (map->mm.hf) map->mm.hf(curr, map->mm.hp);
        spinlock_unlock(&(map->hlocks[lidx]));
        return true;
      }
    }
    iter = iter->next;
  }
  spinlock_unlock(&(map->hlocks[lidx]));
  return false;
}

struct chainmap_expand_worker_info {
  struct chainmap * map;
  u64 new_power;
  struct chainnode ** new_table;
  u64 new_alloc_size;
  au64 uniq;
};

  static void *
chainmap_expand_worker(void * const ptr)
{
  struct chainmap_expand_worker_info * const info = (typeof(info))ptr;
  struct chainmap * const map = info->map;
  debug_assert(map->power >= KVMAP_HLOCK_POWER);
  const u64 hi_max = UINT64_C(1) << (map->power - KVMAP_HLOCK_POWER);
  do {
    const u64 lidx = atomic_fetch_add(&(info->uniq), 1);
    if (lidx >= KVMAP_HLOCK_NR) break;
    spinlock_lock(&(map->hlocks[lidx]));
    for (u64 h = 0; h < hi_max; h++) {
      const u64 hidx = (h << KVMAP_HLOCK_POWER) | lidx;
      struct chainnode * iter = info->map->table[hidx];
      while (iter) {
        struct chainnode * const next = iter->next;
        struct kv * const curr = __u64_to_ptr(iter->e.ptr);
        const u64 h2 = (curr->hash >> KVMAP_BASE_LEVEL) & ((UINT64_C(1) << info->new_power) - UINT64_C(1));
        iter->next = info->new_table[h2];
        info->new_table[h2] = iter;
        iter = next;
      }
    }
  } while (true);
  return NULL;
}

  static void *
chainmap_expand_thread(void * const ptr)
{
  struct chainmap * const map = (typeof(map))ptr;
  struct chainmap_expand_worker_info info;
  info.map = map;
  info.new_power = map->power + 1;
  const u64 table_size = sizeof(map->table[0]) * (UINT64_C(1) << info.new_power);
  info.new_table = (typeof(info.new_table))pages_alloc_best(table_size, true, &(info.new_alloc_size));
  debug_assert(info.new_table);
  atomic_store(&(info.uniq), 0);
  const double dt = thread_fork_join(0, chainmap_expand_worker, &info);
  pages_unmap(map->table, map->alloc_size);
  map->table = info.new_table;
  map->alloc_size = info.new_alloc_size;
  map->power = info.new_power;
  map->trigger = UINT64_C(2) << map->power;
  atomic_thread_fence(memory_order_seq_cst);

  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_unlock(&(map->hlocks[i]));
  }
  mutexlock_unlock(&(map->expand_lock));
  fprintf(stderr, "%s workers time %.2lf\n", __func__, dt);
  return NULL;
}

  static void
chainmap_expand(struct chainmap * const map)
{
  if (mutexlock_trylock(&(map->expand_lock)) == false) return;
  pthread_create(&(map->expand_thread), NULL, chainmap_expand_thread, map);
  pthread_setname_np(map->expand_thread, "chainmap_expand");
}

  bool
chainmap_set(struct chainmap * const map, const struct kv * const kv0)
{
  const struct kv * const kv = kv_alloc_dup(kv0, &(map->mm));
  if (kv == NULL) return false;

  const u64 hash = kv->hash;
  const u64 lidx = kvmap_hlocks_index(hash);
  const u64 pkey = kvmap_pkey(hash);
  spinlock_lock(&(map->hlocks[lidx]));
  const u64 cid = (hash >> KVMAP_BASE_LEVEL) & ((UINT64_C(1) << map->power) - UINT64_C(1));
  struct chainnode * iter = map->table[cid];
  while (iter) {
    if (iter->e.pkey == pkey) {
      const struct kv * const curr = __u64_to_ptr(iter->e.ptr);
      if (kv_keymatch(kv, curr)) { // update
        kvmap_put_entry(&(map->mm), &(iter->e), kv);
        spinlock_unlock(&(map->hlocks[lidx]));
        return true;
      }
    }
    iter = iter->next;
  }
  // insert
  struct chainnode * const node = (typeof(node))malloc(sizeof(*node));
  if (node == NULL) {
    if (map->mm.rf) map->mm.rf((struct kv *)kv, map->mm.rp);
    spinlock_unlock(&(map->hlocks[lidx]));
    return false;
  }
  memset(node, 0, sizeof(*node));
  //chainmap_put_entry(map, &(node->e), kv);
  kvmap_put_entry(&(map->mm), &(node->e), kv);
  const u64 nr_kv = atomic_fetch_add(&(map->nr_kv), 1);
  node->next = map->table[cid];
  map->table[cid] = node;
  spinlock_unlock(&(map->hlocks[lidx]));
  if (nr_kv > map->trigger) {
    chainmap_expand(map);
  }
  return true;
}

  bool
chainmap_del(struct chainmap * const map, const struct kv * const key)
{
  const u64 hash = key->hash;
  const u64 lidx = kvmap_hlocks_index(hash);
  const u64 pkey = kvmap_pkey(hash);
  spinlock_lock(&(map->hlocks[lidx]));
  const u64 cid = (hash >> KVMAP_BASE_LEVEL) & ((UINT64_C(1) << map->power) - UINT64_C(1));
  struct chainnode ** piter = &(map->table[cid]);
  while (*piter) {
    struct chainnode * const iter = *piter;
    if (iter->e.pkey == pkey) {
      const struct kv * const curr = __u64_to_ptr(iter->e.ptr);
      if (kv_keymatch(key, curr)) {
        kvmap_put_entry(&(map->mm), &(iter->e), NULL);
        *piter = iter->next;
        free(iter);
        atomic_fetch_sub(&(map->nr_kv), 1);
        spinlock_unlock(&(map->hlocks[lidx]));
        return true;
      }
    }
    piter = &(iter->next);
  }
  spinlock_unlock(&(map->hlocks[lidx]));
  return false;
}

  inline void
chainmap_clean(struct chainmap * const map)
{
  mutexlock_lock(&(map->expand_lock));
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_lock(&(map->hlocks[i]));
  }
  const u64 nr = UINT64_C(1) << map->power;
  for (u64 i = 0; i < nr; i++) {
    struct chainnode * iter = map->table[i];
    while (iter) {
      kvmap_put_entry(&(map->mm), &(iter->e), NULL);
      atomic_fetch_sub(&(map->nr_kv), 1);
      struct chainnode * const next = iter->next;
      free(iter);
      iter = next;
    }
    map->table[i] = NULL;
  }
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_unlock(&(map->hlocks[i]));
  }
  mutexlock_unlock(&(map->expand_lock));
}

  inline void
chainmap_destroy(struct chainmap * const map)
{
  chainmap_clean(map);
  pages_unmap(map->table, map->alloc_size);
  free(map);
}

  void
chainmap_fprint(struct chainmap * const map, FILE * const out)
{
  fprintf(out, "CHAINMAP POWER %"PRIu64" NR %"PRIu64"\n", map->power, atomic_load(&(map->nr_kv)));
}

struct chainmap_iter {
  struct chainmap * map;
  u64 cid;
  struct chainnode * curr;
};

  struct chainmap_iter *
chainmap_iter_create(struct chainmap * const map)
{
  mutexlock_lock(&(map->expand_lock));
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_lock(&(map->hlocks[i]));
  }
  struct chainmap_iter * const iter = (typeof(iter))malloc(sizeof(*iter));
  iter->map = map;
  u64 cid = 0;
  while (map->table[cid] == NULL) cid++;
  iter->curr = map->table[cid];
  iter->cid = cid;
  return iter;
}

  struct kv *
chainmap_iter_next(struct chainmap_iter * const iter, struct kv * const out)
{
  struct chainnode * curr = iter->curr;
  if (iter->curr == NULL) return NULL;
  struct kv * const ret = kv_dup2(__u64_to_ptr(curr->e.ptr), out);
  curr = curr->next;
  if (curr == NULL) {
    const u64 dirsize = UINT64_C(1) << iter->map->power;
    for (u64 cid = iter->cid + 1; cid < dirsize; cid++) {
      if (iter->map->table[cid]) {
        curr = iter->map->table[cid];
        iter->cid = cid;
        break;
      }
    }
  }
  iter->curr = curr;
  return ret;
}

  void
chainmap_iter_destroy(struct chainmap_iter * const iter)
{
  struct chainmap * const map = iter->map;
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_unlock(&(map->hlocks[i]));
  }
  mutexlock_unlock(&(map->expand_lock));
  free(iter);
}

  struct kvmap_api *
chainmap_api_create(struct chainmap * const map)
{
  if (map == NULL) return NULL;
  const struct kvmap_api api = {
    .map = map,
    .get = (typeof(api.get))chainmap_get,
    .probe = (typeof(api.probe))chainmap_probe,
    .set = (typeof(api.set))chainmap_set,
    .del = (typeof(api.del))chainmap_del,
    .clean = (typeof(api.clean))chainmap_clean,
    .destroy = (typeof(api.destroy))chainmap_destroy,
    .fprint = (typeof(api.fprint))chainmap_fprint,
    .iter_create = (typeof(api.iter_create))chainmap_iter_create,
    .iter_next = (typeof(api.iter_next))chainmap_iter_next,
    .iter_destroy = (typeof(api.iter_destroy))chainmap_iter_destroy,
  };
  struct kvmap_api * const papi = (typeof(papi))malloc(sizeof(*papi));
  *papi = api;
  return papi;
}
// }}} chainmap

// bptree {{{
#define BPTREE_INTM_FO ((250))
#define BPTREE_LEAF_FO (((BPTREE_INTM_FO) * (2)))
struct bpnode {
  bool leaf;
  u64 nr;
  struct bpnode * next;
  struct bpnode * prev;
  union {
    struct bpintm {
      const struct kv * key;
      struct bpnode * sub;
    } iarray[BPTREE_INTM_FO];
    const struct kv * karray[BPTREE_LEAF_FO];
  };
};


struct bptree {
  struct kvmap_mm mm;
  struct bpnode * root;
};

  static inline struct bpnode *
bptree_alloc_node(const bool leaf)
{
  struct bpnode * const node = pages_alloc_4kb(1);
  node->leaf = leaf;
  return node;
}

  struct bptree *
bptree_create(const struct kvmap_mm * const mm)
{
  debug_assert(sizeof(struct bpnode) < PGSZ);
  struct bptree * const tree = (typeof(tree))malloc(sizeof(*tree));
  if (tree == NULL) return NULL;
  tree->mm = mm ? (*mm) : __kvmap_mm_default;
  struct bpnode * const root = bptree_alloc_node(true);
  if (root == NULL) {
    free(tree);
    return NULL;
  }
  root->nr = 1;
  tree->root = root;
  return tree;
}

  static u64
bptree_leaf_locate(struct bpnode * const node, const struct kv * const key)
{
  if (node->nr < 2) {
    return 0;
  }
  u64 l = 0;
  u64 r = node->nr - 1;
  while (l < r) {
    const u64 m = (l + r + 1) >> 1;
    const int cmp = kv_keycompare(key, node->karray[m]);
    if (cmp < 0) {
      r = m - 1;
    } else if (cmp > 0) {
      l = m;
    } else {
      return m;
    }
  }
  return r;
}

  static u64
bptree_intm_locate(struct bpnode * const node, const struct kv * const key)
{
  if (node->nr < 2) {
    return 0;
  }
  u64 l = 0;
  u64 r = node->nr - 1;
  while (l < r) {
    const u64 m = (l + r + 1) >> 1;
    const int cmp = kv_keycompare(key, node->iarray[m].key);
    if (cmp < 0) {
      r = m - 1;
    } else if (cmp > 0) {
      l = m;
    } else {
      return m;
    }
  }
  return r;
}

  static inline struct kv *
bptree_get_leaf(struct bptree * const tree, struct bpnode * const node, const struct kv * const key, struct kv * const out)
{
  struct kv ** const kv = bsearch(&key, node->karray, node->nr, sizeof(&key), __kv_compare_pp);
  if (kv && (*kv)) {
    if (tree->mm.hf) tree->mm.hf(*kv, tree->mm.hp);
    struct kv * const ret = kv_dup2(*kv, out);
    return ret;
  }
  return NULL;
}

  static struct kv *
bptree_get_rec(struct bptree * const tree, struct bpnode * const node, const struct kv * const key, struct kv * const out)
{
  debug_assert(node);
  if (node->leaf) {
    return bptree_get_leaf(tree, node, key, out);
  } else {
    const u64 subidx = bptree_intm_locate(node, key);
    return bptree_get_rec(tree, node->iarray[subidx].sub, key, out);
  }
}

  struct kv *
bptree_get(struct bptree * const tree, const struct kv * const key, struct kv * const out)
{
  return bptree_get_rec(tree, tree->root, key, out);
}

  static inline bool
bptree_probe_leaf(struct bptree * const tree, struct bpnode * const node, const struct kv * const key)
{
  const struct kv ** const kv = bsearch(&key, node->karray, node->nr, sizeof(&key), __kv_compare_pp);
  if (kv && (*kv)) {
    if (tree->mm.hf) tree->mm.hf(*kv, tree->mm.hp);
    return true;
  } else {
    return false;
  }
}

  static bool
bptree_probe_rec(struct bptree * const tree, struct bpnode * const node, const struct kv * const key)
{
  debug_assert(node);
  if (node->leaf) {
    return bptree_probe_leaf(tree, node, key);
  } else {
    const u64 subidx = bptree_intm_locate(node, key);
    return bptree_probe_rec(tree, node->iarray[subidx].sub, key);
  }
}

  bool
bptree_probe(struct bptree * const tree, const struct kv * const key)
{
  return bptree_probe_rec(tree, tree->root, key);
}

  static void
bptree_raw_insert_leaf(struct bpnode * const leaf, const u64 idx, const struct kv * const kv)
{
  debug_assert(idx <= leaf->nr);
  memmove(&(leaf->karray[idx + 1]), &(leaf->karray[idx]), sizeof(leaf->karray[0]) * (leaf->nr - idx));
  leaf->karray[idx] = kv;
  leaf->nr++;
}

  static void
bptree_raw_insert_intm(struct bpnode * const node, const u64 idx, const struct kv * const key, struct bpnode * const sub)
{
  debug_assert(idx <= node->nr);
  memmove(&(node->iarray[idx + 1]), &(node->iarray[idx]), sizeof(node->iarray[0]) * (node->nr - idx));
  node->iarray[idx].key = key;
  node->iarray[idx].sub = sub;
  node->nr++;
  // link
  if (node->nr > 1) {
    if (idx > 0) {
      struct bpnode * const left = node->iarray[idx - 1].sub;
      if (left->next) {
        sub->next = left->next;
        left->next->prev = sub;
      }
      sub->prev = left;
      left->next = sub;
    } else {
      struct bpnode * const right = node->iarray[idx + 1].sub;
      if (right->prev) {
        sub->prev = right->prev;
        right->prev->next = sub;
      }
      sub->next = right;
      right->prev = sub;
    }
  }
}

  bool
bptree_raw_balance_leaf(struct bpnode * const left, struct bpnode * const right)
{
  const u64 nr_r = (left->nr + right->nr) >> 1;
  const u64 nr_l = (left->nr + right->nr) - nr_r;
  if (nr_l == left->nr) {
    return false;
  } else if (nr_l < left->nr) { // shift right
    const u64 rmore = nr_r - right->nr;
    memmove(&(right->karray[rmore]), &(right->karray[0]), sizeof(right->karray[0]) * right->nr);
    memmove(&(right->karray[0]), &(left->karray[nr_l]), sizeof(right->karray[0]) * rmore);
  } else { // shift left
    const u64 rless = nr_l - left->nr;
    memmove(&(left->karray[left->nr]), &(right->karray[0]), sizeof(right->karray[0]) * rless);
    memmove(&(right->karray[0]), &(right->karray[rless]), sizeof(right->karray[0]) * nr_r);
  }
  left->nr = nr_l;
  right->nr = nr_r;
  return true;
}

  bool
bptree_raw_balance_intm(struct bpnode * const left, struct bpnode * const right)
{
  const u64 nr_r = (left->nr + right->nr) >> 1;
  const u64 nr_l = (left->nr + right->nr) - nr_r;
  if (nr_l == left->nr) {
    return false;
  } else if (nr_l < left->nr) { // shift right
    const u64 rmore = nr_r - right->nr;
    memmove(&(right->iarray[rmore]), &(right->iarray[0]), sizeof(right->iarray[0]) * right->nr);
    memmove(&(right->iarray[0]), &(left->iarray[nr_l]), sizeof(right->iarray[0]) * rmore);
  } else { // shift left
    const u64 rless = nr_l - left->nr;
    memmove(&(left->iarray[left->nr]), &(right->iarray[0]), sizeof(right->iarray[0]) * rless);
    memmove(&(right->iarray[0]), &(right->iarray[rless]), sizeof(right->iarray[0]) * nr_r);
  }
  left->nr = nr_l;
  right->nr = nr_r;
  return true;
}

  static void
bptree_split_root_leaf(struct bptree * const tree)
{
  struct bpnode * const left = tree->root;
  struct bpnode * const right = bptree_alloc_node(true);
  (void)bptree_raw_balance_leaf(left, right);

  struct bpnode * const newroot = bptree_alloc_node(false);
  bptree_raw_insert_intm(newroot, 0, kv_dup_key(left->karray[0]), left);
  bptree_raw_insert_intm(newroot, 1, kv_dup_key(right->karray[0]), right);
  tree->root = newroot;
}

  static void
bptree_split_root_intm(struct bptree * const tree)
{
  struct bpnode * const left = tree->root;
  struct bpnode * const right = bptree_alloc_node(false);
  (void)bptree_raw_balance_intm(left, right);

  struct bpnode * const newroot = bptree_alloc_node(false);
  bptree_raw_insert_intm(newroot, 0, kv_dup_key(left->iarray[0].key), left);
  bptree_raw_insert_intm(newroot, 1, kv_dup_key(right->iarray[0].key), right);
  tree->root = newroot;
}

  static void
bptree_split_leaf(struct bpnode * const parent, const u64 subidx)
{
  debug_assert(parent->nr < BPTREE_INTM_FO);
  struct bpnode * const left = parent->iarray[subidx].sub;
  struct bpnode * const right = bptree_alloc_node(true);
  (void)bptree_raw_balance_leaf(left, right);

  const u64 insert = subidx + 1;
  bptree_raw_insert_intm(parent, insert, kv_dup_key(right->karray[0]), right);
}

  static void
bptree_split_intm(struct bpnode * const parent, const u64 subidx)
{
  debug_assert(parent->nr < BPTREE_INTM_FO);
  struct bpnode * const left = parent->iarray[subidx].sub;
  struct bpnode * const right = bptree_alloc_node(false);
  (void)bptree_raw_balance_intm(left, right);

  const u64 insert = subidx + 1;
  bptree_raw_insert_intm(parent, insert, kv_dup_key(right->iarray[0].key), right);
}

  static bool
bptree_set_leaf(struct bptree * const tree, struct bpnode * const leaf, const struct kv * const kv)
{
  debug_assert(leaf->nr < BPTREE_LEAF_FO);

  const u64 idx = bptree_leaf_locate(leaf, kv);
  if (kv_keymatch(kv, leaf->karray[idx])) {
    struct kv * const victim = (typeof(victim))(leaf->karray[idx]);
    if (victim && tree->mm.xf) tree->mm.xf(victim, kv, tree->mm.xp);
    else if (victim && tree->mm.rf) tree->mm.rf(victim, tree->mm.rp);
    leaf->karray[idx] = kv;
  } else {
    const u64 insert = idx + 1;
    bptree_raw_insert_leaf(leaf, insert, kv);
  }
  return true;
}

  static bool
bptree_set_rec(struct bptree * const tree, struct bpnode * const parent, const u64 subidx, const struct kv * const kv)
{
  debug_assert(parent->leaf == false);

  struct bpnode * const sub = parent->iarray[subidx].sub;
  if (sub->leaf) {
    if (sub->nr == BPTREE_LEAF_FO) {
      bptree_split_leaf(parent, subidx);
      const u64 subidx1 = bptree_intm_locate(parent, kv);
      return bptree_set_rec(tree, parent, subidx1, kv);
    } else {
      return bptree_set_leaf(tree, sub, kv);
    }
  } else {
    if (sub->nr == BPTREE_INTM_FO) {
      bptree_split_intm(parent, subidx);
      const u64 subidx1 = bptree_intm_locate(parent, kv);
      return bptree_set_rec(tree, parent, subidx1, kv);
    } else {
      const u64 subsubidx = bptree_intm_locate(sub, kv);
      return bptree_set_rec(tree, sub, subsubidx, kv);
    }
  }
}

  bool
bptree_set(struct bptree * const tree, const struct kv * const kv0)
{
  const struct kv * const kv = kv_alloc_dup(kv0, &(tree->mm));
  if (kv == NULL) return false;

  if ((tree->root->leaf) && (tree->root->nr == BPTREE_LEAF_FO)) {
    bptree_split_root_leaf(tree);
  }
  if (tree->root->leaf) {
    return bptree_set_leaf(tree, tree->root, kv);
  }
  // intm
  if (tree->root->nr == BPTREE_INTM_FO) {
    bptree_split_root_intm(tree);
  }
  const u64 subidx = bptree_intm_locate(tree->root, kv);
  return bptree_set_rec(tree, tree->root, subidx, kv);
}

  bool
bptree_del(struct bptree * const tree, const struct kv * const key)
{
  // TODO:
  (void)tree;
  (void)key;
  return false;
}

  void
bptree_clean_rec(struct bptree * const tree, struct bpnode * const node)
{
  if (node->leaf) {
    for (u64 i = 0; i < node->nr; i++) {
      struct kv * const victim = (typeof(victim))(node->karray[i]);
      if (victim && tree->mm.rf) tree->mm.rf(victim, tree->mm.rp);
    }
  } else {
    for (u64 i = 0; i < node->nr; i++) {
      struct kv * const victim = (typeof(victim))(node->iarray[i].key);
      free(victim);
      bptree_clean_rec(tree, node->iarray[i].sub);
    }
  }
  pages_unmap(node, PGSZ);
}

  void
bptree_clean(struct bptree * const tree)
{
  bptree_clean_rec(tree, tree->root);
  tree->root = bptree_alloc_node(true);
  tree->root->nr = 1;
}

  void
bptree_destroy(struct bptree * const tree)
{
  bptree_clean(tree);
  pages_unmap(tree->root, PGSZ);
  free(tree);
}

  void
bptree_fprint(struct bptree * const tree, FILE * const out)
{
  //TODO:
  (void)tree;
  (void)out;
}

struct bptree_iter {
  struct bpnode * curr;
  u64 i;
};

  struct bptree_iter *
bptree_iter_create(struct bptree * const tree)
{
  struct bptree_iter * const iter = (typeof(iter))malloc(sizeof(*iter));
  struct bpnode * node = tree->root;
  while (node->leaf == false) {
    node = node->iarray[0].sub;
  }
  u64 i = 1;
  while (i >= node->nr) {
    i = 0;
    node = node->next;
  }
  iter->curr = node;
  iter->i = i;
  debug_assert(node->karray[0] == NULL);
  return iter;
}

  struct kv *
bptree_iter_next(struct bptree_iter * const iter, struct kv * const out)
{
  struct bpnode * curr = iter->curr;
  if (curr == NULL) {
    return NULL;
  }
  debug_assert(curr->nr);
  u64 i = iter->i;
  struct kv * const ret = kv_dup2(curr->karray[i], out);
  i++;
  if (i >= curr->nr) {
    i = 0;
    curr = curr->next;
    iter->curr = curr;
  }
  iter->i = i;
  return ret;
}

  void
bptree_iter_destroy(struct bptree_iter * const iter)
{
  free(iter);
}

  struct kvmap_api *
bptree_api_create(struct bptree * const map)
{
  if (map == NULL) return NULL;
  const struct kvmap_api api = {
    .map = map,
    .get = (typeof(api.get))bptree_get,
    .probe = (typeof(api.probe))bptree_probe,
    .set = (typeof(api.set))bptree_set,
    .del = (typeof(api.del))bptree_del,
    .clean = (typeof(api.clean))bptree_clean,
    .destroy = (typeof(api.destroy))bptree_destroy,
    .fprint = (typeof(api.fprint))bptree_fprint,
    .iter_create = (typeof(api.iter_create))bptree_iter_create,
    .iter_next = (typeof(api.iter_next))bptree_iter_next,
    .iter_destroy = (typeof(api.iter_destroy))bptree_iter_destroy,
  };
  struct kvmap_api * const papi = (typeof(papi))malloc(sizeof(*papi));
  *papi = api;
  return papi;
}
// }}} bptree

// mica {{{
#define MICA_ENTRY_NR ((7))
struct mica_slot {
  au32 version;
  u32 chain_id;
  struct __kvmap_entry e[MICA_ENTRY_NR];
};

#define MICA_SPARE_NR ((UINT64_C(256)))
struct mica {
  struct kvmap_mm mm; // alloc/free are ignored
  spinlock hlocks[KVMAP_HLOCK_NR]; // writer's lock
  u64 alloc_size;
  u64 power;
  u64 table_nr;
  struct mica_slot * table;
  struct gcache * gc;
  u64 padding00[8];
  spinlock spare_lock;
  u64 spare_next;
  u64 padding01[8];
  struct mica_slot * spare[MICA_SPARE_NR];
  u64 spare_size[MICA_SPARE_NR];
};

  struct mica *
mica_create(const struct kvmap_mm * const mm, const u64 power)
{
  debug_assert((sizeof(struct mica_slot) % 64) == 0);
  struct mica * const map = (typeof(map))calloc(1, sizeof(*map));
  debug_assert(map);
  memset(map, 0, sizeof(*map));
  map->mm = mm ? (*mm) : __kvmap_mm_default;
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_init(&(map->hlocks[i]));
  }
  map->power = power;
  if (map->power < KVMAP_HLOCK_POWER) map->power = KVMAP_HLOCK_POWER;
  if (map->power > 46) map->power = 46;
  map->table_nr = UINT64_C(1) << map->power;
  const u64 table_size = sizeof(map->table[0]) * map->table_nr;
  map->table = (typeof(map->table))pages_alloc_best(table_size, true, &(map->alloc_size));
  map->gc = gcache_create(128, 8);
  spinlock_init(&(map->spare_lock));
  map->spare_next = 1;
  return map;
}

  static struct mica_slot *
mica_iter(struct mica * const map, struct mica_slot * const curr)
{
  const u32 next_id = curr->chain_id;
  if (next_id) {
    const u32 spare_idx = next_id / map->table_nr;
    const u32 spare_off = next_id % map->table_nr;
    if (spare_idx < MICA_SPARE_NR) {
      struct mica_slot * const next = &(map->spare[spare_idx][spare_off]);
      __builtin_prefetch(next, 0, 1);
      return next;
    }
  }
  return NULL;
}

  static bool
mica_get_try(struct mica * const map, const struct kv * const key, struct kv * const out, struct kv ** const pret)
{
  const u64 hash = key->hash;
  const u64 cid = (hash >> KVMAP_BASE_LEVEL) % map->table_nr;
  struct mica_slot * const head = &(map->table[cid]);
  const u32 v0 = atomic_load(&(head->version));
  if (v0 & UINT32_C(1)) return false;

  const u64 pkey = kvmap_pkey(key->hash);
  struct mica_slot * iter = head;
  while (iter) {
    struct mica_slot * const next = mica_iter(map, iter);
    for (u64 i = 0; i < MICA_ENTRY_NR; i++) {
      if (iter->e[i].ptr && iter->e[i].pkey == pkey) {
        const struct kv * const curr = __u64_to_ptr(iter->e[i].ptr);
        if (kv_keymatch(key, curr)) {
          struct kv * const ret = kv_dup2(curr, out);
          const u32 v1 = atomic_load(&(head->version));
          if (v1 != v0) {
            if (out == NULL) free(ret);
            return false; // abort
          } else {
            // TODO: cached kv might be stale
            if (map->mm.hf) map->mm.hf(curr, map->mm.hp);
            *pret = ret;
            return true;
          }
        }
      }
    }
    iter = next;
  }
  *pret = NULL;
  return true;
}

  struct kv *
mica_get(struct mica * const map, const struct kv * const key, struct kv * const out)
{
  struct kv * ret = NULL;
  while (mica_get_try(map, key, out, &ret) == false);
  return ret;
}

  static bool
mica_probe_try(struct mica * const map, const struct kv * const key, bool * const pret)
{
  const u64 hash = key->hash;
  const u64 cid = (hash >> KVMAP_BASE_LEVEL) % map->table_nr;
  struct mica_slot * const head = &(map->table[cid]);
  const u32 v0 = atomic_load(&(head->version));
  if (v0 & UINT32_C(1)) return false;

  const u64 pkey = kvmap_pkey(key->hash);
  struct mica_slot * iter = head;
  while (iter) {
    struct mica_slot * const next = mica_iter(map, iter);
    for (u64 i = 0; i < MICA_ENTRY_NR; i++) {
      if (iter->e[i].ptr && iter->e[i].pkey == pkey) {
        const struct kv * const curr = __u64_to_ptr(iter->e[i].ptr);
        if (kv_keymatch(key, curr)) {
          const u32 v1 = atomic_load(&(head->version));
          if (v1 != v0) {
            return false; // abort
          } else {
            // TODO: cached kv might be stale
            if (map->mm.hf) map->mm.hf(curr, map->mm.hp);
            *pret = true;
            return true;
          }
        }
      }
    }
    iter = next;
  }
  *pret = false;
  return true;
}

  bool
mica_probe(struct mica * const map, const struct kv * const key)
{
  bool ret = false;
  while (mica_probe_try(map, key, &ret) == false);
  return ret;
}

  static struct kv *
mica_alloc_dup(const struct kv * const kv, struct mica * const map)
{
  struct kv * const kv1 = gcache_pull(map->gc, kv_size_align(kv, 8));
  if (kv1) {
    return kv_dup2(kv, kv1);
  }

  struct kv * const kv2 = kv_alloc_dup(kv, &(map->mm));
  return kv2;
}

  static void
mica_retire(struct kv * const kv, struct mica * const map)
{
  if (map->mm.uf) map->mm.uf(kv, map->mm.up);
  gcache_push(map->gc, kv_size_align(kv, 8), kv);
}

  static void
mica_put_entry(struct mica * const map, struct __kvmap_entry * const e, const struct kv * const kv)
{
  struct kv * const old = __u64_to_ptr(e->ptr);
  if (old) {
    mica_retire(old, map);
  }
  e->ptr = __ptr_to_u64(kv);
  e->pkey = kv ? kvmap_pkey(kv->hash) : 0;
}

  static struct mica_slot *
mica_expand(struct mica * const map, struct mica_slot * const iter)
{
  spinlock_lock(&(map->spare_lock));
  const u64 next_id = map->spare_next;
  map->spare_next++;

  const u32 idx = next_id / map->table_nr;
  if (idx >= MICA_SPARE_NR) {
    spinlock_unlock(&(map->spare_lock));
    return NULL;
  }
  if (map->spare[idx] == NULL) {
    const u64 size = sizeof(map->spare[idx][0]) * map->table_nr;
    map->spare[idx] = (typeof(map->spare[idx]))pages_alloc_best(size, true, &(map->spare_size[idx]));
    if (map->spare[idx] == NULL) {
      spinlock_unlock(&(map->spare_lock));
      return NULL;
    }
  }
  iter->chain_id = next_id;
  spinlock_unlock(&(map->spare_lock));
  return mica_iter(map, iter);
}

  bool
mica_set(struct mica * const map, const struct kv * const kv0)
{
  // force
  struct kv * const kv = mica_alloc_dup(kv0, map);
  if (kv == NULL) return false;

  const u64 hash = kv->hash;
  const u64 lidx = kvmap_hlocks_index(hash);
  const u64 pkey = kvmap_pkey(hash);
  const u64 cid = (hash >> KVMAP_BASE_LEVEL) % map->table_nr;
  struct mica_slot * const head = &(map->table[cid]);

  spinlock_lock(&(map->hlocks[lidx]));
  const u32 v0 = atomic_fetch_add(&(head->version), 1);
  (void)v0;
  debug_assert((v0 & (UINT32_C(1))) == 0);

  struct mica_slot * iter = head;
  // try update
  while (iter) {
    struct mica_slot * const next = mica_iter(map, iter);
    for (u64 i = 0; i < MICA_ENTRY_NR; i++) {
      if (iter->e[i].ptr && iter->e[i].pkey == pkey) {
        const struct kv * const curr = __u64_to_ptr(iter->e[i].ptr);
        if (kv_keymatch(kv, curr)) {
          mica_put_entry(map, &(iter->e[i]), kv);
          const u32 v1 = atomic_fetch_add(&(head->version), 1);
          (void)v1;
          debug_assert((v0 + 1) == v1);
          spinlock_unlock(&(map->hlocks[lidx]));
          return true;
        }
      }
    }
    iter = next;
  }
  // try insert
  iter = head;
  while (iter) {
    for (u64 i = 0; i < MICA_ENTRY_NR; i++) {
      if (iter->e[i].ptr == 0) {
        mica_put_entry(map, &(iter->e[i]), kv);
        const u32 v1 = atomic_fetch_add(&(head->version), 1);
        (void)v1;
        debug_assert((v0 + 1) == v1);
        spinlock_unlock(&(map->hlocks[lidx]));
        return true;
      }
    }
    struct mica_slot * const next = mica_iter(map, iter);
    if (next == NULL) {
      iter = mica_expand(map, iter);
      if (iter == NULL) {
        mica_retire(kv, map);
        const u32 v1 = atomic_fetch_add(&(head->version), 1);
        (void)v1;
        debug_assert((v0 + 1) == v1);
        spinlock_unlock(&(map->hlocks[lidx]));
        return false;
      }
    } else {
      iter = next;
    }
  }
  spinlock_unlock(&(map->hlocks[lidx]));
  const u32 v1 = atomic_fetch_add(&(head->version), 1);
  (void)v1;
  debug_assert((v0 + 1) == v1);
  debug_assert(false);
  return false;
}

  bool
mica_del(struct mica * const map, const struct kv * const key)
{
  const u64 hash = key->hash;
  const u64 lidx = kvmap_hlocks_index(hash);
  const u64 pkey = kvmap_pkey(hash);
  const u64 cid = (hash >> KVMAP_BASE_LEVEL) % map->table_nr;
  struct mica_slot * const head = &(map->table[cid]);

  spinlock_lock(&(map->hlocks[lidx]));
  const u32 v0 = atomic_fetch_add(&(head->version), 1);
  (void)v0;
  debug_assert((v0 & (UINT32_C(1))) == 0);

  struct mica_slot * iter = head;
  while (iter) {
    struct mica_slot * const next = mica_iter(map, iter);
    for (u64 i = 0; i < MICA_ENTRY_NR; i++) {
      if (iter->e[i].ptr && iter->e[i].pkey == pkey) {
        const struct kv * const curr = __u64_to_ptr(iter->e[i].ptr);
        if (kv_keymatch(key, curr)) {
          mica_put_entry(map, &(iter->e[i]), NULL);
          const u32 v1 = atomic_fetch_add(&(head->version), 1);
          (void)v1;
          debug_assert((v0 + 1) == v1);
          spinlock_unlock(&(map->hlocks[lidx]));
          return true;
        }
      }
    }
    iter = next;
  }
  const u32 v1 = atomic_fetch_add(&(head->version), 1);
  (void)v1;
  debug_assert((v0 + 1) == v1);
  spinlock_unlock(&(map->hlocks[lidx]));
  return false;
}

  void
mica_clean(struct mica * const map)
{
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_lock(&(map->hlocks[i]));
  }
  for (u64 i = 0; i < map->table_nr; i++) {
    struct mica_slot * const head = &(map->table[i]);
    const u32 v0 = atomic_fetch_add(&(head->version), 1);
    (void)v0;
    struct mica_slot * iter = head;
    while (iter) {
      struct mica_slot * const next = mica_iter(map, iter);
      for (u64 i = 0; i < MICA_ENTRY_NR; i++) {
        if (iter->e[i].ptr) mica_put_entry(map, &(iter->e[i]), NULL);
      }
      iter = next;
    }
    const u32 v1 = atomic_fetch_add(&(head->version), 1);
    (void)v1;
    debug_assert((v0 + 1) == v1);
  }
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_unlock(&(map->hlocks[i]));
  }
}

  static void
mica_destroy_gcache(struct mica * const map)
{
  if (map->mm.rf) {
    struct gcache_iter * const gi = gcache_iter_create(map->gc);
    for (void * ptr = gcache_iter_next(gi); ptr; ptr = gcache_iter_next(gi)) {
      map->mm.rf(ptr, map->mm.rp);
    }
    gcache_iter_destroy(gi);
  }
  gcache_destroy(map->gc);
}

  void
mica_destroy(struct mica * const map)
{
  mica_clean(map);
  pages_unmap(map->table, map->alloc_size);
  for (u64 i = 0; i < MICA_SPARE_NR; i++) {
    pages_unmap(map->spare[i], map->spare_size[i]);
  }
  mica_destroy_gcache(map);
  free(map);
}

  void
mica_fprint(struct mica * const map, FILE * const out)
{
  fprintf(out, "MICA spare_next %"PRIu64"\n", map->spare_next);
}

  struct kvmap_api *
mica_api_create(struct mica * const map)
{
  if (map == NULL) return NULL;
  const struct kvmap_api api = {
    .map = map,
    .get = (typeof(api.get))mica_get,
    .probe = (typeof(api.probe))mica_probe,
    .set = (typeof(api.set))mica_set,
    .del = (typeof(api.del))mica_del,
    .clean = (typeof(api.clean))mica_clean,
    .fprint = (typeof(api.fprint))mica_fprint,
    .destroy = (typeof(api.destroy))mica_destroy,
  };
  struct kvmap_api * const papi = (typeof(papi))malloc(sizeof(*papi));
  *papi = api;
  return papi;
}
// }}} mica

// tcpmap {{{
struct tcpmap {
  struct stream2 * stream2;
  bool ok;
};

  struct tcpmap *
tcpmap_create(const char * const host, const int port)
{
  struct stream2 * const s = stream2_create(host, port);
  if (s == NULL) {
    return NULL;
  }

  struct tcpmap * const tcpmap = (typeof(tcpmap))malloc(sizeof(*tcpmap));
  tcpmap->stream2 = s;
  tcpmap->ok = true;
  return tcpmap;
}

  static bool
__tcpmap_request(const char op, struct tcpmap * const map, const struct kv * const kv, bool * const ret)
{
  if (map->ok == false) {
    return false;
  }
  if (fwrite(&op, sizeof(op), 1, map->stream2->w) != 1) {
    map->ok = false;
    return false;
  }
  const size_t sz = (op == 'S') ? kv_size(kv) : key_size(kv);
  if (fwrite(kv, sz, 1, map->stream2->w) != 1) {
    map->ok = false;
    return false;
  }
  fflush(map->stream2->w);

  if (fread(ret, sizeof(*ret), 1, map->stream2->r) != 1) {
    map->ok = false;
    return false;
  }
  return true;
}

  struct kv *
tcpmap_get(struct tcpmap * const map, const struct kv * const key, struct kv * const out)
{
  bool ret = false;
  const bool req = __tcpmap_request('G', map, key, &ret);
  if (req == false || ret == false) {
    return NULL;
  }

  struct kv head = {};
  if (fread(&head, sizeof(head), 1, map->stream2->r) != 1) {
    map->ok = false;
    return NULL;
  }
  const bool alloc = (out == NULL) ? true : false;
  struct kv * const kv = alloc ? (typeof(kv))malloc(kv_size(&head)) : out;
  if (kv == NULL) {
    return NULL;
  }
  memcpy(kv, &head, sizeof(head));
  if (fread(kv->kv, kv->klen + kv->vlen, 1, map->stream2->r) != 1) {
    map->ok = false;
    if (alloc) free(kv);
    return NULL;
  }
  return kv;
}

  bool
tcpmap_probe(struct tcpmap * const map, const struct kv * const key)
{
  bool ret = false;
  const bool req = __tcpmap_request('P', map, key, &ret);
  return req && ret;
}

  bool
tcpmap_set(struct tcpmap * const map, const struct kv * const kv)
{
  bool ret = false;
  const bool req = __tcpmap_request('S', map, kv, &ret);
  return req && ret;
}

  bool
tcpmap_del(struct tcpmap * const map, const struct kv * const key)
{
  bool ret = false;
  const bool req = __tcpmap_request('D', map, key, &ret);
  return req && ret;
}

  void
tcpmap_clean(struct tcpmap * const map)
{
  if (map->ok == false) return;
  const char op = 'C';
  if (fwrite(&op, sizeof(op), 1, map->stream2->w) != 1) {
    map->ok = false;
    return;
  }
  fflush(map->stream2->w);
  bool ret = false;
  if (fread(&ret, sizeof(ret), 1, map->stream2->r) != 1) {
    map->ok = false;
  }
}

  void
tcpmap_destroy(struct tcpmap * const map)
{
  stream2_destroy(map->stream2);
  free(map);
}

  struct kvmap_api *
tcpmap_api_create(struct tcpmap * const map)
{
  if (map == NULL) return NULL;
  const struct kvmap_api api = {
    .map = map,
    .get = (typeof(api.get))tcpmap_get,
    .probe = (typeof(api.probe))tcpmap_probe,
    .set = (typeof(api.set))tcpmap_set,
    .del = (typeof(api.del))tcpmap_del,
    .clean = (typeof(api.clean))tcpmap_clean,
    .destroy = (typeof(api.destroy))tcpmap_destroy,
  };
  struct kvmap_api * const papi = (typeof(papi))malloc(sizeof(*papi));
  *papi = api;
  return papi;
}
// }}} tcpmap

// icache {{{
//#define ICACHE_ENTRY_NR ((15))
#define ICACHE_ENTRY_NR ((7))
#define ICACHE_AMAX ((UINT64_C(15)))
struct __icache_slot {
  u64 amap:60; // access count, four bits per slot;
  u64 used:4; // number of free slots, 15 -- 0;
  struct __kvmap_entry e[ICACHE_ENTRY_NR];
};

#define ICACHE_HIST_NR ((7))
struct __icache_hist {
  u16 head;
  u16 tags[ICACHE_HIST_NR];
};

struct icache {
  struct kvmap_mm mm;
  struct kvmap_mm map_mm;
  u64 padding00[8];
  spinlock hlocks[KVMAP_HLOCK_NR];
  u64 padding10[8];

  u64 nr_slots;
  u64 slots_alloc_size;
  struct __icache_slot * table;

  u64 nr_hists;
  u64 hists_alloc_size;
  struct __icache_hist * hists;

  struct kvmap_api * map_api;
};

  struct icache *
icache_create(const struct kvmap_mm * const mm, const u64 nr_mb)
{
  debug_assert(ICACHE_ENTRY_NR == 15 || ICACHE_ENTRY_NR == 7);
  debug_assert((sizeof(struct __icache_slot) % 64) == 0);
  debug_assert((sizeof(struct __icache_hist) % 8) == 0);

  struct icache * const cache = (typeof(cache))malloc(sizeof(*cache));
  memset(cache, 0, sizeof(*cache));
  cache->mm = mm ? (*mm) : __kvmap_mm_default;
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_init(&(cache->hlocks[i]));
  }
  const u64 table_size = nr_mb << 20;
  const u64 nr_slots = table_size / sizeof(cache->table[0]);
  struct __icache_slot * const table = pages_alloc_best(table_size, false, &(cache->slots_alloc_size));
  if (table == NULL) {
    free(cache);
    return NULL;
  }
  debug_assert(nr_slots % KVMAP_HLOCK_NR == 0);
  cache->table = table;
  cache->nr_slots = nr_slots;

  const u64 nr_hists = nr_slots * 16;
  const u64 hists_size = nr_hists * sizeof(cache->hists[0]);
  struct __icache_hist * const hists = pages_alloc_best(hists_size, false, &(cache->hists_alloc_size));
  if (hists == NULL) {
    pages_unmap(table, cache->slots_alloc_size);
    free(cache);
    return NULL;
  }

  cache->nr_hists = nr_hists;
  cache->hists = hists;
  return cache;
}

  static inline void
icache_lock_hash(struct icache * const cache, const u64 hash)
{
  spinlock_lock(&(cache->hlocks[kvmap_hlocks_index(hash)]));
}

  static inline void
icache_unlock_hash(struct icache * const cache, const u64 hash)
{
  spinlock_unlock(&(cache->hlocks[kvmap_hlocks_index(hash)]));
}

  static inline void
icache_amap_inc(struct __icache_slot * const slot, const u64 i)
{
  const u64 v = (slot->amap >> (i << 2)) & ICACHE_AMAX;
  if (v < ICACHE_AMAX) {
    slot->amap += (UINT64_C(1) << (i << 2));
  }
}

  static inline void
icache_amap_set(struct __icache_slot * const slot, const u64 i, const u64 c)
{
  const u64 mask = ICACHE_AMAX << (i << 2);
  slot->amap = (slot->amap & (~mask)) | ((c << (i << 2)) & mask);
}

  static inline void
icache_amap_dec(struct __icache_slot * const slot, const u64 i)
{
  const u64 v = (slot->amap >> (i << 2)) & ICACHE_AMAX;
  if (v) {
    slot->amap -= (UINT64_C(1) << (i << 2));
  }
}

  static inline void
icache_amap_clear(struct __icache_slot * const slot, const u64 i)
{
  slot->amap &= (~(ICACHE_AMAX << (i << 2)));
}

  static inline u64
icache_amap_get(struct __icache_slot * const slot, const u64 i)
{
  return (slot->amap >> (i << 2)) & ICACHE_AMAX;
}

  static inline u64
icache_hist_update(struct __icache_hist * const hist, const u64 pkey)
{
  u64 rep = 0;
  for (u64 i = 0; i < ICACHE_HIST_NR; i++) {
    if (pkey == hist->tags[i]) {
      rep++;
    }
  }
  hist->tags[hist->head] = pkey;
  hist->head = (hist->head + 1) % ICACHE_HIST_NR;
  return rep;
}

  static inline void
icache_put_entry(struct __kvmap_entry * const e, const struct kv * const kv)
{
  e->ptr = __ptr_to_u64(kv);
  e->pkey = kv ? kvmap_pkey(kv->hash) : 0;
}

  static inline bool
icache_update(struct __icache_slot * const slot, const u64 rep, const struct kv * const kv)
{
  if (slot->used < ICACHE_ENTRY_NR) {
    for (u64 i = 0; i < ICACHE_ENTRY_NR; i++) {
      if (slot->e[i].ptr == 0) {
        icache_put_entry(&(slot->e[i]), kv);
        slot->used++;
        return false;
      }
    }
    debug_assert(false);
  }

  const u64 victim = random_u64() % ICACHE_ENTRY_NR;
  icache_amap_dec(slot, victim);

  for (u64 i = 0; i < ICACHE_ENTRY_NR; i++) {
    const u64 vi = (i + victim) % ICACHE_ENTRY_NR;
    const u64 rep1 = icache_amap_get(slot, vi);
    if (rep1 < rep) {
      icache_put_entry(&(slot->e[vi]), kv);
      icache_amap_set(slot, vi, rep);
      return true;
    }
  }
  return false;
}

  struct kv *
icache_get(struct icache * const cache, const struct kv * const key, struct kv * const out)
{
  const u64 hash = key->hash;
  const u64 idx = (hash >> KVMAP_BASE_LEVEL) % cache->nr_slots;
  struct __icache_slot * const slot = &(cache->table[idx]);
  __builtin_prefetch(slot, 0, 3);
  const u64 pkey = kvmap_pkey(hash);
  icache_lock_hash(cache, hash);
  for (u64 i = 0; i < ICACHE_ENTRY_NR; i++) {
    if (slot->e[i].pkey == pkey) {
      struct kv * const curr = __u64_to_ptr(slot->e[i].ptr);
      if (kv_keymatch(curr, key)) {
        icache_amap_inc(slot, i);
        struct kv * const ret = kv_dup2(curr, out);
        icache_unlock_hash(cache, hash);
        // cache hit
        return ret;
      }
    }
  }
  struct kv * const ret2 = cache->map_api->get(cache->map_api->map, key, out);
  // index hit/miss
  icache_unlock_hash(cache, hash);
  return ret2;
}

  bool
icache_probe(struct icache * const cache, const struct kv * const key)
{
  const u64 hash = key->hash;
  const u64 idx = (hash >> KVMAP_BASE_LEVEL) % cache->nr_slots;
  struct __icache_slot * const slot = &(cache->table[idx]);
  __builtin_prefetch(slot, 0, 3);
  const u64 pkey = kvmap_pkey(hash);
  icache_lock_hash(cache, hash);
  for (u64 i = 0; i < ICACHE_ENTRY_NR; i++) {
    if (slot->e[i].pkey == pkey) {
      struct kv * const curr = __u64_to_ptr(slot->e[i].ptr);
      if (kv_keymatch(curr, key)) {
        icache_amap_inc(slot, i);
        // cache hit
        icache_unlock_hash(cache, hash);
        return true;
      }
    }
  }
  const bool r = cache->map_api->probe(cache->map_api->map, key);
  // index hit/miss
  icache_unlock_hash(cache, hash);
  return r;
}

  static inline void
icache_invalidate(struct icache * const cache, const struct kv * const key)
{
  const u64 hash = key->hash;
  const u64 idx = (hash >> KVMAP_BASE_LEVEL) % cache->nr_slots;
  struct __icache_slot * const slot = &(cache->table[idx]);
  const u64 pkey = kvmap_pkey(hash);
  for (u64 i = 0; i < ICACHE_ENTRY_NR; i++) {
    if ((slot->e[i].pkey == pkey) && slot->e[i].ptr) {
      icache_put_entry(&(slot->e[i]), NULL);
      slot->used--;
      icache_amap_clear(slot, i);
    }
  }
}

  static inline void
icache_refresh(struct icache * const cache, const struct kv * const old, const struct kv * const new)
{
  const u64 hash = new->hash;
  const u64 idx = (hash >> KVMAP_BASE_LEVEL) % cache->nr_slots;
  struct __icache_slot * const slot = &(cache->table[idx]);
  for (u64 i = 0; i < ICACHE_ENTRY_NR; i++) {
    struct kv * const curr = __u64_to_ptr(slot->e[i].ptr);
    if (curr == old) {
      icache_put_entry(&(slot->e[i]), new);
      return;
    }
  }
}

  bool
icache_set(struct icache * const cache, const struct kv * const kv)
{
  const u64 hash = kv->hash;
  icache_lock_hash(cache, hash);
  const bool ret = cache->map_api->set(cache->map_api->map, kv);
  icache_unlock_hash(cache, hash);
  return ret;
}

  bool
icache_del(struct icache * const cache, const struct kv * const key)
{
  const u64 hash = key->hash;
  icache_lock_hash(cache, hash);
  const bool ret = cache->map_api->del(cache->map_api->map, key);
  icache_unlock_hash(cache, hash);
  return ret;
}

  void
icache_clean(struct icache * const cache)
{
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_lock(&(cache->hlocks[i]));
  }
  if (cache->map_api) {
    cache->map_api->clean(cache->map_api->map);
  }
  for (u64 i = 0; i < cache->nr_slots; i++) {
    struct __icache_slot * const slot = &(cache->table[i]);
    memset(slot, 0, sizeof(*slot));
  }
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_unlock(&(cache->hlocks[i]));
  }
}

  void
icache_destroy(struct icache * const cache)
{
  icache_clean(cache);
  pages_unmap(cache->table, cache->slots_alloc_size);
  pages_unmap(cache->hists, cache->hists_alloc_size);
  if (cache->map_api) {
    kvmap_api_destroy(cache->map_api);
  }
  free(cache);
}

  void
icache_fprint(struct icache * const cache, FILE * const out)
{
  const u64 cap = cache->nr_slots * ICACHE_ENTRY_NR;
  u64 nr = 0;
  for (u64 i = 0; i < cache->nr_slots; i++) {
    struct __icache_slot * const slot = &(cache->table[i]);
    for (u64 j = 0; j < ICACHE_ENTRY_NR; j++) {
      if (slot->e[j].ptr) nr++;
    }
  }
  //for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
  //  fprintf(out, "%"PRIu64"%c", cache->hlocks[i].padding[7], ((i % 32) == 31) ? '\n' : ' ');
  //}
  fprintf(out, "ICACHE CAP %"PRIu64"/%"PRIu64"\n", nr, cap);
  if (cache->map_api && cache->map_api->fprint) {
    cache->map_api->fprint(cache->map_api->map, out);
  }
}

struct icache_iter {
  struct kvmap_api * map_api;
  void * map_iter;
};

  struct icache_iter *
icache_iter_create(struct icache * const cache)
{
  if (cache->map_api == NULL) return NULL;
  void * const map_iter = cache->map_api->iter_create(cache->map_api->map);
  struct icache_iter * const iter = (typeof(iter))malloc(sizeof(*iter));
  iter->map_api = cache->map_api;
  iter->map_iter = map_iter;
  return iter;
}

  struct kv *
icache_iter_next(struct icache_iter * const iter, struct kv * const out)
{
  if (iter && iter->map_api) {
    return iter->map_api->iter_next(iter->map_iter, out);
  } else {
    return NULL;
  }
}

  void
icache_iter_destroy(struct icache_iter * const iter)
{
  if (iter && iter->map_api) {
    iter->map_api->iter_destroy(iter->map_iter);
  }
  free(iter);
}

  inline void
icache_retire(struct icache * const cache, struct kv * const kv)
{
  icache_invalidate(cache, kv);
  if (cache->map_mm.rf) cache->map_mm.rf(kv, cache->map_mm.rp);
}

  static void
__icache_kv_retire_cb(struct kv * const kv, void * const priv)
{
  struct icache * const cache = (typeof(cache))priv;
  icache_retire(cache, kv);
}

  inline void
icache_uncache(struct icache * const cache, struct kv * const kv)
{
  icache_invalidate(cache, kv);
  if (cache->map_mm.uf) cache->map_mm.uf(kv, cache->map_mm.up);
}

  static void
__icache_kv_uncache_cb(struct kv * const kv, void * const priv)
{
  struct icache * const cache = (typeof(cache))priv;
  icache_uncache(cache, kv);
}

  static void
__icache_kv_refresh_cb(struct kv * const old, const struct kv * const new, void * const priv)
{
  struct icache * const cache = (typeof(cache))priv;
  icache_refresh(cache, old, new);
  icache_retire(cache, old);
}

  inline void
icache_hint(struct icache * const cache, const struct kv * const kv)
{
  const u64 hash = kv->hash;
  const u64 idx = (hash >> KVMAP_BASE_LEVEL) % cache->nr_slots;
  struct __icache_slot * const slot = &(cache->table[idx]);
  if ((slot->used < ICACHE_ENTRY_NR) || ((random_u64() & 0xfff) < 0x80)) {
    struct __icache_hist * const hist = &(cache->hists[(hash >> KVMAP_BASE_LEVEL) % cache->nr_hists]);
    const u64 rep = icache_hist_update(hist, kvmap_pkey(hash));
    if (rep) {
      if (icache_update(slot, rep, kv)) {
        // replaced
      }
    }
  }
  if (cache->map_mm.hf) cache->map_mm.hf(kv, cache->map_mm.hp);
}

  static void
__icache_kv_hit_cb(const struct kv * const kv, void * const priv)
{
  struct icache * const cache = (typeof(cache))priv;
  icache_hint(cache, kv);
}

  inline void
icache_wrap_mm(struct icache * const cache, struct kvmap_mm * const mm)
{
  cache->map_mm = *mm;
  if (mm->rf) {
    mm->rf = __icache_kv_retire_cb;
    mm->rp = cache;
    mm->uf = __icache_kv_uncache_cb;
    mm->up = cache;
  }
  mm->hf = __icache_kv_hit_cb;
  mm->hp = cache;
  mm->xf = __icache_kv_refresh_cb;
  mm->xp = cache;
}

  inline void
icache_wrap_kvmap(struct icache * const cache, struct kvmap_api * const map_api)
{
  cache->map_api = map_api;
}

  struct kvmap_api *
icache_api_create(struct icache * const cache)
{
  if (cache == NULL) return NULL;
  const struct kvmap_api api = {
    .map = cache,
    .get = (typeof(api.get))icache_get,
    .probe = (typeof(api.probe))icache_probe,
    .set = (typeof(api.set))icache_set,
    .del = (typeof(api.del))icache_del,
    .clean = (typeof(api.clean))icache_clean,
    .destroy = (typeof(api.destroy))icache_destroy,
    .fprint = (typeof(api.fprint))icache_fprint,
    .iter_create = (typeof(api.iter_create))icache_iter_create,
    .iter_next = (typeof(api.iter_next))icache_iter_next,
    .iter_destroy = (typeof(api.iter_destroy))icache_iter_destroy,
  };
  struct kvmap_api * const papi = (typeof(papi))malloc(sizeof(*papi));
  *papi = api;
  return papi;
}
// }}} icache

// ucache {{{
  struct kv *
ucache_get(struct icache * const cache, const struct kv * const key, struct kv * const out)
{
  const u64 hash = key->hash;
  const u64 idx = (hash >> KVMAP_BASE_LEVEL) % cache->nr_slots;
  struct __icache_slot * const slot = &(cache->table[idx]);
  __builtin_prefetch(slot, 0, 3);
  const u64 pkey = kvmap_pkey(hash);
  for (u64 i = 0; i < ICACHE_ENTRY_NR; i++) {
    if (slot->e[i].pkey == pkey) {
      struct kv * const curr = __u64_to_ptr(slot->e[i].ptr);
      if (kv_keymatch(curr, key)) {
        icache_amap_inc(slot, i);
        struct kv * const ret = kv_dup2(curr, out);
        // cache hit
        return ret;
      }
    }
  }
  struct kv * const ret2 = cache->map_api->get(cache->map_api->map, key, out);
  return ret2;
}

  bool
ucache_probe(struct icache * const cache, const struct kv * const key)
{
  const u64 hash = key->hash;
  const u64 idx = (hash >> KVMAP_BASE_LEVEL) % cache->nr_slots;
  struct __icache_slot * const slot = &(cache->table[idx]);
  __builtin_prefetch(slot, 0, 3);
  const u64 pkey = kvmap_pkey(hash);
  for (u64 i = 0; i < ICACHE_ENTRY_NR; i++) {
    if (slot->e[i].pkey == pkey) {
      struct kv * const curr = __u64_to_ptr(slot->e[i].ptr);
      if (kv_keymatch(curr, key)) {
        icache_amap_inc(slot, i);
        // cache hit
        return true;
      }
    }
  }
  const bool r = cache->map_api->probe(cache->map_api->map, key);
  // index hit/miss
  return r;
}

  bool
ucache_set(struct icache * const cache, const struct kv * const kv)
{
  const bool ret = cache->map_api->set(cache->map_api->map, kv);
  return ret;
}

  bool
ucache_del(struct icache * const cache, const struct kv * const key)
{
  const bool ret = cache->map_api->del(cache->map_api->map, key);
  return ret;
}

  struct kvmap_api *
ucache_api_create(struct icache * const cache)
{
  if (cache == NULL) return NULL;
  const struct kvmap_api api = {
    .map = cache,
    .get = (typeof(api.get))ucache_get,
    .probe = (typeof(api.probe))ucache_probe,
    .set = (typeof(api.set))ucache_set,
    .del = (typeof(api.del))ucache_del,
    .clean = (typeof(api.clean))icache_clean,
    .destroy = (typeof(api.destroy))icache_destroy,
    .fprint = (typeof(api.fprint))icache_fprint,
    .iter_create = (typeof(api.iter_create))icache_iter_create,
    .iter_next = (typeof(api.iter_next))icache_iter_next,
    .iter_destroy = (typeof(api.iter_destroy))icache_iter_destroy,
  };
  struct kvmap_api * const papi = (typeof(papi))malloc(sizeof(*papi));
  *papi = api;
  return papi;
}
// }}} ucache

// rcache {{{
struct rcache {
  spinlock hlocks[KVMAP_HLOCK_NR];
  u64 padding10[7];

  u64 nr_slots;
  u64 slots_alloc_size;
  struct __icache_slot * table;

  u64 nr_hists;
  u64 hists_alloc_size;
  struct __icache_hist * hists;

  rcache_match_func match;
  rcache_hash_func hash_key;
  rcache_hash_func hash_kv;
};

  struct rcache *
rcache_create(const u64 nr_mb, rcache_match_func match, rcache_hash_func hash_key, rcache_hash_func hash_kv)
{
  debug_assert(ICACHE_ENTRY_NR == 15 || ICACHE_ENTRY_NR == 7);
  debug_assert((sizeof(struct __icache_slot) % 64) == 0);
  debug_assert((sizeof(struct __icache_hist) % 8) == 0);

  if (match == NULL || hash_key == NULL || hash_kv == NULL) {
    return NULL;
  }

  struct rcache * const cache = (typeof(cache))malloc(sizeof(*cache));
  memset(cache, 0, sizeof(*cache));

  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_init(&(cache->hlocks[i]));
  }

  const u64 table_size = nr_mb << 20;
  const u64 nr_slots = table_size / sizeof(cache->table[0]);
  struct __icache_slot * const table = pages_alloc_best(table_size, false, &(cache->slots_alloc_size));
  if (table == NULL) {
    free(cache);
    return NULL;
  }
  debug_assert(nr_slots % KVMAP_HLOCK_NR == 0);
  cache->table = table;
  cache->nr_slots = nr_slots;

  const u64 nr_hists = nr_slots * 16;
  const u64 hists_size = nr_hists * sizeof(cache->hists[0]);
  struct __icache_hist * const hists = pages_alloc_best(hists_size, false, &(cache->hists_alloc_size));
  if (hists == NULL) {
    pages_unmap(table, cache->slots_alloc_size);
    free(cache);
    return NULL;
  }

  cache->nr_hists = nr_hists;
  cache->hists = hists;

  cache->match = match;
  cache->hash_key = hash_key;
  cache->hash_kv = hash_kv;
  return cache;
}

  static inline void
rcache_lock_hash(struct rcache * const cache, const u64 hash)
{
  spinlock_lock(&(cache->hlocks[kvmap_hlocks_index(hash)]));
}

  static inline void
rcache_unlock_hash(struct rcache * const cache, const u64 hash)
{
  spinlock_unlock(&(cache->hlocks[kvmap_hlocks_index(hash)]));
}

  static inline void
rcache_put_entry(struct __kvmap_entry * const e, const void * const kv, const u64 hash)
{
  e->ptr = (u64)kv;
  e->pkey = kv ? kvmap_pkey(hash) : 0;
}

  static inline bool
rcache_update(struct __icache_slot * const slot, const u64 rep, const void * const kv, const u64 hash)
{
  if (slot->used < ICACHE_ENTRY_NR) {
    for (u64 i = 0; i < ICACHE_ENTRY_NR; i++) {
      if (slot->e[i].ptr == 0) {
        rcache_put_entry(&(slot->e[i]), kv, hash);
        slot->used++;
        return false;
      }
    }
    debug_assert(false);
  }

  const u64 victim = random_u64() % ICACHE_ENTRY_NR;
  icache_amap_dec(slot, victim);

  for (u64 i = 0; i < ICACHE_ENTRY_NR; i++) {
    const u64 vi = (i + victim) % ICACHE_ENTRY_NR;
    const u64 rep1 = icache_amap_get(slot, vi);
    if (rep1 < rep) {
      rcache_put_entry(&(slot->e[vi]), kv, hash);
      icache_amap_set(slot, vi, rep);
      return true;
    }
  }
  return false;
}

  void *
rcache_get_hash(struct rcache * const cache, const void * const key, const u64 hash)
{
  const u64 idx = (hash >> KVMAP_BASE_LEVEL) % cache->nr_slots;
  struct __icache_slot * const slot = &(cache->table[idx]);
  __builtin_prefetch(slot, 0, 3);
  const u64 pkey = kvmap_pkey(hash);
  rcache_lock_hash(cache, hash);
  for (u64 i = 0; i < ICACHE_ENTRY_NR; i++) {
    if (slot->e[i].pkey == pkey) {
      void * const curr = (void *)(u64)(slot->e[i].ptr);
      if (curr && cache->match(curr, key)) {
        icache_amap_inc(slot, i);
        rcache_unlock_hash(cache, hash);
        return curr;
      }
    }
  }
  rcache_unlock_hash(cache, hash);
  return NULL;
}

  void *
rcache_get(struct rcache * const cache, const void * const key)
{
  const u64 hash = cache->hash_key(key);
  return rcache_get_hash(cache, key, hash);
}

  void
rcache_hint_hash(struct rcache * const cache, const void * const kv, const u64 hash)
{
  const u64 idx = (hash >> KVMAP_BASE_LEVEL) % cache->nr_slots;
  struct __icache_slot * const slot = &(cache->table[idx]);
  if ((slot->used < ICACHE_ENTRY_NR) || ((random_u64() & 0xfff) < 0x80)) {
    struct __icache_hist * const hist = &(cache->hists[(hash >> KVMAP_BASE_LEVEL) % cache->nr_hists]);
    rcache_lock_hash(cache, hash);
    const u64 rep = icache_hist_update(hist, kvmap_pkey(hash));
    if (rep) {
      (void)rcache_update(slot, rep, kv, hash);
    }
    rcache_unlock_hash(cache, hash);
  }
}

  void
rcache_hint(struct rcache * const cache, const void * const kv)
{
  const u64 hash = cache->hash_kv(kv);
  rcache_hint_hash(cache, kv, hash);
}

  void
rcache_invalidate_hash(struct rcache * const cache, const u64 hash)
{
  const u64 idx = (hash >> KVMAP_BASE_LEVEL) % cache->nr_slots;
  struct __icache_slot * const slot = &(cache->table[idx]);
  const u64 pkey = kvmap_pkey(hash);
  rcache_lock_hash(cache, hash);
  for (u64 i = 0; i < ICACHE_ENTRY_NR; i++) {
    if ((slot->e[i].pkey == pkey) && slot->e[i].ptr) {
      icache_put_entry(&(slot->e[i]), NULL);
      slot->used--;
      icache_amap_clear(slot, i);
    }
  }
  rcache_unlock_hash(cache, hash);
}

  void
rcache_invalidate_key(struct rcache * const cache, const void * key)
{
  const u64 hash = cache->hash_key(key);
  rcache_invalidate_hash(cache, hash);
}

  void
rcache_invalidate_kv(struct rcache * const cache, const void * kv)
{
  const u64 hash = cache->hash_kv(kv);
  rcache_invalidate_hash(cache, hash);
}

  void
rcache_clean(struct rcache * const cache)
{
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_lock(&(cache->hlocks[i]));
  }
  for (u64 i = 0; i < cache->nr_slots; i++) {
    struct __icache_slot * const slot = &(cache->table[i]);
    memset(slot, 0, sizeof(*slot));
  }
  for (u64 i = 0; i < KVMAP_HLOCK_NR; i++) {
    spinlock_unlock(&(cache->hlocks[i]));
  }
}

  void
rcache_destroy(struct rcache * const cache)
{
  rcache_clean(cache);
  pages_unmap(cache->table, cache->slots_alloc_size);
  pages_unmap(cache->hists, cache->hists_alloc_size);
  free(cache);
}
// }}} rcache

// maptest {{{
#define MAPTEST_GEN_OPT_SYNC   ((0))
#define MAPTEST_GEN_OPT_WAIT   ((1))
#define MAPTEST_GEN_OPT_NOWAIT ((2))

#ifdef MAPTEST_PAPI
#include <papi.h>
__attribute__((constructor))
  static void
maptest_papi_init(void)
{
  PAPI_library_init(PAPI_VER_CURRENT);
  PAPI_thread_init(pthread_self);
}

  static void *
maptest_thread_func(void * const ptr)
{
  struct maptest_worker_info * const wi = (typeof(wi))ptr;
  struct maptest_papi_info * const papi_info = wi->papi_info;
  bool papi = false;
  int es = PAPI_NULL;
  if (papi_info && papi_info->nr && (PAPI_create_eventset(&es) == PAPI_OK)) {
    if (PAPI_OK == PAPI_add_events(es, papi_info->events, papi_info->nr)) {
      PAPI_start(es);
      papi = true;
    } else {
      PAPI_destroy_eventset(&es);
    }
  }
  void * const ret = wi->thread_func(ptr);
  if (papi) {
    u64 v[papi_info->nr];
    if (PAPI_OK == PAPI_stop(es, (long long *)v)) {
      for (u64 i = 0; i < papi_info->nr; i++) {
        vctr_set(wi->papi_vctr, i, v[i]);
      }
    }
    PAPI_destroy_eventset(&es);
  }
  return ret;
}

  static struct maptest_papi_info *
maptest_papi_prepare(void)
{
  char ** const tokens = string_tokens(getenv("MAPTEST_PAPI_EVENTS"), ",");
  if (tokens == NULL) return NULL;
  u64 nr_events = 0;
  int events[1024];
  for (u64 i = 0; tokens[i]; i++) {
    if (PAPI_OK == PAPI_event_name_to_code(tokens[i], &(events[nr_events]))) {
      nr_events++;
    }
  }
  struct maptest_papi_info * const pi = (typeof(pi))malloc(sizeof(*pi) + (sizeof(pi->events[0]) * nr_events));
  pi->nr = nr_events;
  memcpy(pi->events, events, sizeof(pi->events[0]) * nr_events);
  return pi;
}

  static void
maptest_papi_report(struct maptest_papi_info * const papi_info, struct maptest_worker_info ** const infov, const u64 cc, FILE * const out)
{
  if (papi_info == NULL) return;
  struct vctr * const va = vctr_create(papi_info->nr);
  for (u64 i = 0; i < cc; i++) {
    debug_assert(infov[i]->papi_vctr);
    vctr_merge(va, infov[i]->papi_vctr);
  }
  fprintf(out, "PAPI");
  char name[1024];
  for (u64 i = 0; i < papi_info->nr; i++) {
    PAPI_event_code_to_name(papi_info->events[i], name);
    fprintf(out, " %s %"PRIu64, name, vctr_get(va, i));
  }
  vctr_destroy(va);
}

#else // no papi

  static void *
maptest_thread_func(void * const ptr)
{
  struct maptest_worker_info * const wi = (typeof(wi))ptr;
  return wi->thread_func(ptr);
}

  static struct maptest_papi_info *
maptest_papi_prepare(void)
{
  return NULL;
}

  static void
maptest_papi_report(struct maptest_papi_info * const papi_info, struct maptest_worker_info ** const infov, const u64 cc, FILE * const out)
{
  (void)papi_info;
  (void)infov;
  (void)cc;
  (void)out;
  return;
}
#endif

  int
maptest_pass(const int argc, char ** const argv, char ** const pref, struct pass_info * const pi)
{
  if ((argc < 10) || (strcmp(argv[0], "pass") != 0)) return -1;

  const u64 c = strtoull(argv[1], NULL, 10);
  const u64 cc = c ? c : process_affinity_core_count();
  const u64 end_type = strtoull(argv[2], NULL, 10);
  const u64 magic = strtoull(argv[3], NULL, 10);
  const u64 rep = strtoull(argv[4], NULL, 10);
  const u64 vlen = strtoull(argv[5], NULL, 10);
  const u64 pset = strtoull(argv[6], NULL, 10);
  const u64 pdel = strtoull(argv[7], NULL, 10);
  const u64 pget = strtoull(argv[8], NULL, 10);
  const u64 rgen_opt = strtoull(argv[9], NULL, 10);
  if (end_type > 1) return -1;

  const u64 nr_cores = process_affinity_core_count();
  u64 cores[nr_cores];
  process_affinity_core_list(nr_cores, cores);
  struct damp * const damp = damp_create(7, 0.004, 0.05);
  struct maptest_papi_info * const papi_info = maptest_papi_prepare();

  char out[1024] = {};
  // per-worker data
  struct maptest_worker_info info[cc];
  struct maptest_worker_info *infov[cc];
  rgen_next_func next_func = (rgen_opt == MAPTEST_GEN_OPT_NOWAIT) ? rgen_next_nowait : rgen_next_wait;
  for (u64 i = 0; i < cc; i++) {
    memset(&(info[i]), 0, sizeof(info[i]));
    infov[i] = &(info[i]);
    info[i].thread_func = pi->wf;
    info[i].api = pi->api;
    info[i].gen = rgen_dup(pi->gen0);
    info[i].rgen_next = next_func;
    info[i].seed = ((i >> 2) + 73) * 17 * 117;
    info[i].vlen = vlen;
    info[i].pset = pset;
    info[i].pdel = pset + pdel;
    info[i].pget = pset + pdel + pget;
    info[i].end_type = end_type;
    if (end_type == MAPTEST_END_COUNT) {
      info[i].end_magic = magic;
    }
    info[i].vctr = vctr_create(pi->vctr_size);
    if (papi_info) {
      info[i].papi_info = papi_info;
      info[i].papi_vctr = vctr_create(papi_info->nr);
    }

    if (rgen_opt == MAPTEST_GEN_OPT_WAIT) {
      const bool rconv = rgen_async_convert(info[i].gen, cores[i % nr_cores] + 1);
      (void)rconv;
      debug_assert(rconv);
    } else if (rgen_opt == MAPTEST_GEN_OPT_NOWAIT) {
      const bool rconv = rgen_async_convert(info[i].gen, cores[i % nr_cores] + 1);
      (void)rconv;
      debug_assert(rconv);
    }
  }

  bool done = false;
  const double t0 = time_sec();
  // until: rep times, or done determined by damp
  for (u64 r = 0; rep ? (r < rep) : (done == false); r++) {
    // prepare
    const double dt1 = time_diff_sec(t0);
    for (u64 i = 0; i < cc; i++) {
      vctr_reset(info[i].vctr);
      if (info[i].papi_vctr) vctr_reset(info[i].papi_vctr);
      rgen_async_wait_all(info[i].gen);
    }

    // set end-time
    if (end_type == MAPTEST_END_TIME) {
      const u64 end_time = time_nsec() + (UINT64_C(1000000000) * magic);
      for (u64 i = 0; i < cc; i++) {
        info[i].end_magic = end_time;
      }
    }

    const double dt = thread_fork_join_private(cc, maptest_thread_func, (void **)infov);
    struct vctr * const va = vctr_create(pi->vctr_size);
    for (u64 i = 0; i < cc; i++) vctr_merge(va, infov[i]->vctr);
    done = pi->af(va, dt, damp, out);
    vctr_destroy(va);

    maptest_papi_report(papi_info, infov, cc, stderr);

    // stderr messages
    fprintf(stderr, " try %"PRIu64" %.2lf %.2lf ", r, dt1, dt);
    for (int i = 0; pref[i]; i++) fprintf(stderr, "%s ", pref[i]);
    for (int i = 0; i < 10; i++) fprintf(stderr, "%s ", argv[i]);
    fprintf(stderr, "%s", out);
    fflush(stderr);
  }

  // clean up
  damp_destroy(damp);
  if (papi_info) free(papi_info);
  for (u64 i = 0; i < cc; i++) {
    rgen_destroy(info[i].gen);
    vctr_destroy(info[i].vctr);
    if (info[i].papi_vctr) vctr_destroy(info[i].papi_vctr);
  }

  // done messages
  for (int i = 0; pref[i]; i++) fprintf(stdout, "%s ", pref[i]);
  for (int i = 0; i < 10; i++) fprintf(stdout, "%s ", argv[i]);
  fprintf(stdout, "%s", out);
  fflush(stdout);
  return 10;
}

  int
maptest_passes(int argc, char ** argv, char ** const pref0, struct pass_info * const pi)
{
  char * pref[64];
  int np = 0;
  while (pref0[np]) {
    pref[np] = pref0[np];
    np++;
  }
  const int n1 = np;

  const int argc0 = argc;
  do {
    int pi1 = n1;
    struct rgen * gen = NULL;
    if ((argc < 1) || (strcmp(argv[0], "rgen") != 0)) {
      break;
    }
    const int n2 = rgen_helper(argc, argv, &gen);
    if (n2 < 0) {
      return n2;
    }
    for (int i = 0; i < n2; i++) {
      pref[pi1++] = argv[i];
    }
    pref[pi1] = NULL;
    argc -= n2;
    argv += n2;

    while ((argc > 0) && (strcmp(argv[0], "pass") == 0)) {
      debug_perf_switch();
      pi->gen0 = gen;
      const int n3 = maptest_pass(argc, argv, pref, pi);
      if (n3 < 0) {
        return n3;
      }
      argc -= n3;
      argv += n3;
    }

    rgen_destroy(gen);
  } while (argc > 0);
  return argc0 - argc;
}

  void
maptest_passes_message(void)
{
  fprintf(stderr, "%s Usage: {rgen ... {pass ...}}\n", __func__);
  rgen_helper_message();
  fprintf(stderr, "%s Usage: pass <nth> <magic-type> <magic> <repeat> <vlen> <S%%> <D%%> <G%%> <rgen-opt>\n", __func__);
  fprintf(stderr, "%s magic-type: 0:time, 1:count\n", __func__);
  fprintf(stderr, "%s repeat: 0:damp\n", __func__);
  fprintf(stderr, "%s rgen-opt: 0:sync, 1:wait, 2:nowait\n", __func__);
}

  bool
maptest_main(int argc, char ** argv, int(*test_func)(const int, char ** const))
{
  if (argc < 4) {
    return false;
  }
  // perf
  char * const perf = getenv("MAP_TEST_PERF");
  if (perf && (strcmp(perf, "y") == 0)) {
    debug_perf_start();
  }

  if (strcmp(argv[1], "-") != 0) {
    const int fd1 = open(argv[1], O_CREAT | O_WRONLY | O_TRUNC, 00644);
    if (fd1 >= 0) {
      dup2(fd1, 1);
      close(fd1);
    }
  }

  if (strcmp(argv[2], "-") != 0) {
    const int fd2 = open(argv[2], O_CREAT | O_WRONLY | O_TRUNC, 00644);
    if (fd2 >= 0) {
      dup2(fd2, 2);
      close(fd2);
    }
  }
  // record args
  for (int i = 0; i < argc; i++) {
    fprintf(stderr, " %s", argv[i]);
  }
  fprintf(stderr, "\n");
  fflush(stderr);

  argc -= 3;
  argv += 3;

  while (argc > 0) {
    if (strcmp(argv[0], "api") != 0) {
      fprintf(stderr, "%s need `api' keyword to start benchmark\n", __func__);
      return false;
    }
    const int consume = test_func(argc, argv);
    if (consume < 0) {
      return false;
    }
    debug_assert(consume <= argc);
    argc -= consume;
    argv += consume;
  }
  debug_perf_stop();
  return true;
}
// }}} tester

// fdm: marker
