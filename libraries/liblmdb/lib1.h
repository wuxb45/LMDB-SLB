/*
 * Copyright (c) 2016  Wu, Xingbo <wuxb45@gmail.com>
 *
 * All rights reserved. No warranty, explicit or implicit, provided.
 */
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// includes {{{
// C headers
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <math.h>

// POSIX headers
#include <unistd.h>
#include <pthread.h>
#include <fcntl.h>

// Linux headers
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/resource.h>
// }}} includes

// types {{{
typedef int_least8_t            s8;
typedef int_least16_t           s16;
typedef int_least32_t           s32;
typedef int_least64_t           s64;

typedef uint_least8_t           u8;
typedef uint_least16_t          u16;
typedef uint_least32_t          u32;
typedef uint_least64_t          u64;
// }}} types

// locks {{{
typedef struct __spinlock {
  union {
    pthread_spinlock_t lock;
    u64 padding[8];
  };
} spinlock;

  extern void
spinlock_init(spinlock * const lock);

  extern void
spinlock_lock(spinlock * const lock);

  extern bool
spinlock_trylock(spinlock * const lock);

  extern void
spinlock_unlock(spinlock * const lock);


typedef struct __mutexlock {
  union {
    pthread_mutex_t lock;
    u64 padding[8];
  };
} mutexlock;

  extern void
mutexlock_init(mutexlock * const lock);

  extern void
mutexlock_lock(mutexlock * const lock);

  extern bool
mutexlock_trylock(mutexlock * const lock);

  extern void
mutexlock_unlock(mutexlock * const lock);

typedef struct __rwlock {
  union {
    u64 var;
    u64 padding[8];
  };
} rwlock;

  extern void
rwlock_init(rwlock * const lock);

  extern bool
rwlock_trylock_read(rwlock * const lock);

  extern void
rwlock_lock_read(rwlock * const lock);

  extern void
rwlock_unlock_read(rwlock * const lock);

  extern bool
rwlock_trylock_write(rwlock * const lock);

  extern void
rwlock_lock_write(rwlock * const lock);

  extern void
rwlock_unlock_write(rwlock * const lock);
// }}} locks

// timing {{{
  extern u64
time_nsec(void);

  extern double
time_sec(void);

  extern u64
time_diff_nsec(const u64 last);

  extern double
time_diff_sec(const double last);

  extern u64
timespec_diff(const struct timespec t0, const struct timespec t1);
// }}} timing

// debug {{{
  extern void
debug_backtrace(void);

  extern void
watch_u64_usr1(u64 * const ptr);

  extern void
debug_wait_gdb(void);

#ifndef NDEBUG
  extern void
debug_assert(const bool v);
#else
#define debug_assert(expr) ((void)0)
#endif

  extern void
debug_dump_maps(FILE * const out);

  extern void
debug_perf_start(void);

  extern void
debug_perf_switch(void);

  extern void
debug_perf_stop(void);
// }}} debug

// bits {{{
  extern u32
bits_reverse_u32(const u32 v);

  extern u64
bits_reverse_u64(const u64 v);

  extern u64
bits_rotl_u64(const u64 v, const u64 n);

  extern u64
bits_rotr_u64(const u64 v, const u64 n);

  extern u32
bits_rotl_u32(const u32 v, const u64 n);

  extern u32
bits_rotr_u32(const u32 v, const u64 n);
// }}} bits

// bitmap {{{
struct bitmap;

  extern struct bitmap *
bitmap_create(const u64 bits);

  extern bool
bitmap_test(const struct bitmap * const bm, const u64 idx);

  extern bool
bitmap_test_all1(struct bitmap * const bm);

  extern bool
bitmap_test_all0(struct bitmap * const bm);

  extern void
bitmap_set1(struct bitmap * const bm, const u64 idx);

  extern void
bitmap_set0(struct bitmap * const bm, const u64 idx);

  extern u64
bitmap_count(struct bitmap * const bm);

  extern void
bitmap_set_all1(struct bitmap * const bm);

  extern void
bitmap_set_all0(struct bitmap * const bm);

  extern void
bitmap_static_init(struct bitmap * const bm, const u64 bits);
// }}} bitmap

// bloom filter {{{
struct bloomfilter;

  extern struct bloomfilter *
bf_create(const u64 bpk, const u64 capacity);

  extern void
bf_mark(struct bloomfilter * const bf, u64 hash64);

  extern bool
bf_test(const struct bloomfilter * const bf, u64 hash64);

  extern void
bf_destroy(struct bloomfilter * const bf);
// }}} bloom filter

// process/thread {{{
  extern u64
process_get_rss(void);

  extern u64
process_affinity_core_count(void);

  extern u64
process_affinity_core_list(const u64 max, u64 * const cores);

  extern u64
process_cpu_time_usec(void);

  extern void
thread_set_affinity(const u64 cpu);

  extern double
thread_fork_join_private(const u64 nr, void *(*func) (void *), void * const * const argv);

  extern double
thread_fork_join(const u64 nr, void *(*func) (void *), void * const arg);

  extern int
thread_create_at(const u64 cpu, pthread_t * const thread, void *(*start_routine) (void *), void * const arg);
// }}} process/thread

// mm {{{
  extern void *
xalloc(const u64 align, const u64 size);

/* hugepages */
// force posix allocators: -DVALGRIND_MEMCHECK
  extern void *
pages_alloc_4kb(const size_t nr_4kb);

  extern void *
pages_alloc_2mb(const size_t nr_2mb);

  extern void *
pages_alloc_1gb(const size_t nr_1gb);

  extern void *
pages_alloc_best(const size_t size, const bool try_1gb, u64 * const size_out);

  extern void
pages_unmap(void * const ptr, const size_t size);
// }}} mm

// oalloc {{{
struct oalloc;

  extern struct oalloc *
oalloc_create(void);

  extern void *
oalloc_alloc(const u64 size, struct oalloc * const oa);

  extern void
oalloc_destroy(struct oalloc * const oa);
// }}} oalloc

// gcache {{{
struct gcache;

  extern struct gcache *
gcache_create(const u64 nr_classes, const u64 inc);

  extern void *
gcache_pull(struct gcache * const g, const u64 size);

  extern bool
gcache_push(struct gcache * const g, const u64 size, void * ptr);

  extern void
gcache_clean(struct gcache * const g);

  extern void
gcache_destroy(struct gcache * const g);

struct gcache_iter;

  extern struct gcache_iter *
gcache_iter_create(struct gcache * const g);

  extern void *
gcache_iter_next(struct gcache_iter * const gi);

  extern void
gcache_iter_destroy(struct gcache_iter * const gi);
// }}} gcache

// cpucache {{{
  extern void
cpu_clflush1(void * const ptr);

  extern void
cpu_clflush(void * const ptr, const size_t size);

  extern void
cpu_mfence(void);
// }}} cpucache

// qsort {{{
  extern void
qsort_u16(u16 * const array, const size_t nr);

  extern void
qsort_u32(u32 * const array, const size_t nr);

  extern void
qsort_u64(u64 * const array, const size_t nr);

  extern void
qsort_double(double * const array, const size_t nr);

  extern void
qsort_u64_sample(const u64 * const array0, const u64 nr, const u64 res, FILE * const out);

  extern void
qsort_double_sample(const double * const array0, const u64 nr, const u64 res, FILE * const out);
// }}} qsort

// hash {{{
  extern u32
crc32(const void * const ptr, const size_t size);

  extern u32
xxhash32(const void * const ptr, const size_t size);

  extern u64
xxhash64(const void * const ptr, const size_t size);
// }}} hash

// xlog {{{
struct xlog {
  u64 nr_rec;
  u64 nr_cap;
  u64 unit_size;
  u8 * ptr;
};

  extern struct xlog *
xlog_create(const u64 nr_init, const u64 unit_size);

  extern void
xlog_append(struct xlog * const xlog, const void * const rec);

  extern void
xlog_append_cycle(struct xlog * const xlog, const void * const rec);

  extern void
xlog_reset(struct xlog * const xlog);

  extern void
xlog_dump(struct xlog * const xlog, FILE * const out);

  extern void
xlog_destroy(struct xlog * const xlog);

struct xlog_iter;

  extern struct xlog_iter *
xlog_iter_create(const struct xlog * const xlog);

  extern bool
xlog_iter_next(struct xlog_iter * const iter, void * const out);
// free iter after use
// }}} ulog/dlog

// string {{{
// size of out should be >= 10
  extern void
str10_u32(void * const out, const u32 v);

// size of out should be >= 20
  extern void
str10_u64(void * const out, const u64 v);

// size of out should be >= 8
  extern void
str16_u32(void * const out, const u32 v);

// size of out should be >= 16
  extern void
str16_u64(void * const out, const u64 v);

// user should free returned ptr after use
  extern char **
string_tokens(const char * const str, const char * const delim);
// }}} string

// damp {{{
struct damp;

  extern struct damp *
damp_create(const u64 cap, const double dshort, const double dlong);

  extern double
damp_average(const struct damp * const d);

  extern double
damp_min(const struct damp * const d);

  extern double
damp_max(const struct damp * const d);

  extern bool
damp_add_test(struct damp * const d, const double v);

  extern void
damp_clean(struct damp * const d);

  extern void
damp_destroy(struct damp * const d);
// }}} damp

// vctr {{{
struct vctr;

  extern struct vctr *
vctr_create(const u64 nr);

  extern u64
vctr_size(struct vctr * const v);

  extern void
vctr_add(struct vctr * const v, const u64 i, const u64 n);

  extern void
vctr_add1(struct vctr * const v, const u64 i);

  extern void
vctr_add_atomic(struct vctr * const v, const u64 i, const u64 n);

  extern void
vctr_add1_atomic(struct vctr * const v, const u64 i);

  extern void
vctr_set(struct vctr * const v, const u64 i, const u64 n);

  extern u64
vctr_get(struct vctr * const v, const u64 i);

  extern void
vctr_merge(struct vctr * const to, const struct vctr * const from);

  extern void
vctr_reset(struct vctr * const v);

  extern void
vctr_destroy(struct vctr * const v);
// }}} vctr

// rgen {{{
  extern u64
xorshift(const u64 x);

  extern u64
random_u64(void);

  extern u64
srandom_u64(const u64 seed);

  extern double
random_double(void);

struct rgen;

typedef u64 (*rgen_next_func)(struct rgen * const);

extern struct rgen * rgen_new_exponential(const double percentile, const double range);
extern struct rgen * rgen_new_constant(const u64 constant);
extern struct rgen * rgen_new_counter(const u64 min, const u64 max);
extern struct rgen * rgen_new_counter_unsafe(const u64 min, const u64 max);
extern struct rgen * rgen_new_skipinc(const u64 min, const u64 max, const u64 inc);
extern struct rgen * rgen_new_skipinc_unsafe(const u64 min, const u64 max, const u64 inc);
extern struct rgen * rgen_new_reducer(const u64 min, const u64 max);
extern struct rgen * rgen_new_reducer_unsafe(const u64 min, const u64 max);
extern struct rgen * rgen_new_zipfian(const u64 min, const u64 max);
extern struct rgen * rgen_new_xzipfian(const u64 min, const u64 max);
extern struct rgen * rgen_new_unizipf(const u64 min, const u64 max, const u64 ufactor);
extern struct rgen * rgen_new_uniform(const u64 min, const u64 max);
extern struct rgen * rgen_new_trace32(const char * const filename);

  extern u64
rgen_next_wait(struct rgen * const gen);

  extern u64
rgen_next_nowait(struct rgen * const gen);

  extern void
rgen_destroy(struct rgen * const gen);

  extern void
rgen_helper_message(void);

  extern int
rgen_helper(const int argc, char ** const argv, struct rgen ** const gen_out);

  extern struct rgen *
rgen_dup(struct rgen * const gen0);

  extern bool
rgen_async_convert(struct rgen * const gen0, const u64 cpu);

  extern void
rgen_async_wait(struct rgen * const gen);

  extern void
rgen_async_wait_all(struct rgen * const gen);
// }}} rgen

// rcu {{{
struct rcu_node;

  extern struct rcu_node *
rcu_node_create(void);

  extern void
rcu_node_init(struct rcu_node * const node);

  extern struct rcu_node *
rcu_node_create(void);

  extern void *
rcu_read_ref(struct rcu_node * const node);

  extern void
rcu_read_unref(struct rcu_node * const node, void * ptr);

  extern void
rcu_update(struct rcu_node * const node, void * ptr);
// }}} rcu

// server {{{
struct stream2 {
  FILE * w;
  FILE * r;
};

struct server;

struct server_wi;

  extern struct server *
server_create(const char * const host, const int port, void*(*worker)(void * const), void * const priv);

  extern void
server_wait(struct server * const server);

  extern void
server_destroy(struct server * const server);

  extern struct stream2 *
server_wi_stream2(struct server_wi * const wi);

  extern void *
server_wi_private(struct server_wi * const wi);

  extern void
server_wi_destroy(struct server_wi * const wi);

  extern struct stream2 *
stream2_create(const char * const host, const int port);

  extern void
stream2_destroy(struct stream2 * const stream2);

// }}} server

// kv {{{
struct kv {
  u64 hash; // hashvalue of the key
  u32 klen;
  u32 vlen;
  u8 kv[];  // len(kv) == klen + vlen
};

  extern size_t
kv_size(const struct kv * const kv);

  extern size_t
kv_size_align(const struct kv * const kv, const u64 align);

  extern size_t
key_size(const struct kv * const key);

  extern size_t
key_size_align(const struct kv * const key, const u64 align);

  extern void
kv_update_hash(struct kv * const kv);

  extern void
kv_refill(struct kv * const kv, const void * const key, const u32 klen, const void * const value, const u32 vlen);

  extern void
kv_refill_str(struct kv * const kv, const char * const key, const char * const value);

  extern struct kv *
kv_create(const void * const key, const u32 klen, const void * const value, const u32 vlen);

  extern struct kv *
kv_create_str(const char * const key, const char * const value);

  extern struct kv *
kv_dup(const struct kv * const kv);

  extern struct kv *
kv_dup_key(const struct kv * const kv);

  extern struct kv *
kv_dup2(const struct kv * const from, struct kv * const to);

  extern struct kv *
kv_dup2_key(const struct kv * const from, struct kv * const to);

  extern bool
kv_keymatch(const struct kv * const key1, const struct kv * const key2);

  extern bool
kv_fullmatch(const struct kv * const kv1, const struct kv * const kv2);

typedef int  (*kv_compare_func)(const struct kv * const kv1, const struct kv * const kv2);

  extern int
kv_keycompare(const struct kv * const kv1, const struct kv * const kv2);

  extern void
kv_qsort(const struct kv ** const kvs, const size_t nr);

  extern void *
kv_value_ptr(struct kv * const kv);

  extern void *
kv_key_ptr(struct kv * const kv);

  extern const void *
kv_value_ptr_const(const struct kv * const kv);

  extern const void *
kv_key_ptr_const(const struct kv * const kv);

  extern struct kv *
kv_alloc_malloc(const u64 size, void * const priv);

  extern void
kv_retire_free(struct kv * const kv, void * const priv);
// }}} kv

// kvmap {{{
struct kvmap_api {
  // no create function in api
  void * map;
  // thread-safe basic functions:
  struct kv * (* get)     (void * map, const struct kv * const key, struct kv * const out);
  bool        (* probe)   (void * map, const struct kv * const key);
  bool        (* set)     (void * map, const struct kv * const kv);
  bool        (* del)     (void * map, const struct kv * const key);
  // unsafe functions:
  void        (* clean)   (void * map);
  void        (* destroy) (void * map);
  void        (* fprint)  (void * map, FILE * const out);
  void *      (* iter_create) (void * map);
  struct kv * (* iter_next) (void * iter, struct kv * const out);
  void        (* iter_destroy) (void * iter);
};

  extern void
kvmap_api_destroy(struct kvmap_api * const api);


typedef struct kv * (* kv_alloc_func)(const u64, void * const);

typedef void (* kv_retire_func)(struct kv * const, void * const);

typedef void (* kvmap_hit_func)(const struct kv * const, void * const);

typedef void (* kvmap_uncache_func)(struct kv * const, void * const);

typedef void (* cache_refresh_func)(struct kv * const, const struct kv * const, void * const);

struct kvmap_mm {
  kv_alloc_func af;
  void * ap;
  kv_retire_func rf;
  void * rp;
  kvmap_hit_func hf;
  void * hp;
  kvmap_uncache_func uf;
  void * up;
  cache_refresh_func xf;
  void * xp;
};

  extern void
kvmap_api_helper_message(void);

  extern int
kvmap_api_helper(int argc, char ** const argv, struct kvmap_api ** const out, struct kvmap_mm * const mm, const bool use_ucache);
// }}} kvmap

// kvmap2 {{{
struct kvmap2;

  extern struct kvmap2 *
kvmap2_create(const struct kvmap_mm * const mm);

  extern struct kv *
kvmap2_get(struct kvmap2 * const map2, const struct kv * const key, struct kv * const out);

  extern bool
kvmap2_probe(struct kvmap2 * const map2, const struct kv * const key);

  extern bool
kvmap2_set(struct kvmap2 * const map2, const struct kv * const kv);

  extern bool
kvmap2_del(struct kvmap2 * const map2, const struct kv * const key);

  extern void
kvmap2_clean(struct kvmap2 * const map2);

  extern void
kvmap2_destroy(struct kvmap2 * const map2);

  extern void
kvmap2_fprint(struct kvmap2 * const map2, FILE * const out);

struct kvmap2_iter;

  extern struct kvmap2_iter *
kvmap2_iter_create(struct kvmap2 * const map2);

  extern struct kv *
kvmap2_iter_next(struct kvmap2_iter * const iter2, struct kv * const out);

  extern void
kvmap2_iter_destroy(struct kvmap2_iter * const iter2);

  extern struct kvmap_api *
kvmap2_api_create(struct kvmap2 * const map);
// }}} kvmap2

// cuckoo {{{
struct cuckoo;

  extern struct cuckoo *
cuckoo_create(const struct kvmap_mm * const mm);

  extern struct kv *
cuckoo_get(struct cuckoo * const map, const struct kv * const key, struct kv * const out);

  extern bool
cuckoo_probe(struct cuckoo * const map, const struct kv * const key);

  extern bool
cuckoo_set(struct cuckoo * const map, const struct kv * const kv);

  extern bool
cuckoo_del(struct cuckoo * const map, const struct kv * const key);

  extern void
cuckoo_clean(struct cuckoo * const map);

  extern void
cuckoo_destroy(struct cuckoo * const map);

  extern void
cuckoo_fprint(struct cuckoo * const map, FILE * const out);

  extern struct cuckoo_iter *
cuckoo_iter_create(struct cuckoo * const map);

  extern struct kv *
cuckoo_iter_next(struct cuckoo_iter * const iter, struct kv * const out);

  extern void
cuckoo_iter_destroy(struct cuckoo_iter * const iter);

  extern struct kvmap_api *
cuckoo_api_create(struct cuckoo * const map);
// }}} cuckoo

// skiplist {{{
struct skiplist;

  extern struct skiplist *
skiplist_create_f(const struct kvmap_mm * const mm, kv_compare_func comp);

  extern struct skiplist *
skiplist_create(const struct kvmap_mm * const mm);

  extern struct kv *
skiplist_get(struct skiplist * const list, const struct kv * const key, struct kv * const out);

  extern bool
skiplist_probe(struct skiplist * const list, const struct kv * const key);

  extern bool
skiplist_set(struct skiplist * const list, const struct kv * const kv);

  extern bool
skiplist_del(struct skiplist * const list, const struct kv * const key);

  extern void
skiplist_clean(struct skiplist * const list);

  extern void
skiplist_destroy(struct skiplist * const list);

  extern void
skiplist_fprint(struct skiplist * const list, FILE * const out);

  extern struct kv *
skiplist_head(struct skiplist * const list, struct kv * const out);

  extern struct kv *
skiplist_tail(struct skiplist * const list, struct kv * const out);

struct skiplist_iter;

  extern struct skiplist_iter *
skiplist_iter_create(struct skiplist * const list);

  extern struct kv *
skiplist_iter_next(struct skiplist_iter * const iter, struct kv * const out);

  extern void
skiplist_iter_destroy(struct skiplist_iter * const iter);

  extern struct kvmap_api *
skiplist_api_create(struct skiplist * const map);
// }}} skiplist

// chainmap {{{
struct chainmap;

  extern struct chainmap *
chainmap_create(const struct kvmap_mm * const mm);

  extern struct kv *
chainmap_get(struct chainmap * const map, const struct kv * const key, struct kv * const out);

  extern bool
chainmap_probe(struct chainmap * const map, const struct kv * const key);

  extern bool
chainmap_set(struct chainmap * const map, const struct kv * const kv);

  extern bool
chainmap_del(struct chainmap * const map, const struct kv * const key);

  extern void
chainmap_clean(struct chainmap * const map);

  extern void
chainmap_destroy(struct chainmap * const map);

  extern void
chainmap_fprint(struct chainmap * const map, FILE * const out);

  extern struct chainmap_iter *
chainmap_iter_create(struct chainmap * const map);

  extern struct kv *
chainmap_iter_next(struct chainmap_iter * const iter, struct kv * const out);

  extern void
chainmap_iter_destroy(struct chainmap_iter * const iter);

  extern struct kvmap_api *
chainmap_api_create(struct chainmap * const map);
// }}} chainmap

// bptree {{{
struct bptree;

  extern struct bptree *
bptree_create(const struct kvmap_mm * const mm);

  extern struct kv *
bptree_get(struct bptree * const tree, const struct kv * const key, struct kv * const out);

  extern bool
bptree_probe(struct bptree * const tree, const struct kv * const key);

  extern bool
bptree_set(struct bptree * const tree, const struct kv * const kv0);

  extern bool
bptree_del(struct bptree * const tree, const struct kv * const key);

  extern void
bptree_clean(struct bptree * const tree);

  extern void
bptree_destroy(struct bptree * const tree);

  extern void
bptree_fprint(struct bptree * const tree, FILE * const out);

struct bptree_iter;

  extern struct bptree_iter *
bptree_iter_create(struct bptree * const tree);

  extern struct kv *
bptree_iter_next(struct bptree_iter * const iter, struct kv * const out);

  extern void
bptree_iter_destroy(struct bptree_iter * const iter);

  extern struct kvmap_api *
bptree_api_create(struct bptree * const map);
// }}} bptree

// mica {{{
struct mica;

  extern struct mica *
mica_create(const struct kvmap_mm * const mm, const u64 power);

  extern struct kv *
mica_get(struct mica * const map, const struct kv * const key, struct kv * const out);

  extern bool
mica_probe(struct mica * const map, const struct kv * const key);

  extern bool
mica_set(struct mica * const map, const struct kv * const kv0);

  extern bool
mica_del(struct mica * const map, const struct kv * const key);

  extern void
mica_clean(struct mica * const map);

  extern void
mica_destroy(struct mica * const map);

  extern struct kvmap_api *
mica_api_create(struct mica * const map);
// }}} mica

// tcpmap {{{
struct tcpmap;

  extern struct tcpmap *
tcpmap_create(const char * const host, const int port);

  extern struct kv *
tcpmap_get(struct tcpmap * const map, const struct kv * const key, struct kv * const out);

  extern bool
tcpmap_probe(struct tcpmap * const map, const struct kv * const key);

  extern bool
tcpmap_set(struct tcpmap * const map, const struct kv * const kv0);

  extern bool
tcpmap_del(struct tcpmap * const map, const struct kv * const key);

  extern void
tcpmap_clean(struct tcpmap * const map);

  extern void
tcpmap_destroy(struct tcpmap * const map);

  extern struct kvmap_api *
tcpmap_api_create(struct tcpmap * const map);
// }}} tcpmap

// icache {{{
struct icache;

  extern struct icache *
icache_create(const struct kvmap_mm * const mm, const u64 nr_mb);

  extern struct kv *
icache_get(struct icache * const cache, const struct kv * const key, struct kv * const out);

  extern bool
icache_probe(struct icache * const cache, const struct kv * const key);

  extern bool
icache_set(struct icache * const cache, const struct kv * const kv);

  extern bool
icache_del(struct icache * const cache, const struct kv * const key);

  extern void
icache_clean(struct icache * const cache);

  extern void
icache_destroy(struct icache * const cache);

  extern void
icache_retire(struct icache * const icache, struct kv * const kv);

  extern  void
icache_uncache(struct icache * const cache, struct kv * const kv);

  extern void
icache_hint(struct icache * const icache, const struct kv * const kv);

  extern void
icache_wrap_mm(struct icache * const cache, struct kvmap_mm * const mm);

  extern void
icache_wrap_kvmap(struct icache * const cache, struct kvmap_api * const map_api);

  extern struct kvmap_api *
icache_api_create(struct icache * const cache);
// }}} icache

// ucache {{{
  extern struct kv *
ucache_get(struct icache * const cache, const struct kv * const key, struct kv * const out);

  extern bool
ucache_probe(struct icache * const cache, const struct kv * const key);

  extern bool
ucache_set(struct icache * const cache, const struct kv * const kv);

  extern bool
ucache_del(struct icache * const cache, const struct kv * const key);

  extern struct kvmap_api *
ucache_api_create(struct icache * const cache);
// }}}

// rcache {{{
typedef bool (* rcache_match_func)(const void * const kv, const void * const key);

typedef u64 (* rcache_hash_func)(const void * const);

struct rcache;

  extern struct rcache *
rcache_create(const u64 nr_mb, rcache_match_func match, rcache_hash_func hash_key, rcache_hash_func hash_kv);

  extern void *
rcache_get_hash(struct rcache * const cache, const void * const key, const u64 hash);

  extern void *
rcache_get(struct rcache * const cache, const void * const key);

  extern void
rcache_hint_hash(struct rcache * const cache, const void * const kv, const u64 hash);

  extern void
rcache_hint(struct rcache * const cache, const void * const kv);

  extern void
rcache_invalidate_hash(struct rcache * const cache, const u64 hash);

  extern void
rcache_invalidate_key(struct rcache * const cache, const void * const key);

  extern void
rcache_invalidate_kv(struct rcache * const cache, const void * const kv);

  extern void
rcache_clean(struct rcache * const cache);

  extern void
rcache_destroy(struct rcache * const cache);
// }}} rcache

// maptest {{{
#define MAPTEST_END_TIME ((0))
#define MAPTEST_END_COUNT ((1))
typedef  bool (*maptest_perf_analyze_func)(struct vctr * const, const double, struct damp * const, char * const);

typedef  void * (*maptest_worker_func)(void *);

struct pass_info {
  struct rgen * gen0;
  void * api;
  u64 vctr_size;
  maptest_worker_func wf;
  maptest_perf_analyze_func af;
};

struct maptest_papi_info {
  u64 nr;
  int events[];
};

struct maptest_worker_info {
  void * (*thread_func)(void *);
  void * api;
  struct rgen * gen;
  rgen_next_func rgen_next;
  u64 seed;
  u64 vlen;
  u64 pset;
  u64 pdel;
  u64 pget;
  u64 end_type;
  u64 end_magic;
  void * priv;
  struct vctr * vctr;
  // PAPI
  struct maptest_papi_info * papi_info;
  struct vctr * papi_vctr;
};

  extern int
maptest_pass(const int argc, char ** const argv, char ** const pref, struct pass_info * const pi);

  extern int
maptest_passes(int argc, char ** argv, char ** const pref0, struct pass_info * const pi);

  extern void
maptest_passes_message(void);

  extern bool
maptest_main(int argc, char ** argv, int(*test_func)(const int, char ** const));
// }}} maptest

#ifdef __cplusplus
}
#endif
// vim:fdm=marker
