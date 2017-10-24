#include "lib1.h"
#include "lmdb.h"

#define XSA ((0))
#define XSS ((1))
#define XDA ((2))
#define XDS ((3))
#define XGA ((4))
#define XGS ((5))
#define XPA ((6))
#define XPS ((7))


  static bool
lmdb_analyse(struct vctr * const va, const double dt, struct damp * const damp, char * const out)
{
  u64 v[8];
  for (u64 i = 0; i < 8; i++) {
    v[i] = vctr_get(va, i);
  }

  const u64 nrop = v[XSA] + v[XDA] + v[XGA] + v[XPA];
  const double mops = ((double)nrop) / (dt * 1.0e6);
  const bool done = damp_add_test(damp, mops);
  const double havg = damp_average(damp);
  const char * const pat = " set %"PRIu64" %"PRIu64" del %"PRIu64" %"PRIu64
      " get %"PRIu64" %"PRIu64" pro %"PRIu64" %"PRIu64" mops %.4lf havg %.4lf\n";
  sprintf(out, pat, v[XSA], v[XSS], v[XDA], v[XDS], v[XGA], v[XGS], v[XPA], v[XPS], mops, havg);
  return done;
}


  static void
lmdb_batch(struct maptest_worker_info * const info, MDB_txn * const txn, MDB_val * const key, void * const data, const u64 nr)
{
  MDB_val value;
  rgen_next_func next = info->rgen_next;
  struct vctr * const v = info->vctr;

  for (u64 i = 0; i < nr; i++) {
    const u64 x = next(info->gen);
    memcpy(key->mv_data, &x, sizeof(x));
    const u64 y = random_u64() % 100lu;
    if (y < info->pset) {
      value.mv_size = info->vlen;
      value.mv_data = data;
      memcpy(value.mv_data, key->mv_data, sizeof(x));
      vctr_add1(v, XSA);
      if (mdb_put(txn, 1, key, &value, 0) == MDB_SUCCESS) vctr_add1(v, XSS);
    } else if (y < info->pdel) {
      vctr_add1(v, XDA);
      if (mdb_del(txn, 1, key, NULL) == MDB_SUCCESS) vctr_add1(v, XDS);
    } else if (y < info->pget) {
      vctr_add1(v, XGA);
      if (mdb_get(txn, 1, key, &value) == MDB_SUCCESS) {
        vctr_add1(v, XGS);
        memcpy(data, value.mv_data, value.mv_size);
      }
    } else { // probe
      vctr_add1(v, XPA);
      if (mdb_get(txn, 1, key, &value) == MDB_SUCCESS) vctr_add1(v, XPS);
    }
  }
}

  static void *
lmdb_worker(void * const ptr)
{
  struct maptest_worker_info * const info = (typeof(info))ptr;
  MDB_val key;
  key.mv_size = sizeof(u64);
  key.mv_data = calloc(1, sizeof(u64));
  void * const data = (typeof(data))calloc(1, 4096);
  MDB_env * env = (typeof(env))info->api;
  MDB_txn * txn = NULL;
  int flags = 0;
  if ((info->pset + info->pdel) == 0) {
    flags |= MDB_RDONLY;
  }
  mdb_txn_begin(env, NULL, flags, &txn);

  srandom_u64(info->seed);
  const u64 end_magic = info->end_magic;
  srandom_u64(time_nsec());
  if (info->end_type == MAPTEST_END_TIME) {
    do {
      lmdb_batch(info, txn, &key, data, 4096);
    } while (time_nsec() < end_magic);
  } else if (info->end_type == MAPTEST_END_COUNT) {
    u64 count = 0;
    do {
      const u64 nr = (end_magic - count) > 4096 ? 4096 : (end_magic - count);
      lmdb_batch(info, txn, &key, data, nr);
      count += nr;
    } while (count < end_magic);
  }
  if (flags) {
    mdb_txn_abort(txn);
  } else {
    mdb_txn_commit(txn);
  }
  free(data);
  free(key.mv_data);
  return NULL;
}

  static void
test_lmdb_message(void)
{
  fprintf(stderr, "%s usage: <stdout> <stderr> api <icache-size-mb> {rgen ... {pass ...}}}\n", __func__);
  maptest_passes_message();
}

  static int
lmdb_helper_open(int argc, char ** argv, MDB_env ** const env_out)
{
  if (argc < 2 || strcmp(argv[0], "api") != 0) {
    return -1;
  }
  MDB_env * env = NULL;
  mdb_env_create(&env);
  mdb_env_set_maxreaders(env, 64);
  mdb_env_set_mapsize(env, 1lu << 36);

  const size_t size = strtoull(argv[1], NULL, 10);
  mdb_env_set_rcache(env, size);

  mkdir("./testdb", 0755);
  mdb_env_open(env, "./testdb", 0, 0664);

  MDB_txn * txn = NULL;
  mdb_txn_begin(env, NULL, 0, &txn);
  MDB_dbi dbi;
  mdb_dbi_open(txn, NULL, MDB_CREATE, &dbi);
  mdb_txn_commit(txn);
  // dbi == MAIN_DBI
  debug_assert(dbi == 1);

  *env_out = env;
  return 2;
}

  static void
lmdb_helper_close(MDB_env * env)
{
  MDB_txn * txn = NULL;
  mdb_txn_begin(env, NULL, 0, &txn);
  mdb_drop(txn, 1, 0);
  mdb_txn_commit(txn);
  mdb_dbi_close(env, 1);
  mdb_env_close(env);
}

  static int
test_lmdb(const int argc, char ** const argv)
{
  MDB_env * env = NULL;
  const int n1 = lmdb_helper_open(argc, argv, &env);
  if (n1 < 0) {
    test_lmdb_message();
    return n1;
  }

  char *pref[64] = {};
  for (int i = 0; i < n1; i++) {
    pref[i] = argv[i];
  }
  pref[n1] = NULL;

  struct pass_info pi = {};
  pi.api = env;
  pi.vctr_size = 8;
  pi.wf = lmdb_worker;
  pi.af = lmdb_analyse;
  const int n2 = maptest_passes(argc - n1, argv + n1, pref, &pi);
  if (n2 < 0) {
    test_lmdb_message();
    return n2;
  }
  lmdb_helper_close(env);
  return n1 + n2;
}

  int
main(int argc, char ** argv)
{
  const bool r = maptest_main(argc, argv, test_lmdb);
  if (r == false) test_lmdb_message();
  return 0;
}
