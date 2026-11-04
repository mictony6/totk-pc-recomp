// Save profiles: which folder holds the save data and the account identity of this run (save_profile.h).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#include "save_profile.h"

static void need(int ok, const char *what) {
    if (!ok) rt_fatal("%s", what);
}

static PlatOnce once = PLAT_ONCE_INIT;
static char dir[1100], active[SAVE_PROFILE_NAME_BYTES];

static int lower(int c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }

// ASCII case-insensitive comparison: Windows folder names ignore case, so "Bob" and "bob" are one profile.
static int name_cmp(const char *a, const char *b) {
    for (;; a++, b++) {
        int d = lower((unsigned char)*a) - lower((unsigned char)*b);
        if (d || !*a) return d;
    }
}

// Windows refuses these as file names, with or without an extension ("con.txt").
static int reserved(const char *name) {
    static const char *const names[] = {"con", "prn", "aux", "nul", "com", "lpt"};
    size_t stem = strcspn(name, ".");
    for (int i = 0; i < 6; i++) {
        if (stem < 3) break;
        int match = 1;
        for (int k = 0; k < 3; k++) match &= lower((unsigned char)name[k]) == names[i][k];
        if (!match) continue;
        if (i < 4 && stem == 3) return 1;
        if (i >= 4 && stem == 4 && name[3] >= '0' && name[3] <= '9') return 1;
    }
    return 0;
}

int save_profile_name_ok(const char *name) {
    size_t n = name ? strlen(name) : 0;
    if (!n || n >= SAVE_PROFILE_NAME_BYTES) return 0;
    if (name[0] == ' ' || name[0] == '.' || name[n - 1] == ' ' || name[n - 1] == '.') return 0;
    for (const unsigned char *p = (const unsigned char *)name; *p; p++)
        if (*p < 0x20 || *p == 0x7F || strchr("<>:\"/\\|?*", *p)) return 0;
    return !reserved(name) && name_cmp(name, "default") != 0;
}

static void users_dir(char *out, size_t cap) {
    need(snprintf(out, cap, "%s/users", rt_config()->profile) < (int)cap, "save profile: path too long");
}

static void choose(void *arg) {
    (void)arg;
    const char *root = rt_config()->profile, *want = cfg_value("TOTK_SAVE_PROFILE");
    if (*want && !save_profile_name_ok(want)) {
        rt_log("save profile: \"%s\" cannot name a profile folder; using the default profile", want);
        want = "";
    }
    snprintf(active, sizeof active, "%s", want);
    if (*want) {
        char users[1024];
        users_dir(users, sizeof users);
        need(snprintf(dir, sizeof dir, "%s/%s", users, want) < (int)sizeof dir, "save profile: path too long");
    } else {
        need(snprintf(dir, sizeof dir, "%s", root) < (int)sizeof dir, "save profile: path too long");
    }
    rt_log("save profile: %s (%s)", *active ? active : "default", dir);
}

const char *save_profile_dir(void) {
    plat_once(&once, choose, NULL);
    return dir;
}

const char *save_profile_active(void) {
    plat_once(&once, choose, NULL);
    return active;
}

typedef struct { char (*names)[SAVE_PROFILE_NAME_BYTES]; int n, max; } List;

static int add_entry(const char *name, int kind, uint64_t size, void *ctx) {
    (void)size;
    List *l = ctx;
    if (kind == PLAT_FS_DIR && save_profile_name_ok(name) && l->n < l->max)
        snprintf(l->names[l->n++], SAVE_PROFILE_NAME_BYTES, "%s", name);
    return l->n < l->max;
}

static int by_name(const void *a, const void *b) { return name_cmp(a, b); }

int save_profile_list(char (*names)[SAVE_PROFILE_NAME_BYTES], int max) {
    char users[1024];
    users_dir(users, sizeof users);
    List l = {names, 0, max};
    if (max > 0 && plat_dir_exists(users)) plat_fs_list(users, add_entry, &l);
    qsort(names, (size_t)l.n, SAVE_PROFILE_NAME_BYTES, by_name);
    return l.n;
}

int save_profile_create(const char *name) {
    if (!save_profile_name_ok(name)) return SAVE_PROFILE_BAD_NAME;
    char users[1024], path[1100];
    users_dir(users, sizeof users);
    if (snprintf(path, sizeof path, "%s/%s", users, name) >= (int)sizeof path) return SAVE_PROFILE_FAILED;
    if (plat_fs_kind(path) != PLAT_FS_MISSING) return SAVE_PROFILE_EXISTS;
    plat_file_mkdir_p(path);
    if (!plat_dir_exists(path)) {
        rt_log("save profile: cannot create %s", path);
        return SAVE_PROFILE_FAILED;
    }
    rt_log("save profile: created \"%s\"", name);
    return SAVE_PROFILE_CREATED;
}
