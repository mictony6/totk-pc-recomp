// Save profiles (save_profile.c): separate sets of saves for the people who play on one machine. A profile is a folder
// holding what fs_host.c and the account service keep per player (save data, its journal and backups, the local account
// identity); the shared files of the profile directory (generated RomFS overlays) stay where they are.
//
//   <profile dir>/                 the default profile, the folder every earlier build used: its saves stay in place
//   <profile dir>/users/<name>/    every other profile, created by the menu's Profiles tab or on first use
//
// TOTK_SAVE_PROFILE names the profile ("" or unset: the default). It is read once, at the first storage or account use;
// a change takes effect at the next start, as the game mounts its save data once and keeps it open.
#pragma once

// Bytes of a profile name with its NUL: 32 bytes of UTF-8, the length of an nn::account nickname.
enum { SAVE_PROFILE_NAME_BYTES = 33 };

// The folder of the profile in use, chosen at the first call and fixed for the process. Any thread.
const char *save_profile_dir(void);
// The name of the profile in use ("" for the default), fixed with save_profile_dir.
const char *save_profile_active(void);

// 1 when name can name a profile folder: 1 to 32 bytes, no control characters or <>:"/\|?*, no leading or trailing space
// or dot, not a name Windows reserves (CON, NUL, COM1, ...), and not "default" (the default profile has no folder name).
int save_profile_name_ok(const char *name);

// The profiles under users/ (the default not included), sorted without regard to case, at most max. Returns the count.
int save_profile_list(char (*names)[SAVE_PROFILE_NAME_BYTES], int max);

// Creates the folder of a new profile. SAVE_PROFILE_CREATED, or why it was not.
enum { SAVE_PROFILE_CREATED, SAVE_PROFILE_BAD_NAME, SAVE_PROFILE_EXISTS, SAVE_PROFILE_FAILED };
int save_profile_create(const char *name);
