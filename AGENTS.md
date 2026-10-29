# Cavity implementation workspace restrictions

User-authorized bootstrap, 29 October 2026. These restrictions supplement
existing repository guidance and take precedence over older execution prompts
where they conflict. Bootstrap does not authorize starting implementation or
launching subagents; wait for the user's explicit launch instruction.

- The only writable project directories are
  `/Users/jack/xsuite/cavity-modelling` and `/Users/jack/xsuite/xtrack`,
  including their required Git metadata. Both checkouts have ordinary `.git/`
  directories inside these roots; no external Git common directory is needed.
  Other repositories, installed environments, the knowledge vault and global
  configuration are read-only. Do not broaden permissions or edit global settings.
- Keep all authored files and generated results inside these two repositories.
  Use ignored `cavity-modelling/paper/build/` scratch for compiler/JIT caches,
  test temporaries and logs where supported. Bootstrap scratch and a process-local
  environment file are at `paper/build/implementation-session-20261029/` in
  cavity-modelling. Source its `env.sh` in each build/test shell; use pytest
  `--basetemp="$TMPDIR/pytest" -o cache_dir="$XDG_CACHE_HOME/pytest"` when
  applicable. Keep JIT build output context-local in that scratch tree, following
  the native implementation contract. Never install into or modify shared
  environments. Report any additional unavoidable temporary write path explicitly
  before using it; do not silently fall back to system temporary directories.
- After bootstrap, every edit and local commit in either repository must occur
  on `dev/implementation`. Before EACH mutation batch and EACH commit, root must
  check BOTH repositories with `git symbolic-ref --short HEAD` and require that
  exact branch. Stop dependent work if either differs or HEAD is detached.
- Bootstrap verified cavity-modelling on `dev/implementation` at
  `a0986a935982f212633c0431b532b2045f3905bc`. Xtrack's clean
  `feature/cavity-physics` tip and HEAD were both verified as
  `3cd8bfca8d7c611ba413088d3d343bc469e57006`; root created and switched
  Xtrack's `dev/implementation` from that branch, without using local main.
  Do not reset an existing branch or infer these baseline hashes are current HEADs.
- No further branch switches, merges into main, pushes, history rewrites,
  destructive operations or unrelated-repository changes are authorized.
  Preserve existing work. Ordinary local commits are authorized without another
  confirmation; stage explicit whole-file paths and inspect the staged diff.
- Root alone coordinates all Git mutations and owns each repository's index.
  Subagents are currently prohibited. If later explicitly authorized, they must
  not switch branches, perform Git mutations or share/manipulate the index;
  root must give disjoint file ownership and coordinate integration.
- Branch checks are procedural safeguards, not an enforced branch lock. No
  hook or filesystem policy here prevents another process changing HEAD between
  checks. Do not claim branch enforcement beyond the checks actually performed.

Permission verification at bootstrap: the active managed sandbox declares
`/Users/jack/xsuite`, `/private/tmp` and
`/private/var/folders/t6/2x625kh908d1kb8dh1xsbln40000gn/T` writable. This is
BROADER than the authorized project roots. The vault and global configuration
are outside those declared roots. The narrower project scope above is an
instruction, not a session-wide enforced filesystem restriction. No available
session tool can replace the managed policy. Nested `sandbox-exec` failed with
`sandbox_apply: Operation not permitted`; a no-write narrower sandbox test
succeeded through an approved outside-sandbox invocation, which does not change
the active session or constrain other tools. No global settings were changed.
Do not launch the overnight run until the active session policy is narrowed and
reverified. It must grant writes to both repositories AND their `.git/` metadata,
with other project paths read-only. No extra temporary-directory permission has
been shown necessary for compilation/tests during bootstrap; none were run.
