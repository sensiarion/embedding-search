//! Project-root resolution for indexing.
//!
//! Used by both the MCP server (no user-provided path) and the CLI
//! subcommands (explicit `path` arg). Refuses `$HOME`, `/`, and any
//! ancestor of `$HOME` — closes the "agent spawned the MCP server from
//! `~` and it tried to walk the whole home tree" footgun.

use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// `git rev-parse --show-toplevel`, run from `from`. Returns `None` when
/// `from` is not inside a git work tree or git is unavailable.
pub fn git_toplevel(from: &Path) -> Option<PathBuf> {
    let out = Command::new("git")
        .arg("-C")
        .arg(from)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let path = std::str::from_utf8(&out.stdout).ok()?.trim();
    if path.is_empty() {
        None
    } else {
        Some(PathBuf::from(path))
    }
}

/// Pure resolver — all environment inputs are arguments so unit tests
/// don't fight global state.
///
/// Priority:
/// 1. `requested` — CLI `path` argument when explicitly provided. Skip
///    this slot by passing `None`; clap should only pass `Some` when
///    the user typed a path, not on a defaulted "." (else env_override
///    is silently dropped on every CLI invocation).
/// 2. `env_override` — `EMBEDDING_SEARCH_PROJECT_DIR`.
/// 3. `git_lookup(cwd)` — repo toplevel above CWD.
/// 4. `cwd` — last resort.
///
/// The resolved path is canonicalized (tolerated if the dir doesn't
/// yet exist — old MCP code did the same via `unwrap_or`) and then
/// refused if it equals `$HOME`, `/`, or any ancestor of `$HOME`. When
/// `HOME` is unset, the path must instead live inside a git repo —
/// otherwise a sandboxed launcher with `EMBEDDING_SEARCH_PROJECT_DIR=/Users`
/// (or `/home`, `/var`, …) could still steer the indexer over every
/// user account.
pub fn resolve_project_root_with<F: Fn(&Path) -> Option<PathBuf>>(
    env_override: Option<&Path>,
    requested: Option<&Path>,
    cwd: &Path,
    home: Option<&Path>,
    git_lookup: F,
) -> Result<PathBuf> {
    let candidate: PathBuf = match (requested, env_override) {
        (Some(r), _) => r.to_path_buf(),
        (None, Some(e)) => e.to_path_buf(),
        (None, None) => git_lookup(cwd).unwrap_or_else(|| cwd.to_path_buf()),
    };

    // Tolerate a missing dir (the old MCP `serve_async` did
    // `canonicalize(...).unwrap_or(project_dir)` so launchers that
    // create the project dir just after spawn still worked). The
    // refusal check below runs against whatever resolution succeeded.
    let resolved = std::fs::canonicalize(&candidate).unwrap_or(candidate);

    refuse_unsafe_root(&resolved, home, &git_lookup)?;
    Ok(resolved)
}

fn refuse_unsafe_root<F: Fn(&Path) -> Option<PathBuf>>(
    resolved: &Path,
    home: Option<&Path>,
    git_lookup: &F,
) -> Result<()> {
    if resolved == Path::new("/") {
        return Err(anyhow!(
            "refusing to index `/` — run inside a project directory or set \
             EMBEDDING_SEARCH_PROJECT_DIR"
        ));
    }
    match home {
        Some(home) => {
            let home_canon = std::fs::canonicalize(home).unwrap_or_else(|_| home.to_path_buf());
            if resolved == home_canon {
                return Err(anyhow!(
                    "refusing to index $HOME ({}) — run inside a project \
                     directory or set EMBEDDING_SEARCH_PROJECT_DIR to an \
                     explicit subdir",
                    home_canon.display()
                ));
            }
            if home_canon.starts_with(resolved) {
                return Err(anyhow!(
                    "refusing to index {} — it is an ancestor of $HOME ({})",
                    resolved.display(),
                    home_canon.display()
                ));
            }
        }
        None => {
            // No HOME → no anchor for the $HOME/ancestor checks. A
            // sandboxed launcher without HOME could otherwise be
            // steered at /Users, /home, /root, etc. Require evidence
            // that the target is a real project (a git repo).
            if git_lookup(resolved).is_none() {
                return Err(anyhow!(
                    "refusing to index {} — HOME is not set and the path is \
                     not inside a git repository; set HOME or point \
                     EMBEDDING_SEARCH_PROJECT_DIR at a git checkout",
                    resolved.display()
                ));
            }
        }
    }
    Ok(())
}

/// Production wrapper. CLI subcommands pass `Some(path)`; the MCP server
/// passes `None`.
pub fn resolve_project_root(requested: Option<&Path>) -> Result<PathBuf> {
    let env = std::env::var_os("EMBEDDING_SEARCH_PROJECT_DIR").map(PathBuf::from);
    let cwd = std::env::current_dir().context("resolve current dir")?;
    let home = std::env::var_os("HOME").map(PathBuf::from);
    resolve_project_root_with(
        env.as_deref(),
        requested,
        &cwd,
        home.as_deref(),
        git_toplevel,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> PathBuf {
        std::env::temp_dir().canonicalize().unwrap()
    }

    #[test]
    fn refuses_home_exactly() {
        let home = tmp().join("es_root_refuse_home");
        std::fs::create_dir_all(&home).unwrap();
        let err =
            resolve_project_root_with(None, Some(&home), Path::new("/tmp"), Some(&home), |_| None)
                .unwrap_err();
        assert!(err.to_string().contains("$HOME"), "got: {err}");
    }

    #[test]
    fn refuses_root_slash() {
        let err =
            resolve_project_root_with(None, Some(Path::new("/")), Path::new("/"), None, |_| None)
                .unwrap_err();
        assert!(err.to_string().contains("`/`"), "got: {err}");
    }

    #[test]
    fn refuses_ancestor_of_home() {
        let parent = tmp().join("es_root_ancestor");
        let home = parent.join("user");
        std::fs::create_dir_all(&home).unwrap();
        let err =
            resolve_project_root_with(None, Some(&parent), Path::new("/tmp"), Some(&home), |_| {
                None
            })
            .unwrap_err();
        assert!(err.to_string().contains("ancestor of $HOME"), "got: {err}");
    }

    #[test]
    fn picks_git_toplevel_when_no_explicit_path() {
        let repo = tmp().join("es_root_repo");
        let sub = repo.join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        let resolved =
            resolve_project_root_with(None, None, &sub, Some(&tmp().join("nobody_home")), |_| {
                Some(repo.clone())
            })
            .unwrap();
        assert_eq!(resolved, repo);
    }

    #[test]
    fn falls_back_to_cwd_when_no_git() {
        let dir = tmp().join("es_root_norepo");
        std::fs::create_dir_all(&dir).unwrap();
        let resolved =
            resolve_project_root_with(None, None, &dir, Some(&tmp().join("nobody_home")), |_| None)
                .unwrap();
        assert_eq!(resolved, dir);
    }

    #[test]
    fn explicit_request_beats_env() {
        let req = tmp().join("es_root_req");
        let env = tmp().join("es_root_env");
        std::fs::create_dir_all(&req).unwrap();
        std::fs::create_dir_all(&env).unwrap();
        let resolved = resolve_project_root_with(
            Some(&env),
            Some(&req),
            Path::new("/tmp"),
            Some(&tmp().join("nobody_home")),
            |_| Some(PathBuf::from("/tmp/ignored")),
        )
        .unwrap();
        assert_eq!(resolved, req);
    }

    #[test]
    fn env_override_used_when_no_request() {
        let env = tmp().join("es_root_env_only");
        std::fs::create_dir_all(&env).unwrap();
        let resolved = resolve_project_root_with(
            Some(&env),
            None,
            Path::new("/tmp"),
            Some(&tmp().join("nobody_home")),
            |_| Some(PathBuf::from("/tmp/ignored")),
        )
        .unwrap();
        assert_eq!(resolved, env);
    }

    #[test]
    fn refuses_when_home_unset_and_path_not_in_git_repo() {
        // Sandboxed launcher / systemd unit / CI runner without HOME and
        // EMBEDDING_SEARCH_PROJECT_DIR=/Users (or /home, /var, …) must
        // not pass — there is no $HOME to anchor the refusal, so we
        // require evidence the target is a real project.
        let dir = tmp().join("es_root_no_home_no_git");
        std::fs::create_dir_all(&dir).unwrap();
        let err = resolve_project_root_with(Some(&dir), None, Path::new("/tmp"), None, |_| None)
            .unwrap_err();
        assert!(err.to_string().contains("HOME is not set"), "got: {err}");
    }

    #[test]
    fn accepts_when_home_unset_but_path_is_in_git_repo() {
        // Same scenario but the path is inside a git work tree — the
        // resolver should accept it (real project, deliberate target).
        let dir = tmp().join("es_root_no_home_git");
        std::fs::create_dir_all(&dir).unwrap();
        let resolved = resolve_project_root_with(Some(&dir), None, Path::new("/tmp"), None, |p| {
            Some(p.to_path_buf())
        })
        .unwrap();
        assert_eq!(resolved, dir);
    }

    #[test]
    fn tolerates_canonicalize_failure_on_missing_dir() {
        // Old MCP serve_async used `canonicalize(...).unwrap_or(project_dir)`
        // so launchers that mkdir the project root just after spawning
        // the server still worked. Preserve that tolerance.
        let missing = tmp().join("es_root_does_not_exist_yet_xyz");
        let _ = std::fs::remove_dir_all(&missing);
        let resolved = resolve_project_root_with(
            Some(&missing),
            None,
            Path::new("/tmp"),
            Some(&tmp().join("nobody_home")),
            |_| None,
        )
        .unwrap();
        assert_eq!(resolved, missing);
    }
}
