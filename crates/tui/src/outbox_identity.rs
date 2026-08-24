//! Stable cross-process identity for lifecycle-outbox records.
//!
//! Boot reconciliation pairs a killed process's unpaired `turn_start` by
//! scanning for the *current* session's `thread_id` — which only works when
//! the next launch emits under the same id the killed process used. A fresh
//! random id per launch (the hook executor's per-launch `sess_*`) can never
//! match, so the outbox gets its own identity: one id per surface (`tui`,
//! `exec`), minted once, persisted under the codewhale home, and reused on
//! every launch. The next boot therefore finds the prior process's records
//! and repairs them.
//!
//! **Live-instance guard.** The id file doubles as the claim: the session
//! holds an exclusive, non-blocking `flock`/`LockFileEx` on it for the
//! process lifetime. A second live instance of the same surface fails the
//! non-blocking acquire and falls back to an ephemeral per-launch id, so two
//! live sessions never share a pairing identity (a shared id would let one
//! session's signal flush fabricate `turn_end`s for the other's genuinely
//! open turns). When the holder dies — SIGKILL included — the kernel drops
//! the flock and the next boot takes the stable id and reconciles.
//!
//! The registry below holds every claim for the process lifetime: a codewhale
//! process serves exactly one session, so the claim must outlive any single
//! call site. `acquire` is idempotent within a process (a registry hit
//! returns the same id without re-flocking, which would otherwise lose to
//! this process's own claim).

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// One held identity claim: the id records under it carry, plus the flocked
/// identity file (kept open for the process lifetime; `None` for an
/// ephemeral fallback id).
struct Claim {
    id: String,
    _file: Option<std::fs::File>,
}

/// Process-lifetime registry of surface claims, keyed by identity-file path.
static CLAIMS: OnceLock<Mutex<Vec<(PathBuf, Claim)>>> = OnceLock::new();

/// Resolve this session's outbox identity for `surface` (`"tui"` / `"exec"`).
///
/// Returns `None` when the codewhale home cannot be resolved (the caller
/// falls back to its per-launch id, matching the pre-identity behavior).
/// Idempotent per process: repeated calls return the same id.
pub(crate) fn acquire(surface: &str) -> Option<String> {
    let home = codewhale_config::codewhale_home().ok()?;
    let dir = home.join("outbox");
    let path = dir.join(format!("{surface}.identity"));
    let registry = CLAIMS.get_or_init(|| Mutex::new(Vec::new()));
    let claims = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((_, claim)) = claims
        .iter()
        .find(|(claimed_path, _)| claimed_path == &path)
    {
        return Some(claim.id.clone());
    }
    drop(claims);

    let (id, held_file) = acquire_from_path(surface, &dir, &path);
    registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .push((
            path,
            Claim {
                id: id.clone(),
                _file: held_file,
            },
        ));
    Some(id)
}

/// Mint, persist, and claim the identity under the flock, or fall back to an
/// ephemeral id when another live process holds the claim.
///
/// The mint-and-read critical section runs only under the flock: whoever
/// holds the claim decides the file's content, so two racing first launches
/// cannot mint two different stable ids.
fn acquire_from_path(surface: &str, dir: &Path, path: &Path) -> (String, Option<std::fs::File>) {
    if let Err(error) = std::fs::create_dir_all(dir) {
        tracing::warn!(
            target: "lifecycle_outbox",
            %error,
            "could not create the outbox identity directory; falling back to an ephemeral id"
        );
        return (ephemeral_id(surface), None);
    }
    let file = match open_identity_file(path) {
        Ok(file) => file,
        Err(error) => {
            tracing::warn!(
                target: "lifecycle_outbox",
                %error,
                "could not open the outbox identity file; falling back to an ephemeral id"
            );
            return (ephemeral_id(surface), None);
        }
    };
    match try_lock_exclusive(&file) {
        Ok(()) => {
            let id = match read_persisted_id(path) {
                Some(id) => id,
                None => {
                    let id = stable_id(surface);
                    persist_id(&file, path, &id);
                    id
                }
            };
            (id, Some(file))
        }
        Err(_) => {
            // Another live process of this surface holds the claim. Do not
            // share its id: mint an ephemeral one so the live instance keeps
            // sole ownership of pairing under the stable id.
            tracing::debug!(
                target: "lifecycle_outbox",
                "another live instance holds the {surface} outbox identity; using an ephemeral id"
            );
            (ephemeral_id(surface), None)
        }
    }
}

/// `surface_<uuid8>` — the persisted, cross-launch id shape.
fn stable_id(surface: &str) -> String {
    format!("{surface}_{}", &uuid::Uuid::new_v4().to_string()[..8])
}

/// `surface_ephemeral_<uuid8>` — never persisted, never claimed.
fn ephemeral_id(surface: &str) -> String {
    format!(
        "{surface}_ephemeral_{}",
        &uuid::Uuid::new_v4().to_string()[..8]
    )
}

fn open_identity_file(path: &Path) -> std::io::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true).write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

fn read_persisted_id(path: &Path) -> Option<String> {
    let mut text = String::new();
    // A fresh read handle: the flock is on the claim's open file
    // description, so this reads the same inode safely.
    let mut reader = std::fs::File::open(path).ok()?;
    use std::io::Read;
    reader.read_to_string(&mut text).ok()?;
    let id = text.trim();
    if id.is_empty() {
        None
    } else {
        Some(id.to_string())
    }
}

fn persist_id(file: &std::fs::File, path: &Path, id: &str) {
    use std::io::{Seek, SeekFrom, Write};
    let mut handle = file.try_clone().ok();
    let Some(ref mut handle) = handle else {
        tracing::warn!(
            target: "lifecycle_outbox",
            "could not clone the outbox identity file handle; the id will not persist"
        );
        return;
    };
    if handle.set_len(0).is_ok()
        && handle.seek(SeekFrom::Start(0)).is_ok()
        && handle
            .write_all(id.as_bytes())
            .and_then(|()| handle.write_all(b"\n"))
            .is_ok()
        && handle.flush().is_ok()
    {
        tracing::debug!(target: "lifecycle_outbox", %id, "persisted the outbox identity");
    } else {
        tracing::warn!(
            target: "lifecycle_outbox",
            "could not persist the outbox identity to {}",
            path.display()
        );
    }
}

/// Non-blocking exclusive lock: `LOCK_EX | LOCK_NB` on Unix, `LockFileEx`
/// with `LOCKFILE_FAIL_IMMEDIATELY` on Windows. Released by the kernel when
/// the process dies, SIGKILL included, so the next boot can take the claim.
#[cfg(unix)]
fn try_lock_exclusive(file: &std::fs::File) -> std::io::Result<()> {
    use rustix::fs::{FlockOperation, flock};
    use std::os::unix::io::AsFd;
    match flock(file.as_fd(), FlockOperation::NonBlockingLockExclusive) {
        Ok(()) => Ok(()),
        Err(error) => Err(std::io::Error::from_raw_os_error(error.raw_os_error())),
    }
}

#[cfg(windows)]
fn try_lock_exclusive(file: &std::fs::File) -> std::io::Result<()> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY, LockFileEx,
    };
    let mut overlapped =
        std::mem::MaybeUninit::<windows_sys::Win32::System::IO::OVERLAPPED>::zeroed();
    let result = unsafe {
        LockFileEx(
            file.as_raw_handle() as _,
            LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
            0,
            u32::MAX,
            u32::MAX,
            overlapped.as_mut_ptr(),
        )
    };
    if result != 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// Test seam: release every held claim. The real session never calls this —
/// claims live for the process lifetime — but the kill/relaunch regression
/// needs to simulate process death in-process.
#[cfg(test)]
pub(crate) fn release_all_for_tests() {
    if let Some(registry) = CLAIMS.get() {
        registry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::lock_test_env;
    use codewhale_hooks::{LifecycleEvent, LifecycleOutbox};
    use serde_json::json;

    #[test]
    fn stable_identity_survives_a_restart() {
        let _lock = lock_test_env();
        let dir = tempfile::tempdir().expect("tempdir");
        // SAFETY: serialised by `lock_test_env`; removed under the same lock.
        unsafe {
            std::env::set_var("CODEWHALE_HOME", dir.path());
        }

        let first = acquire("tui").expect("acquire");
        assert!(first.starts_with("tui_"), "stable shape: {first}");
        release_all_for_tests();

        let second = acquire("tui").expect("acquire after restart");
        assert_eq!(first, second, "the persisted id is reused across launches");

        // SAFETY: cleanup under the same lock.
        unsafe {
            std::env::remove_var("CODEWHALE_HOME");
        }
    }

    #[test]
    fn a_concurrent_holder_forces_an_ephemeral_id() {
        let _lock = lock_test_env();
        let dir = tempfile::tempdir().expect("tempdir");
        // SAFETY: serialised by `lock_test_env`; removed under the same lock.
        unsafe {
            std::env::set_var("CODEWHALE_HOME", dir.path());
        }
        let identity_dir = dir.path().join("outbox");
        let identity_path = identity_dir.join("tui.identity");

        // Hold the claim directly (bypassing the registry) to simulate
        // another live process: flock is per open-file-description, so a
        // second handle in this process conflicts exactly like a second
        // process would.
        let (stable_id, held) = acquire_from_path("tui", &identity_dir, &identity_path);
        let held = held.expect("first acquirer holds the claim");
        assert!(stable_id.starts_with("tui_"));

        let (other_id, other_claim) = acquire_from_path("tui", &identity_dir, &identity_path);
        assert!(
            other_id.starts_with("tui_ephemeral_"),
            "a concurrent instance must not share the stable id: {other_id}"
        );
        assert!(other_claim.is_none(), "an ephemeral id holds no claim");
        drop(held);

        // After the holder dies, the claim is free and the stable id returns.
        let (reacquired, reacquired_claim) =
            acquire_from_path("tui", &identity_dir, &identity_path);
        assert_eq!(reacquired, stable_id);
        assert!(reacquired_claim.is_some());

        // SAFETY: cleanup under the same lock.
        unsafe {
            std::env::remove_var("CODEWHALE_HOME");
        }
    }

    #[test]
    fn registry_makes_acquire_idempotent_within_a_process() {
        let _lock = lock_test_env();
        let dir = tempfile::tempdir().expect("tempdir");
        // SAFETY: serialised by `lock_test_env`; removed under the same lock.
        unsafe {
            std::env::set_var("CODEWHALE_HOME", dir.path());
        }
        let first = acquire("exec").expect("acquire");
        let second = acquire("exec").expect("acquire again");
        assert_eq!(first, second);
        release_all_for_tests();

        // SAFETY: cleanup under the same lock.
        unsafe {
            std::env::remove_var("CODEWHALE_HOME");
        }
    }

    /// The TUI boot path's inputs, regression-tested in-process: a session
    /// that emitted under the stable identity "dies" (claims released, as
    /// process death would); the next "boot" reacquires the same persisted id
    /// and boot reconciliation pairs the orphan — the exact contract
    /// `event_loop.rs` wires between `acquire`, the outbox emits, and
    /// `reconcile_interrupted_turns`.
    #[test]
    fn boot_reconciliation_pairs_prior_session_records_under_the_stable_identity() {
        let _lock = lock_test_env();
        let dir = tempfile::tempdir().expect("tempdir");
        // SAFETY: serialised by `lock_test_env`; removed under the same lock.
        unsafe {
            std::env::set_var("CODEWHALE_HOME", dir.path());
        }
        let outbox_path = dir.path().join("outbox.jsonl");
        let outbox = LifecycleOutbox::new(Some(outbox_path.clone()), None, None);

        let thread_id = acquire("tui").expect("session boot acquires the identity");
        outbox
            .emit_blocking(LifecycleEvent {
                event: "turn_start".to_string(),
                kind: "turn.started".to_string(),
                thread_id: thread_id.clone(),
                turn_id: Some("open-turn".to_string()),
                item_id: None,
                payload: json!({ "workspace": "/tmp/w" }),
            })
            .expect("turn_start under the stable identity");

        // Simulate process death (SIGKILL runs no code): the claim is gone.
        release_all_for_tests();

        // Next boot: same persisted id, and boot reconciliation pairs the
        // killed session's open turn with a synthetic turn_end.
        let rebooted_id = acquire("tui").expect("next boot acquires the identity");
        assert_eq!(thread_id, rebooted_id, "the persisted identity is reused");
        let reconciled = outbox
            .reconcile_interrupted_turns(&rebooted_id, "boot_reconciliation")
            .expect("reconcile");
        assert_eq!(reconciled, 1);

        let text = std::fs::read_to_string(&outbox_path).expect("read outbox");
        let lines: Vec<serde_json::Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).expect("json line"))
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1]["event"], "turn_end");
        assert_eq!(lines[1]["thread_id"], thread_id.as_str());
        assert_eq!(lines[1]["turn_id"], "open-turn");
        assert_eq!(lines[1]["payload"]["reconciled"], true);
        assert_eq!(lines[1]["payload"]["reason"], "boot_reconciliation");

        // SAFETY: cleanup under the same lock.
        unsafe {
            std::env::remove_var("CODEWHALE_HOME");
        }
    }
}
