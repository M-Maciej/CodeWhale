//! End-to-end contract for the `[lifecycle_outbox]` feature on headless
//! `codewhale exec`: with a path configured, a run appends one JSONL
//! `RuntimeEventEnvelope` line per turn boundary (`turn_start` at message
//! dispatch, `turn_end` at the terminal receipt), the per-file `seq` recovers
//! across processes, and with no path configured no file is ever created.
//!
//! A `wiremock` OpenAI-compatible endpoint stands in for the provider, so the
//! run is a real `exec` process end to end — same loader, same engine, same
//! outbox writer — with no external network.

#![cfg(unix)]

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use serde_json::{Value, json};
use tempfile::TempDir;
use wait_timeout::ChildExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const TEST_MODEL: &str = "lifecycle-outbox-model";
const RUN_TIMEOUT: Duration = Duration::from_secs(60);

/// Placeholder in `outbox_toml` replaced with the isolated home's absolute
/// outbox path (so callers can read the file back after the run).
const OUTBOX_PATH_TOKEN: &str = "__OUTBOX_PATH__";

fn sse_chunk(value: Value) -> String {
    format!(
        "data: {}\n\n",
        serde_json::to_string(&value).expect("SSE JSON")
    )
}

/// Final-answer SSE: one content delta, then a clean stop.
fn answer_sse(answer: &str) -> String {
    [
        sse_chunk(json!({
            "id": "chatcmpl-outbox",
            "object": "chat.completion.chunk",
            "model": TEST_MODEL,
            "choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": null}]
        })),
        sse_chunk(json!({
            "id": "chatcmpl-outbox",
            "object": "chat.completion.chunk",
            "model": TEST_MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        })),
        "data: [DONE]\n\n".to_string(),
    ]
    .join("")
}

async fn start_mock_llm() -> MockServer {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(json!({
                    "object": "list",
                    "data": [{ "id": TEST_MODEL, "object": "model" }]
                })),
        )
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .insert_header("cache-control", "no-cache")
                .set_body_string(answer_sse("ok")),
        )
        .mount(&server)
        .await;

    server
}

fn preserve_host_env(command: &mut Command) {
    command.env_clear();
    for key in [
        "PATH",
        "PATHEXT",
        "SystemRoot",
        "SystemDrive",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "TERM",
        "COLORTERM",
        "LANG",
        "LC_ALL",
    ] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
}

/// Build the `codewhale exec` command against the mock provider with the
/// given `[lifecycle_outbox]` config block (already TOML-formatted, may be
/// empty). Any `__OUTBOX_PATH__` token in it is replaced with the isolated
/// home's absolute outbox path. Writes the isolated home's config file.
fn exec_command(
    server: &MockServer,
    outbox_toml: &str,
    home: &TempDir,
    workspace: &TempDir,
) -> Command {
    let outbox_path = home_outbox_path(home);
    let outbox_toml = outbox_toml.replace(OUTBOX_PATH_TOKEN, &outbox_path.display().to_string());

    std::fs::create_dir_all(home.path().join(".codewhale")).expect("create codewhale config dir");
    std::fs::create_dir_all(home.path().join(".deepseek")).expect("create deepseek config dir");
    std::fs::write(
        home.path().join(".codewhale").join("config.toml"),
        format!("provider = \"deepseek\"\nmodel = \"{TEST_MODEL}\"\n{outbox_toml}"),
    )
    .expect("write exec config");

    let mut command = Command::new(codewhale_tui_binary());
    preserve_host_env(&mut command);
    command
        .current_dir(workspace.path())
        .arg("--workspace")
        .arg(workspace.path())
        .arg("--no-project-config")
        .arg("exec")
        .arg("--auto")
        .arg("--model")
        .arg(TEST_MODEL)
        .arg("answer briefly")
        .env("HOME", home.path())
        .env("USERPROFILE", home.path())
        .env("XDG_CONFIG_HOME", home.path().join(".config"))
        .env("XDG_DATA_HOME", home.path().join(".local").join("share"))
        .env("XDG_CACHE_HOME", home.path().join(".cache"))
        .env(
            "CODEWHALE_CONFIG_PATH",
            home.path().join(".codewhale").join("config.toml"),
        )
        .env(
            "DEEPSEEK_CONFIG_PATH",
            home.path().join(".deepseek").join("config.toml"),
        )
        .env("DEEPSEEK_API_KEY", "ci-test-key-not-real")
        .env("DEEPSEEK_BASE_URL", server.uri())
        .env("CODEWHALE_BASE_URL", server.uri())
        .env("DEEPSEEK_MODEL", TEST_MODEL)
        .env("CODEWHALE_MODEL", TEST_MODEL)
        .env("RUST_LOG", "warn")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

/// A spawned exec process plus its background pipe readers.
type ExecSpawn = (
    std::process::Child,
    std::thread::JoinHandle<std::io::Result<Vec<u8>>>,
    std::thread::JoinHandle<std::io::Result<Vec<u8>>>,
);

/// Spawn the exec command and read its pipes in the background.
fn spawn_exec_command(mut command: Command) -> ExecSpawn {
    let mut child = command.spawn().expect("spawn codewhale-tui exec");
    let stdout_reader = read_pipe_in_background(child.stdout.take().expect("stdout pipe"));
    let stderr_reader = read_pipe_in_background(child.stderr.take().expect("stderr pipe"));
    (child, stdout_reader, stderr_reader)
}

/// Wait for the exec process and assert success, returning its output.
fn finish_exec_success(
    mut child: std::process::Child,
    stdout_reader: std::thread::JoinHandle<std::io::Result<Vec<u8>>>,
    stderr_reader: std::thread::JoinHandle<std::io::Result<Vec<u8>>>,
) -> (Vec<u8>, Vec<u8>) {
    let status = match child
        .wait_timeout(RUN_TIMEOUT)
        .expect("wait for codewhale-tui")
    {
        Some(status) => status,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            let stdout = join_pipe_reader(stdout_reader, "stdout");
            let stderr = join_pipe_reader(stderr_reader, "stderr");
            panic!(
                "codewhale-tui exec timed out after {RUN_TIMEOUT:?}\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&stdout),
                String::from_utf8_lossy(&stderr)
            );
        }
    };

    let stdout = join_pipe_reader(stdout_reader, "stdout");
    let stderr = join_pipe_reader(stderr_reader, "stderr");
    assert!(
        status.success(),
        "codewhale-tui exec failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&stdout),
        String::from_utf8_lossy(&stderr)
    );
    (stdout, stderr)
}

/// Run `codewhale exec` against the mock provider with the given
/// `[lifecycle_outbox]` config block (already TOML-formatted, may be empty).
/// Any `__OUTBOX_PATH__` token in it is replaced with the isolated home's
/// absolute outbox path. Returns the isolated home and workspace dirs (the
/// latter so callers can assert the outbox `payload.workspace` exactly).
fn run_exec_with_outbox_config(server: &MockServer, outbox_toml: &str) -> (TempDir, TempDir) {
    let workspace = TempDir::new().expect("workspace tempdir");
    let home = TempDir::new().expect("home tempdir");
    let command = exec_command(server, outbox_toml, &home, &workspace);
    let (child, stdout_reader, stderr_reader) = spawn_exec_command(command);
    finish_exec_success(child, stdout_reader, stderr_reader);
    (home, workspace)
}

fn read_pipe_in_background<R>(mut reader: R) -> std::thread::JoinHandle<std::io::Result<Vec<u8>>>
where
    R: Read + Send + 'static,
{
    std::thread::spawn(move || {
        let mut output = Vec::new();
        reader.read_to_end(&mut output).map(|_| output)
    })
}

fn join_pipe_reader(
    handle: std::thread::JoinHandle<std::io::Result<Vec<u8>>>,
    stream_name: &str,
) -> Vec<u8> {
    handle
        .join()
        .expect("pipe reader join")
        .unwrap_or_else(|err| panic!("failed to read {stream_name}: {err}"))
}

fn read_outbox_lines(path: &Path) -> Vec<Value> {
    let text = std::fs::read_to_string(path).expect("read outbox file");
    text.lines()
        .map(|line| {
            serde_json::from_str(line)
                .unwrap_or_else(|err| panic!("outbox line should parse: {err}\nline: {line}"))
        })
        .collect()
}

fn codewhale_tui_binary() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_codewhale-tui") {
        return PathBuf::from(path);
    }
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_codewhale-tui") {
        return PathBuf::from(path);
    }

    let mut path = std::env::current_exe().expect("current test executable path");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push(format!("codewhale-tui{}", std::env::consts::EXE_SUFFIX));
    path
}

fn home_outbox_path(home: &TempDir) -> PathBuf {
    home.path()
        .join(".codewhale")
        .join("notifications")
        .join("outbox.jsonl")
}

#[tokio::test(flavor = "multi_thread")]
async fn exec_emits_turn_start_and_turn_end_to_the_configured_outbox() {
    let server = start_mock_llm().await;
    let (home, workspace) = run_exec_with_outbox_config(
        &server,
        &format!("[lifecycle_outbox]\npath = {}\n", json!(OUTBOX_PATH_TOKEN)),
    );

    let outbox_path = home_outbox_path(&home);
    assert!(outbox_path.exists(), "outbox file must be created");
    let lines = read_outbox_lines(&outbox_path);
    assert_eq!(
        lines.len(),
        2,
        "one turn_start and one turn_end line: {lines:#?}"
    );

    let start = &lines[0];
    assert_eq!(start["event"], "turn_start");
    assert_eq!(start["kind"], "turn.started");
    assert_eq!(start["schema_version"], 1);
    assert_eq!(start["seq"], 1);
    assert!(start["timestamp"].as_str().is_some());
    // One run identity minted before dispatch: both boundaries carry the
    // same stable exec thread id and the same per-run turn id, so a
    // supervisor can pair them (the engine's own session id changes during
    // the run and is deliberately not the pairing key).
    let thread_id = start["thread_id"].as_str().expect("thread_id");
    assert!(
        thread_id.starts_with("exec_"),
        "thread_id is the stable exec identity: {thread_id}"
    );
    let turn_id = start["turn_id"].as_str().expect("turn_id");
    assert!(turn_id.starts_with("turn_"), "turn_id: {turn_id}");
    // Every payload carries the workspace for consumer-side routing; exec
    // runs with `--workspace <dir>`, so the emitted path must match it.
    assert_eq!(
        start["payload"]["workspace"],
        json!(workspace.path().to_string_lossy().as_ref()),
        "turn_start must carry the workspace"
    );

    let end = &lines[1];
    assert_eq!(end["event"], "turn_end");
    assert_eq!(end["kind"], "turn.completed");
    assert_eq!(end["seq"], 2);
    assert_eq!(end["payload"]["status"], "completed");
    assert!(end["payload"]["error"].is_null());
    assert!(end["payload"]["duration_ms"].as_u64().is_some());
    assert_eq!(
        end["payload"]["workspace"],
        json!(workspace.path().to_string_lossy().as_ref()),
        "turn_end must carry the workspace"
    );
    // The boundary pair is correlatable: same thread id, same turn id.
    assert_eq!(end["thread_id"], start["thread_id"]);
    assert_eq!(end["turn_id"], start["turn_id"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn exec_without_outbox_config_writes_no_file() {
    let server = start_mock_llm().await;
    let (home, _workspace) = run_exec_with_outbox_config(&server, "");

    assert!(
        !home_outbox_path(&home).exists(),
        "no outbox file must be created when [lifecycle_outbox] is unset"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn outbox_seq_recovers_across_processes() {
    let server = start_mock_llm().await;

    // First run writes seq 1 (turn_start) and 2 (turn_end).
    let (home, _workspace) = run_exec_with_outbox_config(
        &server,
        &format!("[lifecycle_outbox]\npath = {}\n", json!(OUTBOX_PATH_TOKEN)),
    );
    let shared_outbox = home_outbox_path(&home);

    // Second process, pointing at the SAME file: seq must continue at 3.
    let (_second_home, _second_workspace) = run_exec_with_outbox_config(
        &server,
        &format!(
            "[lifecycle_outbox]\npath = {}\n",
            json!(shared_outbox.display().to_string())
        ),
    );

    let lines = read_outbox_lines(&shared_outbox);
    assert_eq!(lines.len(), 4, "two runs, four lines: {lines:#?}");
    let seqs: Vec<u64> = lines
        .iter()
        .map(|line| line["seq"].as_u64().expect("seq"))
        .collect();
    assert_eq!(
        seqs,
        vec![1, 2, 3, 4],
        "seq must be monotonic across processes"
    );
}

/// A mock endpoint that stalls: the chat-completions request stays open far
/// beyond the test window, so the exec run sits mid-turn until it is killed.
async fn start_stalled_mock_llm() -> MockServer {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(json!({
                    "object": "list",
                    "data": [{ "id": TEST_MODEL, "object": "model" }]
                })),
        )
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_delay(Duration::from_secs(600)),
        )
        .mount(&server)
        .await;

    server
}

/// Poll the outbox file until `want` lines exist (or the timeout expires),
/// returning the lines read so far.
fn wait_for_outbox_lines(path: &Path, want: usize, timeout: Duration) -> Vec<Value> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if path.exists() {
            let lines = read_outbox_lines(path);
            if lines.len() >= want {
                return lines;
            }
        }
        assert!(
            std::time::Instant::now() < deadline,
            "outbox {} never reached {want} line(s)\nnow:\n{:?}",
            path.display(),
            if path.exists() {
                read_outbox_lines(path)
            } else {
                Vec::new()
            }
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Shared-file concurrency: two exec processes appending to ONE outbox file
/// at the same time produce only complete, parseable lines with unique,
/// strictly increasing seqs — the per-append lock serializes recovery + write
/// across processes. Both runs share the home, so the second run's identity
/// acquire exercises the live-holder guard (ephemeral fallback) while the
/// first holds the stable claim.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_exec_runs_interleave_cleanly_in_one_shared_outbox() {
    let server = start_mock_llm().await;
    let workspace = TempDir::new().expect("workspace tempdir");
    let home = TempDir::new().expect("home tempdir");
    let outbox_toml = format!("[lifecycle_outbox]\npath = {}\n", json!(OUTBOX_PATH_TOKEN));

    let first_command = exec_command(&server, &outbox_toml, &home, &workspace);
    let second_command = exec_command(&server, &outbox_toml, &home, &workspace);
    let (first_child, first_stdout, first_stderr) = spawn_exec_command(first_command);
    let (second_child, second_stdout, second_stderr) = spawn_exec_command(second_command);

    finish_exec_success(first_child, first_stdout, first_stderr);
    finish_exec_success(second_child, second_stdout, second_stderr);

    let lines = read_outbox_lines(&home_outbox_path(&home));
    assert_eq!(lines.len(), 4, "two runs × two boundaries: {lines:#?}");
    let seqs: Vec<u64> = lines
        .iter()
        .map(|line| line["seq"].as_u64().expect("seq"))
        .collect();
    let mut sorted = seqs.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(seqs, sorted, "seqs are unique and monotonic in file order");
    assert_eq!(sorted, vec![1, 2, 3, 4]);
    // Interleaving between the two processes is legal; pairing is by
    // identity, not file position: each run's two lines share one turn id
    // and one thread id.
    let mut by_turn: std::collections::BTreeMap<String, Vec<&Value>> =
        std::collections::BTreeMap::new();
    for line in &lines {
        by_turn
            .entry(line["turn_id"].as_str().expect("turn_id").to_string())
            .or_default()
            .push(line);
    }
    assert_eq!(by_turn.len(), 2, "two runs, two distinct turn ids");
    for (turn_id, pair) in by_turn {
        assert_eq!(
            pair.len(),
            2,
            "turn {turn_id} has exactly one start and one end"
        );
        let mut events: Vec<&str> = pair
            .iter()
            .map(|line| line["event"].as_str().expect("event"))
            .collect();
        events.sort_unstable();
        assert_eq!(events, vec!["turn_end", "turn_start"]);
        assert_eq!(
            pair[0]["thread_id"], pair[1]["thread_id"],
            "a run's boundaries share its thread id"
        );
    }
}

/// Killed/restarted exec regression: run one dies mid-turn to a SIGKILL (no
/// code runs, so its `turn_start` stays unpaired in the file); the next run
/// on the same machine (same home, same stable exec identity) reconciles it
/// at boot with a synthetic `turn_end`, before emitting its own pair.
#[tokio::test(flavor = "multi_thread")]
async fn killed_exec_run_is_reconciled_by_the_next_run_on_boot() {
    let stalled_server = start_stalled_mock_llm().await;
    let workspace = TempDir::new().expect("workspace tempdir");
    let home = TempDir::new().expect("home tempdir");
    let outbox_toml = format!("[lifecycle_outbox]\npath = {}\n", json!(OUTBOX_PATH_TOKEN));
    let outbox_path = home_outbox_path(&home);

    // Run one: stalls mid-turn; wait for its turn_start to land, then SIGKILL.
    let command = exec_command(&stalled_server, &outbox_toml, &home, &workspace);
    let (mut first_child, first_stdout, first_stderr) = spawn_exec_command(command);
    let lines = wait_for_outbox_lines(&outbox_path, 1, Duration::from_secs(60));
    assert_eq!(
        lines.len(),
        1,
        "the stalled run has emitted only turn_start"
    );
    assert_eq!(lines[0]["event"], "turn_start");
    first_child.kill().expect("SIGKILL the stalled run");
    let _ = first_child.wait();
    drop(join_pipe_reader(first_stdout, "stdout"));
    drop(join_pipe_reader(first_stderr, "stderr"));

    // Run two (fresh mock, same home → same persisted exec identity, and the
    // dead process's claim is gone): its boot reconciliation pairs the orphan.
    let fast_server = start_mock_llm().await;
    let command = exec_command(&fast_server, &outbox_toml, &home, &workspace);
    let (second_child, second_stdout, second_stderr) = spawn_exec_command(command);
    finish_exec_success(second_child, second_stdout, second_stderr);

    let lines = read_outbox_lines(&outbox_path);
    assert_eq!(
        lines.len(),
        4,
        "orphan start + synthetic end + the second run's own pair: {lines:#?}"
    );

    let killed_start = &lines[0];
    let synthetic_end = &lines[1];
    let second_start = &lines[2];
    let second_end = &lines[3];
    assert_eq!(killed_start["event"], "turn_start");
    assert_eq!(synthetic_end["event"], "turn_end");
    assert_eq!(synthetic_end["payload"]["status"], "interrupted");
    assert_eq!(synthetic_end["payload"]["reconciled"], true);
    assert_eq!(synthetic_end["payload"]["reason"], "boot_reconciliation");
    assert_eq!(
        synthetic_end["thread_id"], killed_start["thread_id"],
        "the boot reconciled under the same stable exec identity"
    );
    assert_eq!(
        synthetic_end["turn_id"], killed_start["turn_id"],
        "the synthetic end pairs the killed run's turn id"
    );
    assert!(
        killed_start["thread_id"]
            .as_str()
            .expect("thread_id")
            .starts_with("exec_"),
        "the stable identity was used, not a per-launch session id"
    );
    assert_eq!(second_start["event"], "turn_start");
    assert_eq!(second_end["event"], "turn_end");
    assert_eq!(second_start["thread_id"], killed_start["thread_id"]);
    assert_eq!(second_start["turn_id"], second_end["turn_id"]);
    assert_ne!(
        second_start["turn_id"], killed_start["turn_id"],
        "each run mints its own turn id"
    );
}
