#![cfg_attr(not(unix), allow(dead_code))]
//! `shem sign` and `shem verify`, which hand a receipt to cosign. A script
//! stands in for cosign: it records its arguments, writes the bundle that
//! `sign-blob` is asked for, and fails when told to.
use std::path::Path;
use std::process::{Command, Output};

fn cosign(dir: &Path) -> std::path::PathBuf {
    let script = dir.join("cosign");
    std::fs::write(
        &script,
        r#"#!/bin/sh
echo "$@" >> "$0.log"
[ -n "$FAKE_COSIGN_FAIL" ] && exit 1
if [ "$1" = sign-blob ]; then echo '{}' > "$3"; fi
exit 0
"#,
    )
    .unwrap();
    shem::sys::fs::set_mode(&script, 0o755).unwrap();
    script
}

fn shem(dir: &Path, args: &[&str], fail: bool) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_shem"));
    command.args(args).env("SHEM_COSIGN", cosign(dir));
    if fail {
        command.env("FAKE_COSIGN_FAIL", "1");
    }
    command.output().unwrap()
}

#[cfg(unix)] // the stand-in for cosign is a shell script
#[test]
fn a_receipt_is_signed_and_its_signature_checked_through_cosign() {
    let dir = tempfile::tempdir().unwrap();
    let receipt = dir.path().join("receipt.json");
    std::fs::write(&receipt, "{}").unwrap();
    let path = receipt.to_str().unwrap();
    let bundle = format!("{path}.sigstore.json");

    // Without a signature there is nothing to check
    let out = shem(dir.path(), &["verify", path], false);
    assert_eq!(out.status.code(), Some(125));
    assert!(String::from_utf8_lossy(&out.stderr).contains("shem sign"));

    let out = shem(dir.path(), &["sign", path, "--key", "cosign.key"], false);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(Path::new(&bundle).is_file());
    let out = shem(
        dir.path(),
        &["verify", path, "--certificate-identity", "me@example.com"],
        false,
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("the signature holds"));

    let log = std::fs::read_to_string(dir.path().join("cosign.log")).unwrap();
    let lines: Vec<&str> = log.lines().collect();
    assert_eq!(
        lines[0],
        format!("sign-blob --bundle {bundle} --yes --key cosign.key {path}")
    );
    assert_eq!(
        lines[1],
        format!("verify-blob --bundle {bundle} --certificate-identity me@example.com {path}")
    );
}

#[cfg(unix)] // the stand-in for cosign is a shell script
#[test]
fn a_signature_cosign_refuses_is_a_failure() {
    let dir = tempfile::tempdir().unwrap();
    let receipt = dir.path().join("receipt.json");
    std::fs::write(&receipt, "{}").unwrap();
    std::fs::write(dir.path().join("receipt.json.sigstore.json"), "{}").unwrap();
    let out = shem(dir.path(), &["verify", receipt.to_str().unwrap()], true);
    assert_eq!(out.status.code(), Some(1));
    assert!(!String::from_utf8_lossy(&out.stderr).contains("the signature holds"));
    assert_eq!(
        shem(dir.path(), &["verify", "/no/such/receipt"], false)
            .status
            .code(),
        Some(125)
    );
}
