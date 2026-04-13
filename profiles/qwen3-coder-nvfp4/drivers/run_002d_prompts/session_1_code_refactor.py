"""Run 002d -- Session 1 prompts: multi-turn Python code refactor.

Scenario: A senior engineer is asked to review and incrementally refactor the
``mesh/commands/smb.py`` module that handles SMB/CIFS share mounting on Linux.
The seed message embeds the entire module verbatim.  Subsequent turns iterate
toward a production-grade refactor with tests, cross-file interaction analysis
(with ``mesh/utils/process.py``), migration plan, observability, hypothesis
property tests, merge-conflict resolution, and a PR description.

Embedded files are pulled verbatim from ``mesh/src/mesh/``:
    - Seed primary: ``mesh/src/mesh/commands/smb.py``
    - Follow-up #3 second file: ``mesh/src/mesh/utils/process.py``
      (imported by smb.py at line 12: ``from mesh.utils.process import
      command_exists, run_sudo``).

All string literals are valid Python; triple-double-quote sequences in the
embedded sources are escaped as backslash-quote triples inside the raw
r-string SEED_PROMPT and as escaped quotes in the regular-string FOLLOW_UPS
so the module parses cleanly.

Token counts reported by ``estimate_tokens()`` use the convention
``len(text) // 4`` (character-based approximation).
"""

from __future__ import annotations

SEED_TOPIC: str = (
    "Python code refactor: mesh/src/mesh/commands/smb.py "
    "(SMB/CIFS share setup, client mapping, status — with subprocess, "
    "credential management, systemd, and cross-platform code paths)"
)


# ---------------------------------------------------------------------------
# SEED PROMPT
# ---------------------------------------------------------------------------
#
# The seed embeds the primary refactor target (commands/smb.py).  It is the
# Typer subcommand group that handles Samba server setup, user management,
# Windows client drive mapping via SSH, and service status checking.  The
# module directly uses subprocess calls (both via run_sudo from
# mesh.utils.process and via raw subprocess.run), manages credentials, writes
# to system config files (/etc/samba/smb.conf), creates systemd drop-ins,
# and shells out to SSH for remote Windows operations.
#
# The single file under active refactor is commands/smb.py.
# ---------------------------------------------------------------------------

SEED_PROMPT: str = r"""You are reviewing this module as a senior engineer. First, read it carefully and identify the five most concerning code quality, correctness, or architectural issues. Cite specific line ranges and function names. For each issue, explain what could go wrong in production and propose the fix direction without writing the fix yet.

The primary file under review is `mesh/commands/smb.py`. It is the Typer subcommand group that handles SMB/CIFS file sharing across a self-hosted mesh network: setting up a Samba server on Linux, adding SMB users, mapping network drives on Windows hosts via SSH, and checking service status. At ~595 lines it is one of the larger modules in the mesh CLI codebase and touches an unusually wide surface: it writes to `/etc/samba/smb.conf`, creates systemd drop-in files, manages SMB passwords via `smbpasswd`, generates PowerShell scripts, pipes them over SSH, checks firewall rules, and queries service state. That is a lot of system-level side effects for one module, which is part of why we are reviewing it.

Because the review needs to account for how this module interacts with its dependencies, here is the import context:

- `from mesh.core.config import get_shared_folder` — returns the platform-appropriate shared folder Path
- `from mesh.core.environment import OSType, detect_os_type` — returns an enum (UBUNTU, WSL2, WINDOWS, MACOS, UNKNOWN)
- `from mesh.utils.output import error, info, ok, section, warn` — Rich console output helpers
- `from mesh.utils.process import command_exists, run_sudo` — `command_exists` wraps `shutil.which`; `run_sudo` prepends `sudo` and returns a `CommandResult(returncode, stdout, stderr)` dataclass
- `from mesh.utils.ssh import ssh_to_host` — runs a command on a remote host via SSH, returns `(success: bool, output: str)`

Specific things I want you to scrutinize in `commands/smb.py`, because they are the places I suspect we have latent bugs or security issues:

- **Credential handling in `_set_smb_password`** (lines 65-80): The password is piped to `smbpasswd -a -s` via `subprocess.run(input=...)`. Is this secure? What happens if the password contains shell metacharacters, newlines, or null bytes? What happens if `sudo` prompts for authentication — does the password get consumed as the sudo password instead of the smbpasswd password? Is there a TOCTOU between the password prompt and the smbpasswd call?

- **The `setup_server` command's config file manipulation** (lines 206-236): It appends a share definition to `/etc/samba/smb.conf` using `sudo tee -a`. What happens if setup_server is run twice — does it check for an existing share before appending? (It does call `_share_exists_in_testparm`, but is testparm's output format guaranteed to be stable across Samba versions?) What happens if the append fails mid-write — is the config file left in a valid state? Is there a backup mechanism?

- **The subprocess usage pattern** — the module mixes two approaches: (1) using `run_sudo()` from `mesh.utils.process` which returns a `CommandResult` with proper error handling, and (2) calling `subprocess.run()` directly with ad-hoc error handling. This inconsistency means some calls get timeout handling and some don't, some get proper error reporting and some swallow stderr. Count how many direct `subprocess.run` calls there are vs `run_sudo` calls and assess whether the direct calls could be replaced.

- **The `setup_client` command** (lines 344-513): It generates a PowerShell script, pipes it over SSH to a Windows host, and then tells the user to run it manually. The script embeds the SMB password in a `net use` command via `$Password` variable. Is the generated script safe against injection if the username, server, share name, or drive letter contain special characters? What about the startup reconnect script at line 451-453 — it uses `net use` without a password, relying on persistent credentials. Does that actually work across reboots? What happens if the user runs setup_client twice with different parameters?

- **Error handling asymmetry** — some operations call `raise typer.Exit(1)` on failure, others just `warn()` and continue. Is there a consistent policy? For instance, if the systemd drop-in creation fails (line 274), the code warns but continues. If the UFW rule fails (line 261), same. But if the share directory creation fails (line 203), it exits. Is this the right severity ranking? What should the invariant be?

- **The `status` command** (lines 515-595): It queries multiple subsystems (Samba installation, service status, testparm output, port 445, Tailscale). If any of these checks fail or time out, the command continues to the next check. Is this the right UX? Should there be a machine-readable output mode (JSON) for scripting? Is the 10-second timeout on each subprocess call appropriate, or could a hung `testparm` or `ss` command block the status command for 40+ seconds total?

- **Platform detection** — `setup_server` checks for `OSType.UBUNTU` specifically (line 155), but `status` checks for `OSType.UBUNTU` or `OSType.WSL2` (line 523). Should `setup_server` also work on WSL2? The inconsistency suggests either a bug or an undocumented design decision.

- **The `_write_systemd_dropin` helper** (lines 97-128): It checks if the file exists, creates the directory, and writes the file — but there's no atomicity. If the directory creation succeeds but the file write fails, subsequent runs will skip the directory creation (it exists) but still need to write the file. The `Path(dropin_path).exists()` check at line 108 short-circuits if the file exists, but what if the file exists with wrong content from a previous partial write?

- **State management across the setup_server steps** — the command runs 10 sequential steps. If step 6 (set SMB password) fails but steps 1-5 succeeded, the system is left with Samba installed, a share directory created, and a share configured in smb.conf, but no valid user. Is there any rollback? If the user re-runs setup_server, will steps 1-5 be idempotent? The docstring claims "Idempotent: safe to run multiple times" — verify whether that claim is actually true for every step.

- **The generated PowerShell script** (lines 398-467): It uses `$Cred = Get-Credential` which pops up a GUI dialog. What happens if this script is run in a non-interactive PowerShell session (e.g., scheduled task, headless SSH)? The script also modifies the Windows registry (`EnableLinkedConnections`) and creates files in the Startup folder — these are significant side effects that persist beyond the script's execution. Is the user warned about this?

Produce the five most concerning findings. Rank them by severity (highest first). For each:

- A short headline (one sentence).
- Line range and function name(s) involved.
- What can go wrong in production — concrete failure modes, not abstract worries.
- Fix direction in 2-4 sentences. Do not write the fix yet.

Below is the full source of the file under review.

=== FILE: mesh/commands/smb.py (PRIMARY REFACTOR TARGET) ===

```python
\"\"\"Samba/SMB file sharing commands.\"\"\"

import getpass
import subprocess
from pathlib import Path

import typer

from mesh.core.config import get_shared_folder
from mesh.core.environment import OSType, detect_os_type
from mesh.utils.output import error, info, ok, section, warn
from mesh.utils.process import command_exists, run_sudo
from mesh.utils.ssh import ssh_to_host

app = typer.Typer(
    name="smb",
    help="Samba/SMB file sharing",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_tailscale_connected() -> bool:
    \"\"\"Check if Tailscale is connected.\"\"\"
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            import json

            data = json.loads(result.stdout)
            return data.get("BackendState") == "Running"
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return False


def _share_exists_in_testparm(share_name: str) -> bool:
    \"\"\"Check if a share section already exists via testparm.\"\"\"
    try:
        result = subprocess.run(
            ["testparm", "-s"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # testparm outputs section headers as [name]
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped == f"[{share_name}]":
                return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False


def _set_smb_password(user: str, password: str) -> bool:
    \"\"\"Set Samba password for a user via smbpasswd.

    Uses subprocess directly because run_sudo() does not support input=.
    \"\"\"
    try:
        result = subprocess.run(
            ["sudo", "smbpasswd", "-a", "-s", user],
            input=f"{password}\n{password}\n",
            text=True,
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _systemctl_is_active(service: str) -> bool:
    \"\"\"Check if a systemd service is active.\"\"\"
    try:
        result = subprocess.run(
            ["systemctl", "is-active", service],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() == "active"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _write_systemd_dropin(service: str) -> bool:
    \"\"\"Create a systemd restart drop-in for the given service.

    Creates /etc/systemd/system/{service}.service.d/10-restart.conf.
    Uses subprocess + sudo tee since Write tool cannot access /etc.
    \"\"\"
    dropin_dir = f"/etc/systemd/system/{service}.service.d"
    dropin_path = f"{dropin_dir}/10-restart.conf"
    dropin_content = "[Service]\nRestart=on-failure\nRestartSec=5\n"

    # Check if already exists
    if Path(dropin_path).exists():
        return True

    # Create directory
    result = run_sudo(["mkdir", "-p", dropin_dir])
    if not result.success:
        return False

    # Write file via sudo tee
    try:
        proc = subprocess.run(
            ["sudo", "tee", dropin_path],
            input=dropin_content,
            text=True,
            capture_output=True,
            timeout=10,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def setup_server(
    share: str = typer.Option("shared", "--share", help="Share name"),
    path: str = typer.Option(None, "--path", "-p", help="Share directory path"),
    user: str = typer.Option(None, "--user", "-u", help="SMB user (default: current user)"),
    password: str = typer.Option(
        ..., "--password", prompt=True, hide_input=True, help="SMB password"
    ),
) -> None:
    \"\"\"Set up Samba server on this Linux host.

    Installs Samba, creates a share, sets a user password,
    configures systemd restart policies, and enables services.

    Idempotent: safe to run multiple times.
    \"\"\"
    section("SMB Server Setup")

    # --- 1. Check OS is Linux ---
    os_type = detect_os_type()
    if os_type != OSType.UBUNTU:
        error(f"This command requires Ubuntu/Linux, detected: {os_type.value}")
        raise typer.Exit(1)

    # --- 2. Check Tailscale ---
    if command_exists("tailscale"):
        if _is_tailscale_connected():
            ok("Tailscale connected")
        else:
            warn("Tailscale not connected - SMB will work on LAN but mesh IPs won't resolve")
    else:
        warn("Tailscale not installed - SMB will only be available on LAN")

    # --- Resolve defaults ---
    share_path = Path(path) if path else get_shared_folder()
    smb_user = user or getpass.getuser()

    # --- 3. Install Samba ---
    if command_exists("smbd"):
        ok("Samba already installed")
    else:
        info("Installing Samba...")
        run_sudo(
            ["apt-get", "update"],
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        result = run_sudo(
            ["apt-get", "install", "-y", "samba"],
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        if result.success:
            ok("Samba installed")
        else:
            error(f"Failed to install Samba: {result.stderr}")
            raise typer.Exit(1)

    # --- 4. Create share directory ---
    if share_path.exists():
        ok(f"Share directory exists: {share_path}")
    else:
        info(f"Creating share directory: {share_path}")
        result = run_sudo(["mkdir", "-p", str(share_path)])
        if result.success:
            # Set ownership to the SMB user
            run_sudo(["chown", f"{smb_user}:{smb_user}", str(share_path)])
            run_sudo(["chmod", "2775", str(share_path)])
            ok(f"Created {share_path}")
        else:
            error(f"Failed to create directory: {result.stderr}")
            raise typer.Exit(1)

    # --- 5. Configure share in smb.conf ---
    if _share_exists_in_testparm(share):
        ok(f"Share [{share}] already configured")
    else:
        info(f"Adding [{share}] to /etc/samba/smb.conf...")
        share_config = (
            f"\n[{share}]\n"
            f"   path = {share_path}\n"
            f"   browseable = yes\n"
            f"   read only = no\n"
            f"   guest ok = no\n"
            f"   valid users = {smb_user}\n"
            f"   create mask = 0664\n"
            f"   directory mask = 2775\n"
        )
        try:
            proc = subprocess.run(
                ["sudo", "tee", "-a", "/etc/samba/smb.conf"],
                input=share_config,
                text=True,
                capture_output=True,
                timeout=10,
            )
            if proc.returncode == 0:
                ok(f"Share [{share}] added to smb.conf")
            else:
                error(f"Failed to update smb.conf: {proc.stderr}")
                raise typer.Exit(1)
        except subprocess.TimeoutExpired:
            error("Timed out writing to smb.conf")
            raise typer.Exit(1) from None

    # --- 6. Set SMB password ---
    info(f"Setting SMB password for user '{smb_user}'...")
    if _set_smb_password(smb_user, password):
        ok(f"SMB password set for '{smb_user}'")
    else:
        error(f"Failed to set SMB password for '{smb_user}'")
        raise typer.Exit(1)

    # --- 7. Check UFW ---
    if command_exists("ufw"):
        try:
            result = subprocess.run(
                ["ufw", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if "active" in result.stdout.lower() and "inactive" not in result.stdout.lower():
                info("UFW active - adding Samba rule...")
                fw_result = run_sudo(["ufw", "allow", "samba"])
                if fw_result.success:
                    ok("Samba allowed through UFW")
                else:
                    warn(f"Failed to add UFW rule: {fw_result.stderr}")
            else:
                info("UFW inactive - no firewall rule needed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            info("Could not check UFW status")
    else:
        info("UFW not installed - skipping firewall configuration")

    # --- 8. Create systemd drop-ins ---
    for service in ("smbd", "nmbd"):
        if _write_systemd_dropin(service):
            ok(f"Systemd restart drop-in for {service}")
        else:
            warn(f"Failed to create systemd drop-in for {service}")

    # --- 9. Enable and start services ---
    run_sudo(["systemctl", "daemon-reload"])
    for service in ("smbd", "nmbd"):
        result = run_sudo(["systemctl", "enable", "--now", service])
        if result.success:
            ok(f"{service} enabled and started")
        else:
            warn(f"Failed to enable {service}: {result.stderr}")

    # --- 10. Verify ---
    info("Verifying configuration...")
    try:
        result = subprocess.run(
            ["testparm", "-s"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and f"[{share}]" in result.stdout:
            ok("Configuration verified via testparm")
        else:
            warn("testparm did not confirm share - check /etc/samba/smb.conf manually")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        warn("Could not run testparm to verify")

    section("SMB Server Setup Complete")
    info(f"Share: \\\\<this-host>\\{share}")
    info(f"Path: {share_path}")
    info(f"User: {smb_user}")


@app.command()
def add_user(
    user: str = typer.Argument(..., help="Username to add to Samba"),
    password: str = typer.Option(
        ..., "--password", prompt=True, hide_input=True, help="SMB password"
    ),
) -> None:
    \"\"\"Add or update a Samba user password.

    The user must already exist as a system user.
    \"\"\"
    section("Add SMB User")

    info(f"Setting SMB password for '{user}'...")
    if _set_smb_password(user, password):
        ok(f"SMB password set for '{user}'")
    else:
        error(f"Failed to set SMB password for '{user}'")
        raise typer.Exit(1)

    # Verify user appears in pdbedit
    info("Verifying user in Samba database...")
    try:
        result = subprocess.run(
            ["sudo", "pdbedit", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and user in result.stdout:
            ok(f"User '{user}' confirmed in Samba database")
        else:
            warn(f"Could not confirm '{user}' in pdbedit output")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        warn("Could not run pdbedit to verify")


@app.command()
def setup_client(
    host: str = typer.Option(..., "--host", "-h", help="Windows SSH host"),
    port: int = typer.Option(22, "--port", "-p", help="SSH port"),
    server: str = typer.Option(..., "--server", "-s", help="SMB server hostname or IP"),
    share: str = typer.Option("shared", "--share", help="Share name"),
    drive: str = typer.Option("Z:", "--drive", "-d", help="Windows drive letter"),
    user: str = typer.Option(..., "--user", "-u", help="SMB username"),
) -> None:
    \"\"\"Set up SMB drive mapping on a Windows host via SSH.

    Runs from Linux, connects to Windows via SSH, and generates
    a PowerShell script for drive mapping.
    \"\"\"
    section("SMB Client Setup")

    # --- 1. Test SSH connectivity ---
    info(f"Testing SSH connectivity to {host}:{port}...")
    success, output = ssh_to_host(host, "echo connected", port=port)
    if not success:
        error(f"Cannot SSH to {host}:{port}: {output}")
        raise typer.Exit(1)
    ok(f"Connected to {host}")

    # --- 2. Test SMB port on server ---
    info(f"Testing SMB connectivity from {host} to {server}:445...")
    smb_test_cmd = (
        f'powershell -Command "'
        f"(Test-NetConnection {server} -Port 445 -WarningAction SilentlyContinue)"
        f'.TcpTestSucceeded"'
    )
    success, output = ssh_to_host(host, smb_test_cmd, port=port)
    if success and "True" in output:
        ok(f"SMB port 445 reachable on {server}")
    else:
        warn(f"SMB port 445 may not be reachable on {server} from {host}")
        info("Continuing anyway - the generated script will retry at runtime")

    # --- 3. Ensure C:\temp exists ---
    info("Ensuring C:\\temp exists on Windows...")
    mkdir_cmd = (
        'powershell -Command "New-Item -Path C:\\temp -ItemType Directory -Force | Out-Null"'
    )
    success, output = ssh_to_host(host, mkdir_cmd, port=port)
    if success:
        ok("C:\\temp ready")
    else:
        warn(f"Could not create C:\\temp: {output}")

    # --- 4. Generate PS1 script ---
    script_path = "C:\\temp\\map-smb-drive.ps1"
    smb_path = f"\\\\{server}\\{share}"
    startup_folder = "$env:APPDATA\\Microsoft\\Windows\\Start Menu\\Programs\\Startup"

    lines = [
        "# map-smb-drive.ps1 - Generated by mesh smb setup-client",
        f"# Server: {server}",
        f"# Share: {share}",
        f"# Drive: {drive}",
        f"# User: {user}",
        "",
        "$ErrorActionPreference = 'Continue'",
        "",
        "Write-Host '=== SMB Drive Mapping ===' -ForegroundColor Cyan",
        "",
        "# Step 1: Get credentials",
        f"Write-Host 'Enter SMB password for {user}' -ForegroundColor Yellow",
        f"$Cred = Get-Credential -UserName '{user}' -Message 'Enter SMB password'",
        "if (-not $Cred) {",
        "    Write-Host 'Cancelled.' -ForegroundColor Red",
        "    Read-Host 'Press Enter to exit'",
        "    exit 1",
        "}",
        "$Password = $Cred.GetNetworkCredential().Password",
        "",
        "# Step 2: Remove existing mapping",
        f"Write-Host 'Removing existing {drive} mapping...' -ForegroundColor Yellow",
        f"net use {drive} /delete /y 2>$null",
        "",
        "# Step 3: Map drive",
        f"Write-Host 'Mapping {drive} to {smb_path}...' -ForegroundColor Yellow",
        f"$mapResult = net use {drive} {smb_path} /user:{user} $Password /persistent:yes 2>&1",
        "if ($LASTEXITCODE -eq 0) {",
        f"    Write-Host 'SUCCESS: {drive} mapped to {smb_path}' -ForegroundColor Green",
        "} else {",
        f"    Write-Host 'FAILED to map {drive}' -ForegroundColor Red",
        "    Write-Host $mapResult -ForegroundColor Red",
        "    Read-Host 'Press Enter to exit'",
        "    exit 1",
        "}",
        "",
        "# Step 4: Enable linked connections (for elevated processes)",
        "Write-Host 'Setting EnableLinkedConnections...' -ForegroundColor Yellow",
        "$regPath = 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Policies\\System'",
        "try {",
        "    Set-ItemProperty -Path $regPath -Name 'EnableLinkedConnections' -Value 1 -Type DWord",
        "    Write-Host 'EnableLinkedConnections set' -ForegroundColor Green",
        "} catch {",
        "    Write-Host 'WARNING: Could not set EnableLinkedConnections (run as admin)'"
        " -ForegroundColor Yellow",
        "}",
        "",
        "# Step 5: Create reconnect script in Startup folder",
        "Write-Host 'Creating startup reconnect script...' -ForegroundColor Yellow",
        f'$startupPath = "{startup_folder}"',
        "$cmdPath = Join-Path $startupPath 'reconnect-smb.cmd'",
        '$cmdContent = @"',
        "@echo off",
        f"net use {drive} {smb_path} /user:{user} /persistent:yes",
        '"@',
        "try {",
        "    $cmdContent | Set-Content -Path $cmdPath -Force",
        '    Write-Host "Startup script created: $cmdPath" -ForegroundColor Green',
        "} catch {",
        '    Write-Host "WARNING: Could not create startup script: $_" -ForegroundColor Yellow',
        "}",
        "",
        "Write-Host ''",
        f"Write-Host '{drive} is now mapped to {smb_path}' -ForegroundColor Cyan",
        "Write-Host 'Note: Log off and back on for EnableLinkedConnections to take effect'"
        " -ForegroundColor Yellow",
        "Write-Host ''",
        "Read-Host 'Press Enter to exit'",
    ]

    # --- 5. Write script via SSH + stdin pipe ---
    info("Writing PowerShell script to Windows...")
    script_content = "\n".join(lines)
    ps_write_cmd = f"powershell -Command \"$input | Set-Content -Path '{script_path}'\""

    try:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=30",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-p",
                str(port),
                host,
                ps_write_cmd,
            ],
            input=script_content,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            ok(f"Script written to {script_path}")
        else:
            error(f"Failed to write script: {result.stdout + result.stderr}")
            raise typer.Exit(1)
    except subprocess.TimeoutExpired:
        error("Timed out writing script to Windows")
        raise typer.Exit(1) from None

    # --- 6. Print instructions ---
    section("SMB Client Setup Complete")
    info("Run the following on the Windows desktop (as Administrator):")
    info(f"  powershell -ExecutionPolicy Bypass -File {script_path}")
    info("")
    info("The script will:")
    info(f"  1. Prompt for the SMB password for '{user}'")
    info(f"  2. Map {drive} to {smb_path}")
    info("  3. Set EnableLinkedConnections for elevated process access")
    info("  4. Create a startup script for automatic reconnection")


@app.command()
def status() -> None:
    \"\"\"Check SMB/Samba status on this host.\"\"\"
    section("SMB Status")

    os_type = detect_os_type()
    info(f"OS type: {os_type.value}")

    if os_type not in (OSType.UBUNTU, OSType.WSL2):
        warn("SMB status is only supported on Linux")
        info("For Windows, check drive mappings with: net use")
        return

    # Check if Samba is installed
    info("Checking Samba installation...")
    if command_exists("smbd"):
        ok("Samba installed")
    else:
        warn("Samba not installed")
        info("Install with: mesh smb setup-server")
        return

    # Check service status
    info("Checking services...")
    for service in ("smbd", "nmbd"):
        if _systemctl_is_active(service):
            ok(f"{service} is active")
        else:
            warn(f"{service} is not active")
            info(f"Start with: sudo systemctl start {service}")

    # List shares via testparm
    info("Checking shares...")
    try:
        result = subprocess.run(
            ["testparm", "-s"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            shares = []
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    name = stripped[1:-1]
                    if name not in ("global", "printers", "print$"):
                        shares.append(name)
            if shares:
                ok(f"Shares: {', '.join(shares)}")
            else:
                warn("No user shares configured")
                info("Add a share with: mesh smb setup-server")
        else:
            warn(f"testparm failed: {result.stderr.strip()}")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        warn("Could not run testparm")

    # Check port 445
    info("Checking port 445...")
    try:
        result = subprocess.run(
            ["ss", "-tlnp"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and ":445 " in result.stdout:
            ok("Port 445 is listening")
        else:
            warn("Port 445 is not listening")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        warn("Could not check port 445")

    # Check Tailscale for mesh accessibility
    if command_exists("tailscale"):
        if _is_tailscale_connected():
            ok("Tailscale connected - share accessible via mesh IPs")
        else:
            warn("Tailscale not connected - share only accessible on LAN")
```

=== END OF FILE ===

A few meta constraints for this review that I want to bake in up front, because they will shape which "fixes" are realistic when we get to the next turn:

- This CLI ships as a single pip-installable package and runs on one host at a time. You will not get to introduce a database, a message queue, or a background worker. Filesystem and subprocess are the only side-effect primitives on the table.
- The `/etc/samba/smb.conf` file is managed by the system; our tool appends to it but does not own it. Any refactor must not break existing share configurations that were added manually or by other tools.
- The `mesh.utils.process` module provides `run_sudo` and `command_exists` — these are the preferred subprocess abstractions. Direct `subprocess.run` calls should only be used when `run_sudo` genuinely cannot handle the use case (e.g., `input=` for piped passwords).
- Backwards compatibility matters: the `mesh` CLI is invoked from setup scripts and documentation. Command names, option names, exit codes, and stdout framing via `ok`/`info`/`warn`/`error`/`section` are part of the contract.
- The target platform is NVIDIA GB10 (ARM64 + Blackwell) running Ubuntu. The code must remain ARM64-clean.
- There is no test suite for this module today. Your refactor will be the first test target.
- Production usage looks like: one operator on one GB10 machine, running `mesh smb setup-server` once during initial provisioning, `mesh smb setup-client` once per Windows workstation, and `mesh smb status` occasionally for diagnostics. Latency is not a concern; correctness, idempotency, and "does not leave the system in a half-configured state" are the concerns.

Go.
"""


# ---------------------------------------------------------------------------
# FOLLOW-UPS
# ---------------------------------------------------------------------------
#
# 8-step progression: refactor, tests, cross-file interaction fix with a
# SECOND embedded real file (process.py), migration plan, observability,
# hypothesis property tests, merge-conflict resolution, and PR description.
# ---------------------------------------------------------------------------

FOLLOW_UPS: list[str] = [
    # --- Turn 2: Refactor -----------------------------------------------------------
    (
        "Now produce the refactored version of `mesh/commands/smb.py` addressing "
        "your top 3 issues from the previous analysis. Full file rewrite, not a "
        "diff. Preserve the module's public interface so existing callers and "
        "documentation don't break \u2014 the Typer commands `setup-server`, `add-user`, "
        "`setup-client`, and `status` must keep the same option names, the same "
        "exit codes, and the same stdout framing that the CLI currently emits via "
        "`ok`, `info`, `warn`, `error`, and `section`. Keep the module-level "
        "`app = typer.Typer(...)` object intact because `mesh/cli.py` registers "
        "it via `app.add_typer(smb.app, name=\"smb\")`. Inside the function bodies "
        "you may restructure freely. Explain each non-obvious change inline as a "
        "brief comment \u2014 at most one line of justification per change, but every "
        "non-obvious edit should have one. If you introduce any new helper "
        "functions, give them a private underscore prefix and colocate them in the "
        "same module rather than creating new files. If you consolidate the "
        "inconsistent subprocess usage (direct `subprocess.run` vs `run_sudo`), "
        "explain which calls you kept as direct subprocess and why \u2014 presumably "
        "only the ones that genuinely need `input=` piping. The goal of this turn "
        "is a drop-in replacement for `mesh/commands/smb.py` that fixes the top 3 "
        "findings and nothing else \u2014 do not attempt to fix the other findings yet."
    ),

    # --- Turn 3: Tests --------------------------------------------------------------
    (
        "Write a comprehensive unit test suite for the refactored code you just "
        "produced. Cover the happy path, the edge cases, and the error paths. Use "
        "pytest. Include fixtures and parametrized cases. Every public function "
        "and every Typer command callback from the refactored module should have "
        "at least two tests \u2014 one asserting correct behavior on valid input, one "
        "asserting correct behavior on an adversarial or edge-case input. For the "
        "`setup_server` command specifically, I want to see tests for: "
        "(a) running on a non-Ubuntu OS type (should exit 1), "
        "(b) running when Samba is already installed (should skip installation), "
        "(c) running when the share already exists in testparm (should skip "
        "smb.conf append), "
        "(d) a failure mid-way through the 10-step sequence (e.g., smbpasswd "
        "fails after smb.conf was already updated), "
        "(e) the idempotency claim \u2014 running setup_server twice should not "
        "duplicate the share in smb.conf. "
        "For `setup_client`, test that the generated PowerShell script contains "
        "the expected drive letter, server, share, and user values. Test what "
        "happens when SSH connectivity fails on the first probe. "
        "For `_set_smb_password`, test that passwords containing newlines, "
        "backslashes, and shell metacharacters are piped correctly. "
        "For filesystem and subprocess state use `tmp_path` and monkeypatch to "
        "replace `subprocess.run` and `run_sudo` with spies that record the "
        "argv and return canned `CommandResult` objects. Do not call real "
        "system commands from any test. Show me the full test file, placed under "
        "`mesh/tests/commands/test_smb.py` \u2014 include the pytest fixtures, the "
        "parametrize decorators, and the conftest changes if any are needed."
    ),

    # --- Turn 4: Cross-file interaction with process.py ----------------------------
    (
        "Here is a related module that is imported by `smb.py` at line 12: "
        "`from mesh.utils.process import command_exists, run_sudo`. This is the "
        "subprocess helper layer that smb.py depends on for most of its system "
        "interactions. Walk through the interaction between these two modules "
        "carefully. Identify two interface mismatches or coupling problems that "
        "your refactor either introduced or that already existed and should be "
        "fixed as part of this cleanup. Show the fixes by producing both files "
        "in their post-fix state.\n\n"
        "Specifically, think about:\n"
        "- `run_sudo` does not support `input=` for piping data to stdin, which "
        "is why `_set_smb_password` and the smb.conf append both bypass it and "
        "call `subprocess.run` directly. Is that the right boundary, or should "
        "`run_sudo` be extended to support `input=`? What are the implications "
        "for every other caller of `run_sudo` across the mesh codebase?\n"
        "- `run_sudo` merges `env` with `os.environ.copy()`, but the direct "
        "`subprocess.run` calls in smb.py do NOT set `env` at all \u2014 they inherit "
        "the parent environment implicitly. Is there a scenario where these two "
        "approaches diverge (e.g., if the parent env has been modified by an "
        "earlier step in setup_server)?\n"
        "- `CommandResult.returncode` is `-1` for both timeout and "
        "file-not-found, making them indistinguishable. The smb.py code that "
        "uses direct subprocess catches `TimeoutExpired` and `FileNotFoundError` "
        "separately. If we migrate those calls to `run_sudo`, we lose the "
        "ability to distinguish these failure modes. Is that acceptable?\n"
        "- `command_exists` uses `shutil.which` which checks PATH. But smb.py "
        "also calls tools via `sudo` \u2014 and sudo's secure_path may differ from "
        "the user's PATH. A tool might exist in the user's PATH but not in "
        "sudo's PATH, or vice versa. Does this matter for any of the tools "
        "smb.py checks (`smbd`, `testparm`, `ufw`, `tailscale`, `ss`)?\n\n"
        "Here is `mesh/utils/process.py` verbatim:\n\n"
        "```python\n"
        "\"\"\"Subprocess execution helpers.\"\"\"\n\n"
        "import shutil\n"
        "import subprocess\n"
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\n"
        "class CommandResult:\n"
        "    \"\"\"Result of a command execution.\"\"\"\n\n"
        "    returncode: int\n"
        "    stdout: str\n"
        "    stderr: str\n\n"
        "    @property\n"
        "    def success(self) -> bool:\n"
        "        \"\"\"Check if command succeeded.\"\"\"\n"
        "        return self.returncode == 0\n\n\n"
        "def run(\n"
        "    cmd: list[str],\n"
        "    *,\n"
        "    check: bool = False,\n"
        "    capture: bool = True,\n"
        "    timeout: int | None = None,\n"
        "    env: dict[str, str] | None = None,\n"
        ") -> CommandResult:\n"
        "    \"\"\"Run a command and return the result.\"\"\"\n"
        "    import os\n\n"
        "    # Merge provided env with current environment\n"
        "    run_env = None\n"
        "    if env:\n"
        "        run_env = os.environ.copy()\n"
        "        run_env.update(env)\n\n"
        "    try:\n"
        "        result = subprocess.run(\n"
        "            cmd,\n"
        "            capture_output=capture,\n"
        "            text=True,\n"
        "            timeout=timeout,\n"
        "            env=run_env,\n"
        "        )\n"
        "        return CommandResult(\n"
        "            returncode=result.returncode,\n"
        "            stdout=result.stdout if capture else \"\",\n"
        "            stderr=result.stderr if capture else \"\",\n"
        "        )\n"
        "    except subprocess.TimeoutExpired:\n"
        "        return CommandResult(returncode=-1, stdout=\"\", stderr=\"Command timed out\")\n"
        "    except FileNotFoundError:\n"
        "        return CommandResult(returncode=-1, stdout=\"\", stderr=f\"Command not found: {cmd[0]}\")\n\n\n"
        "def run_sudo(cmd: list[str], **kwargs) -> CommandResult:\n"
        "    \"\"\"Run a command with sudo.\"\"\"\n"
        "    return run([\"sudo\"] + cmd, **kwargs)\n\n\n"
        "def command_exists(cmd: str) -> bool:\n"
        "    \"\"\"Check if a command exists in PATH.\"\"\"\n"
        "    return shutil.which(cmd) is not None\n\n\n"
        "def require_command(cmd: str) -> None:\n"
        "    \"\"\"Raise error if command doesn't exist.\"\"\"\n"
        "    if not command_exists(cmd):\n"
        "        raise RuntimeError(f\"Required command not found: {cmd}\")\n"
        "```\n\n"
        "Now produce both files in their post-fix state. I want `commands/smb.py` "
        "and `utils/process.py` fully rewritten with the interface improvements "
        "applied. Name the two coupling problems explicitly in a short paragraph "
        "between the two file listings so a reviewer can verify that each file's "
        "change targets a specific problem. Do not touch any other file in this turn."
    ),

    # --- Turn 5: Migration plan ----------------------------------------------------
    (
        "Write a migration plan: how do we roll this out across the existing "
        "deployments without breaking anything? Cover rollback, canary strategy, "
        "and the exact git workflow. Reference the specific function names from "
        "your refactor.\n\n"
        "The deployment topology is: one GB10 production host (`sfspark1`) that "
        "runs the mesh CLI for Samba server management, and two developer "
        "workstations that run `mesh smb setup-client` to map drives. We ship "
        "as a uv-installed package pinned to a git SHA in the consuming repo's "
        "`pyproject.toml`. We do not publish to a private PyPI; bumps happen by "
        "updating the git SHA and re-running `uv sync`.\n\n"
        "The plan should cover: "
        "(1) the exact sequence of commits and branches you would create, "
        "(2) which hosts need to be updated in lockstep and which can lag, "
        "(3) how we smoke-test the refactor on a developer workstation before "
        "the production host ever sees the change \u2014 including what `mesh smb "
        "status` output to check, "
        "(4) what the rollback procedure is if the refactor breaks `mesh smb "
        "setup-server` on prod \u2014 including the exact git commands to pin back "
        "to the previous SHA, "
        "(5) any changes to `/etc/samba/smb.conf` handling (new backup strategy, "
        "changed append logic) and how existing smb.conf files remain valid, "
        "(6) a deprecation window for any removed or renamed private helpers, "
        "(7) how we verify that the refactored `setup_server` produces the same "
        "smb.conf output and the same service state as the old code \u2014 i.e., a "
        "golden-file check or integration test. "
        "Put the whole plan in a single markdown document with sections. Include "
        "go/no-go criteria for each stage."
    ),

    # --- Turn 6: Observability ------------------------------------------------------
    (
        "Add observability (structured logging, timing, error classification) to "
        "BOTH refactored files \u2014 `mesh/commands/smb.py` and `mesh/utils/process.py`. "
        "Use Python's `logging` module with structured format. Show the full "
        "updated source of both files with observability integrated. Describe "
        "the three SLOs and alerts you would set up on top of these signals.\n\n"
        "A few constraints for the observability pass: "
        "The CLI is short-lived: `mesh smb setup-server` runs once during "
        "provisioning, so push-style metrics (a text file collector or a JSON "
        "log that a log shipper can ingest) make more sense than a long-lived "
        "HTTP exposition endpoint. Pick one, justify it, and wire it up \u2014 do "
        "not leave this as 'in a real system we would...'. "
        "The `logging` configuration should respect `LOG_LEVEL` from the "
        "environment, default to INFO, and use a structured format (JSON or "
        "logfmt, your call) so logs can be ingested downstream. "
        "For each subprocess call (both via `run_sudo` and direct), log the "
        "command, the exit code, the wall-clock duration, and classify the "
        "outcome as success/failure/timeout. "
        "The three SLOs should be concrete \u2014 names, metrics, thresholds, alert "
        "conditions. Example: 'setup_server_duration_seconds p95 < 120s measured "
        "over rolling 30-day window; alert if 3 consecutive runs exceed 180s'. "
        "Do not write 'high reliability' \u2014 write the number. "
        "Do not break the existing stdout framing from Rich (`ok`/`info`/`warn`/"
        "`error`/`section`) \u2014 the user-facing output is still the primary channel "
        "and logs are secondary. The two output channels should not duplicate "
        "each other verbatim; pick a convention and stick to it.\n\n"
        "Deliverable for this turn: full source of both files with observability "
        "integrated, followed by the three SLOs/alerts as a short bulleted list "
        "with metric names that match what you actually emit."
    ),

    # --- Turn 7: Hypothesis property tests ------------------------------------------
    (
        "Write property-based tests using `hypothesis` that exercise the invariants "
        "your refactor preserves. Make the invariants explicit \u2014 state them in "
        "English at the top of the test file \u2014 and then test them. At least four "
        "properties, each with a meaningful strategy.\n\n"
        "Some invariants that are candidates for this refactor: "
        "(A) For any valid share name and path, running `setup_server` with the "
        "same parameters twice produces the same smb.conf content \u2014 the append "
        "is truly idempotent thanks to the `_share_exists_in_testparm` guard. "
        "Model this by mocking testparm output and verifying the code path. "
        "(B) For any password string (including those containing newlines, null "
        "bytes, backslashes, shell metacharacters, and Unicode), the "
        "`_set_smb_password` function passes the password to smbpasswd via stdin "
        "without corruption \u2014 the `input=` argument to subprocess.run is not "
        "subject to shell expansion. Verify by capturing the `input` kwarg. "
        "(C) The generated PowerShell script in `setup_client` is syntactically "
        "valid for any combination of server hostname, share name, drive letter, "
        "and username that passes basic validation \u2014 no unescaped special "
        "characters break the script structure. "
        "(D) For any sequence of `_write_systemd_dropin` calls with the same "
        "service name, the drop-in file is created exactly once and subsequent "
        "calls are no-ops (the `Path.exists()` guard works). "
        "(E) `command_exists` and `run_sudo` agree on tool availability \u2014 if "
        "`command_exists('smbd')` returns True, then `run_sudo(['smbd', '--version'])` "
        "does not fail with 'command not found' (modulo sudo secure_path "
        "differences, which should be documented as a known limitation).\n\n"
        "Pick at least four of these, or substitute equivalents you think are "
        "stronger. Write strategies that actually exercise the edge cases: "
        "hypothesis's `text()` with a custom alphabet for passwords, `from_regex` "
        "for share names, and `sampled_from` for OS types. Use `@given(...)` on "
        "every property, not `@example(...)` alone. Use "
        "`@settings(max_examples=200, deadline=None)` at least on the slow ones. "
        "Put the full test file in `mesh/tests/commands/test_smb_properties.py` "
        "and show the complete source."
    ),

    # --- Turn 8: Merge-conflict resolution ------------------------------------------
    (
        "Imagine a pre-existing open PR introduces a conflicting change to the "
        "same module. Here is the plausible conflict: someone else opened PR #87 "
        "titled 'smb: add `mount` and `unmount` subcommands for Linux CIFS "
        "client-side mounting'. Their PR adds two new Typer subcommands to "
        "`mesh/commands/smb.py`: `mount` (which runs `sudo mount -t cifs` to "
        "mount a remote SMB share on the local Linux filesystem) and `unmount` "
        "(which runs `sudo umount`). Their PR also adds a `_find_mount_point` "
        "helper that parses `/proc/mounts` and a `_validate_cifs_utils` helper "
        "that checks for the `mount.cifs` binary. Their PR was opened two weeks "
        "before yours, has been reviewed, and is queued to merge tomorrow. Your "
        "refactor rewrites the subprocess handling, the error handling pattern, "
        "and several private helper functions inside `commands/smb.py`.\n\n"
        "Draft a resolution: which parts of your refactor you keep, which you "
        "cede, how you rebase. Show the merge strategy in git commands and "
        "highlight the semantic conflicts beyond what git can auto-resolve.\n\n"
        "Your answer should cover: "
        "(1) which side should merge first \u2014 yours or PR #87 \u2014 and why. "
        "(2) the exact `git rebase` / `git merge` commands the losing side has "
        "to run to pick up the winning side. "
        "(3) the syntactic conflicts git will flag and how you would resolve each "
        "(give me the before/after lines). "
        "(4) the semantic conflicts git CANNOT detect. Example candidates: "
        "PR #87's `mount` command calls `subprocess.run` directly \u2014 does your "
        "refactored subprocess pattern apply to their new code? PR #87's "
        "`_validate_cifs_utils` calls `command_exists` \u2014 did you change how "
        "`command_exists` interacts with sudo's PATH? If yes, their validation "
        "check might give false positives. "
        "(5) a short post-merge sanity checklist \u2014 not an exhaustive test run, "
        "just the three things a human reviewer should manually verify after "
        "applying the merge. "
        "(6) whether you would push back on PR #87 and ask them to rebase on "
        "your refactor first, or whether you would absorb the conflict yourself. "
        "Justify the choice in two sentences."
    ),

    # --- Turn 9: PR description -----------------------------------------------------
    (
        "Produce the complete PR description for your refactor: title, summary, "
        "motivation, design decisions, test plan, rollout plan, risk assessment "
        "table (with likelihood and blast radius for each risk), and a reviewer "
        "checklist. This should be the kind of PR description a staff engineer "
        "would write for a major cleanup \u2014 polished, specific, and reviewable.\n\n"
        "Format requirements: "
        "Title under 72 characters, imperative mood, no trailing period, starts "
        "with the conventional-commits-style scope prefix `smb:` since the "
        "refactor is scoped to `mesh/commands/smb.py` plus the "
        "`mesh/utils/process.py` interface improvement. "
        "Summary is exactly three bullets. "
        "Motivation is a short paragraph (3-5 sentences) explaining why this "
        "refactor is worth merging now and not next quarter \u2014 reference the "
        "concrete bugs and security issues from earlier turns. "
        "Design decisions section enumerates at least four non-obvious choices "
        "you made, each with a one-sentence 'why this instead of the obvious "
        "alternative'. "
        "Test plan is a checklist of the test files added or touched plus any "
        "manual smoke tests the reviewer should run on a GB10 host \u2014 including "
        "running `mesh smb setup-server` and `mesh smb status` and verifying "
        "the output matches expectations. "
        "Rollout plan is the summarized form of the migration plan from the "
        "earlier turn, not a re-derivation \u2014 three bullets, not three paragraphs. "
        "Risk table has columns: risk, likelihood (low/med/high), blast radius "
        "(one host / all hosts / all hosts + data loss), mitigation. At least "
        "five rows. Order by likelihood * blast radius, descending. "
        "Reviewer checklist is an explicit `- [ ]` markdown list of at least six "
        "things the reviewer must verify \u2014 not 'code looks good', but specific "
        "things like 'smb.conf append is still guarded by testparm check' or "
        "'_set_smb_password still uses input= not shell piping'.\n\n"
        "Output the whole PR description in one markdown block. No preamble, no "
        "postamble \u2014 just the PR body the way it would appear in the GitHub "
        "compose box. Assume the reviewer has not seen any of the previous turns "
        "in this thread; the PR description must stand on its own."
    ),
]


# ---------------------------------------------------------------------------
# estimate_tokens() helper — uses len(text) // 4 (character-based)
# ---------------------------------------------------------------------------

def estimate_tokens() -> None:
    """Print approximate token counts using len(text) // 4."""
    def _tok(text: str) -> int:
        return len(text) // 4

    print(f"SEED_TOPIC            : {_tok(SEED_TOPIC):>6} tokens")
    print(f"SEED_PROMPT           : {_tok(SEED_PROMPT):>6} tokens")
    print(f"FOLLOW_UPS (count)    : {len(FOLLOW_UPS):>6}")
    total_followup = 0
    for i, fu in enumerate(FOLLOW_UPS, start=1):
        t = _tok(fu)
        total_followup += t
        print(f"  follow_up[{i}]         : {t:>6} tokens")
    print(f"FOLLOW_UPS total      : {total_followup:>6} tokens")
    print(f"SESSION total         : {_tok(SEED_PROMPT) + total_followup:>6} tokens")


if __name__ == "__main__":
    estimate_tokens()
