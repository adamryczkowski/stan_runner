from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

import fabric


class OSType(Enum):
    UBUNTU = 1
    CENTOS = 2
    REDHAT = 3
    FEDORA = 4
    DEBIAN = 5
    SUSE = 6
    OTHER = 7


class SystemInfo:
    _os: OSType

    @staticmethod
    def DetectOS(ssh: fabric.Connection) -> SystemInfo:
        try:
            out = ssh.run("cat /etc/os-release", hide=True)
            if "ubuntu" in out.stdout.lower():
                return SystemInfo(OSType.UBUNTU)
            elif "centos" in out.stdout.lower():
                return SystemInfo(OSType.CENTOS)
            elif "redhat" in out.stdout.lower():
                return SystemInfo(OSType.REDHAT)
            elif "fedora" in out.stdout.lower():
                return SystemInfo(OSType.FEDORA)
            elif "debian" in out.stdout.lower():
                return SystemInfo(OSType.DEBIAN)
            elif "suse" in out.stdout.lower():
                return SystemInfo(OSType.SUSE)
            else:
                return SystemInfo(OSType.OTHER)
        except Exception as e:
            raise RuntimeError(f"Failed to detect the OS on the remote server: {e}")

    def __init__(self, os: OSType):
        self._os = os

    @property
    def os(self) -> OSType:
        return self._os


def command_succeed(ssh: fabric.Connection, cmd: str) -> bool:
    try:
        ssh.run(cmd, warn=True, hide=True)
        return True
    except Exception as e:
        return False


def install_r(system: SystemInfo, ssh: fabric.Connection):
    packages = []
    if system.os == OSType.UBUNTU:
        if not command_succeed(ssh, "Rscript --version"):
            packages.append("r-base")
        if not command_succeed(ssh, "dpkg -s libcurl4-openssl-dev"):
            packages.append("libcurl4-openssl-dev")
        if not command_succeed(ssh, "dpkg -s libsodium-dev"):
            packages.append("libsodium-dev")
        if not command_succeed(ssh, "dpkg -s build-essential"):
            packages.append("build-essential")
        if len(packages) > 0:
            print(f"Missing dependencies: {', '.join(packages)}. Trying to install them (if root is provided)...")
            try:
                ssh.sudo(f"apt-get install  --yes {' '.join(packages)}")
            except Exception as e:
                raise RuntimeError(f"Failed to install R on the remote server: {e}")
    else:
        raise RuntimeError(f"Unsupported OS: {system.os}")


def ensure_dependencies(ssh_address: str, ssh_port: int = 22, ssh_user: str = None,
                        driver_script: Path = None, runner_script: Path = None,
                        installer_script:Path=None)->fabric.Connection:
    if ssh_user is None:
        ssh_user = os.getlogin()

    try:
        ssh = fabric.Connection(ssh_address, port=ssh_port, user=ssh_user)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to the remote server: {e}")

    system = SystemInfo.DetectOS(ssh)
    install_r(system, ssh)

    if driver_script is None:
        driver_script = Path(__file__).parent.parent / "R/rest_server.R"
    if runner_script is None:
        runner_script = Path(__file__).parent.parent / "R/rest_runner.R"
    if installer_script is None:
        installer_script = Path(__file__).parent.parent / "R/rest_installer.R"
    assert driver_script.exists()
    assert runner_script.exists()
    assert installer_script.exists()

    # Upload the driver and runner scripts
    try:
        ssh.put(str(driver_script), str(driver_script.name))
        ssh.put(str(runner_script), str(runner_script.name))
        ssh.put(str(installer_script), str(installer_script.name))
    except Exception as e:
        raise RuntimeError(
            f"Failed to upload the scripts to the remote server: {e}"
        )

    try:
        ans = ssh.run(
            f"Rscript {str(driver_script.name)}",
            hide=False,
        )
        if ans.return_code != 0:
            raise RuntimeError(f"The driver script failed: {ans}")
    except Exception as e:
        raise RuntimeError(f"Failed to run the driver script on the remote server: {e}")


    return ssh
