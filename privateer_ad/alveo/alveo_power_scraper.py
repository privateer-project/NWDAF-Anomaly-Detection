import sys
import subprocess
import datetime
import re
from typing import Dict, Any, Tuple, Optional, List

# Define the threshold version tuple for <= comparison
LEGACY_THRESHOLD_VERSION = (2, 13)


class ALVEOPowerScraper:
    def __init__(self, device_bus: str, xrt_version: str = "2.13") -> None:
        self.device_bus: Optional[str] = None
        self.shell_name: Optional[str] = None
        try:
            # Validate device_bus and get shell name
            # This method raises ValueError or RuntimeError on failure
            self.shell_name = self._validate_device_bus(device_bus)
            self.device_bus = device_bus  # Assign only after validation
            print(
                f"ALVEOPowerScraper configured for device: {self.device_bus} (Shell: {self.shell_name})."
            )
        except (ValueError, RuntimeError) as e:
            # Re-raise a consolidated error message to be caught by AlveoRunner
            raise ValueError(
                f"Failed to initialize ALVEOPowerScraper for BDF '{device_bus}': {e}"
            )

        self.device_bus = device_bus
        self.command = f"xbutil examine -d {self.device_bus} --r electrical"
        try:
            # Parse "2.13" into (2, 13) for comparison
            self.xrt_version_tuple: Tuple[int, ...] = tuple(
                map(int, xrt_version.split("."))
            )
        except ValueError:
            raise ValueError(
                f"xrt_version '{xrt_version}' is not in a valid format like '2.13' or '2.14.1'"
            )

    def get_power(self) -> Dict[str, Any]:
        try:
            if sys.version_info >= (3, 7):
                result = subprocess.run(
                    self.command,
                    check=True,
                    capture_output=True,
                    universal_newlines=True,
                    shell=True,
                )
            else:
                result = subprocess.run(
                    self.command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    shell=True,
                )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "command '{}' return with error (code {}): {}".format(
                    e.cmd, e.returncode, e.output
                )
            )
        data = self.parse_power_data(result.stdout)
        data["timestamp"] = datetime.datetime.utcnow().isoformat()
        return data

    def parse_power_data(self, output: str) -> Dict[str, Any]:
        lines = output.split("\n")
        data = {}
        power_rails = {}
        data["Max Power"] = int(lines[5].split(":")[1].strip().split()[0])
        data["Power"] = float(lines[6].split(":")[1].strip().split()[0])
        data["Power Warning"] = lines[7].split(":")[1].strip()
        # Determine parsing strategy based on XRT version
        is_legacy_format: bool = self.xrt_version_tuple <= LEGACY_THRESHOLD_VERSION
        if is_legacy_format:
            power_rail_start_index = 10  # Corresponds to `index > 9`
            value_delimiter = "  "  # Double space
        else:
            power_rail_start_index = 11  # Corresponds to `index > 10`
            value_delimiter = ","  # Comma

        for index, line in enumerate(lines):
            if index >= power_rail_start_index and line.strip() != "":
                name = line.split(":")[0].strip()
                voltage_current = line.split(":")[1].strip().split(value_delimiter)
                temp_dir = {"Voltage": float(voltage_current[0].strip().split()[0])}
                if len(voltage_current) > 1:
                    temp_dir["Current"] = float(voltage_current[1].strip().split()[0])
                power_rails[name] = temp_dir
        data["Power Rails"] = power_rails
        return data

    def _validate_device_bus(self, device_bus_to_check: str) -> str:
        """
        Validates if the given device_bus (BDF) exists by running 'xbutil examine'.
        Returns the shell name if valid.
        Raises ValueError if the device_bus BDF is malformed or not found among detected devices.
        Raises RuntimeError if 'xbutil examine' command itself fails (e.g., not found, execution error).
        """
        # BDF format check: e.g., 0000:af:00.1
        if not re.match(
            r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]$",
            device_bus_to_check,
        ):
            raise ValueError(
                f"Device BDF '{device_bus_to_check}' is not in the correct format (e.g., xxxx:xx:xx.x)."
            )

        try:
            cmd = "xbutil examine"
            process = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                shell=True,
            )
            output = process.stdout
        except subprocess.CalledProcessError as e:
            error_msg = f"Command '{e.cmd}' failed with code {e.returncode}."
            if e.stdout:
                error_msg += f"\nStdout: {e.stdout.strip()}"
            if e.stderr:
                error_msg += f"\nStderr: {e.stderr.strip()}"
            if (
                "not found" in str(e.stderr).lower()
                or "not found" in str(e.stdout).lower()
            ):
                error_msg += "\nHint: 'xbutil' command not found. Ensure XRT is installed and its environment is sourced."
            raise RuntimeError(error_msg)
        except FileNotFoundError:
            raise RuntimeError(
                "'xbutil' command not found. Ensure XRT is installed and in PATH, and its environment is sourced."
            )
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while running 'xbutil examine': {e}"
            )

        available_devices_info: List[Tuple[str, str]] = []
        # Regex to capture BDF and Shell from lines like:
        # [0000:af:00.1]  :  xilinx_u280_xdma_201920_3         0x5e278820...
        device_line_re = re.compile(r"^\[([0-9a-fA-F:.]+)\]\s*:\s*([\w.-]+)")

        in_devices_section = False
        for line in output.splitlines():
            if "Devices present" in line:
                in_devices_section = True
                continue

            if not in_devices_section:
                continue

            match = device_line_re.match(line.strip())
            if match:
                bdf, shell_name = match.group(1), match.group(2)
                available_devices_info.append((bdf, shell_name))
                if bdf == device_bus_to_check:
                    return shell_name

        available_devices_str_list = [
            f"  - BDF: {bdf}, Shell: {shell}" for bdf, shell in available_devices_info
        ]
        if not available_devices_info:
            err_msg = (
                f"Target BDF '{device_bus_to_check}' not found. "
                f"Additionally, no devices were successfully parsed from 'xbutil examine' output."
            )
        else:
            err_msg = (
                f"Target BDF '{device_bus_to_check}' not found.\n"
                f"Available devices detected by 'xbutil examine':\n"
                + "\n".join(available_devices_str_list)
            )
        raise ValueError(err_msg)