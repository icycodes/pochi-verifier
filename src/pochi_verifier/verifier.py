import subprocess
import shutil
import json
import os
import re
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class VerificationResult:
    """
    Represents the result of a verification.
    """
    status: str
    reason: str
    stdout: str
    stderr: str

class PochiOutputError(Exception):
    """Custom exception for when the output from pochi is malformed."""
    pass

class PochiVerifier:
    """
    A Python wrapper for the pochi CLI.
    """

    def __init__(self, pochi_path: str = "pochi", ffmpeg_path: Optional[str] = None):
        """
        Initializes the PochiVerifier.
        """
        if not shutil.which(pochi_path):
            raise FileNotFoundError(
                f"The 'pochi' executable was not found at '{pochi_path}'. "
                "Please ensure that the pochi CLI is installed and that its location "
                "is in your system's PATH, or provide the correct path to it."
            )
        self.pochi_path = pochi_path
        self.ffmpeg_path = ffmpeg_path
        if self.ffmpeg_path:
            if not shutil.which(self.ffmpeg_path):
                print("Warning: ffmpeg not found at the specified path. Video recording will not be available.")
        elif not shutil.which("ffmpeg"):
            print("Warning: ffmpeg not found in PATH. Video recording will not be available.")

        self._has_browser_agent = self._check_browser_agent()
        if not self._has_browser_agent:
            print("Warning: 'agent-browser' not found. Browser verification will not be available.")

    def _check_browser_agent(self) -> bool:
        """Checks if the browser agent is available."""
        try:
            subprocess.run(
                [self.pochi_path, "agent-browser", "--version"],
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8'
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def verify(
        self,
        reason: str,
        truth: str,
        use_browser_agent: bool = True,
        model: str = "google/gemini-3.5-flash",
        trajectory_dir: Optional[str] = None,
    ) -> VerificationResult:
        """
        Runs the pochi CLI's verify command with the given specification.

        Args:
            reason (str): The reason for this verification.
            truth (str): The steps to perform for the verification.
            use_browser_agent (bool): Whether to use the browser agent for verification.
                Only use_browser_agent = True is supported for now.
            model (str): The model to use for the verification.
            trajectory_dir (Optional[str]): The directory to save trajectory files.
                                If None, a default directory will be created.

        Returns:
            VerificationResult: An object containing the verification result.
                Both passing and failing verifications are returned here.

        Raises:
            ValueError: If the specification is invalid.
            subprocess.CalledProcessError: If the pochi CLI returns a non-zero exit code.
            PochiOutputError: If the output from pochi CLI is malformed.
            FileNotFoundError: If the pochi executable is not found.
        """

        if not use_browser_agent:
            raise ValueError("Only use_browser_agent = True is supported for now.")

        if use_browser_agent and not self._has_browser_agent:
            raise ValueError("Browser agent is not available. Please install 'agent-browser' for pochi.")

        prompt = self._create_prompt(reason, truth)
        
        if trajectory_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            pid = os.getpid()
            trajectory_dir = os.path.join(os.getcwd(), "pochi-verifier", "trajectories", f"{timestamp}-{pid}")
        
        os.makedirs(trajectory_dir, exist_ok=True)

        schema = 'z.object({ "status": z.enum(["pass", "fail"]), "reason": z.string() })'
        stream_json_path = os.path.join(trajectory_dir, "trajectory.jsonl")
        blobs_dir_path = os.path.join(trajectory_dir, "blobs")
        stdout_file_path = os.path.join(trajectory_dir, "stdout.txt")
        stderr_file_path = os.path.join(trajectory_dir, "stderr.txt")

        # Compose the full shell command as a string
        shell_command = (
            f"{self.pochi_path} "
            f"--model {model} "
            f"--attempt-completion-schema '{schema}' "
            f"--experimental-stream-trajectory {stream_json_path} "
            f"--blobs-dir {blobs_dir_path} "
        )
        if self.ffmpeg_path:
            shell_command += f"--ffmpeg {self.ffmpeg_path} "
        shell_command += (
            f"> >(tee {stdout_file_path}) "
            f"2> >(tee {stderr_file_path} >&2) "
            "<<'EOF'\n"
            f"{prompt}\n"
            "EOF"
        )

        try:
            proc = subprocess.run(
                shell_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                executable="/bin/bash"
            )
            return self._parse_result(proc.stdout, proc.stderr)

        except FileNotFoundError:
            raise FileNotFoundError(
                f"The 'pochi' executable was not found at '{self.pochi_path}'. "
                "Please ensure that the pochi CLI is installed and that its location "
                "is in your system's PATH, or provide the correct path to it."
            )
        except subprocess.CalledProcessError as e:
            raise e

    def _parse_result(self, stdout: str, stderr: str) -> VerificationResult:
        """Parses the stdout from pochi to extract the verification result."""
        
        lines = stdout.splitlines()
        json_lines = []
        capturing = False
        start_line_index = -1

        # Find the start of the JSON block
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line == "🎉 Task Completed":
                if i + 1 < len(lines) and lines[i+1].strip() == '{':
                    start_line_index = i + 2
                    capturing = True
                    break
        
        if not capturing:
            raise PochiOutputError("Cannot find the start of the JSON block.")

        # Find the end of the JSON block and extract content
        end_line_index = -1
        for i in range(start_line_index, len(lines)):
            line = lines[i]
            stripped_line = line.strip()

            if stripped_line == '}':
                # This is the final closing brace on its own line
                end_line_index = i
                break
            
            json_lines.append(line)
        
        if end_line_index == -1:
            raise PochiOutputError("Cannot find the end of the JSON block.")
        
        # Reconstruct the JSON string
        # The outer braces are handled by the start/end line checks
        json_str = "{\n" + "\n".join(json_lines) + "\n}"

        try:
            parsed_json = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise PochiOutputError(f"Failed to parse JSON from pochi output: {e}") from e

        status = parsed_json.get("status")
        reason = parsed_json.get("reason")

        if not status or not reason:
            raise PochiOutputError("The result JSON is missing 'status' or 'reason'.")

        return VerificationResult(status=status, reason=reason, stdout=stdout, stderr=stderr)

    def _create_prompt(self, reason: str, truth: str) -> str:
        """Creates the prompt for the pochi CLI for browser verification."""
        
        return f"""# System Instructions
Follow these rules and steps exactly.

## Tool Rules
You **MUST ONLY** use the following tools; using any other tool is forbidden.
1. newTask: use the `newTask` tool to create a `browser` sub-agent.
2. attemptCompletion: use `attemptCompletion` to report the final result in JSON format.

## Steps to Follow
1. Create a `browser` sub-agent with exactly this prompt:
    ``````markdown
    You are a software tester assigned to verify a test case using the agent-browser.

    ## Tool Rules
    You **MUST ONLY** use the following tools; using any other tool is forbidden. If you encounter a case that requires a tool not listed here, stop and immediately return a `"fail"` status.
    1. executeCommand: use `executeCommand` to run an `agent-browser` command; running any other command is forbidden.
    2. readFile: if the command output is truncated, you may use `readFile` to read the saved result file; any other use is forbidden.
    3. attemptCompletion: use `attemptCompletion` to report the final result in JSON format.

    ## Critical Instructions
    - You **MUST** use the agent-browser to perform all verification steps.
    - If the verification steps reference a URL or port that is unreachable via the agent-browser (i.e., the agent-browser returns an access error), immediately return a `"fail"` status.
    - If you encounter any verification step that you cannot execute, or are unsure how to proceed at any point, immediately return a `"fail"` status. For example: if you see a login page but no login step exists in the instructions, stop and return `"fail"`.

    ## Forbidden Actions
    You **MUST NOT** perform any of the following actions. If you encounter a case that requires any of them, stop and immediately return a `"fail"` status.
    - **DO NOT** read any files in the workspace, including source files, scripts, or directory listings.
    - **DO NOT** execute any system commands, scripts, or actions that start, open, terminate, or kill any process or service (including but not limited to starting servers, opening ports, or killing background jobs).

    ## Reason for This Test
    {reason}

    ## Verification Steps
    Follow these steps precisely and in order:
    {truth}

    ## Result Format
    At the end of the test, respond with the following JSON:
    {{
        "status": "pass" | "fail",
        "reason": "A detailed explanation of why the test passed or failed."
    }}
    ``````
2. Report the result of the `browser` sub-agent in JSON format. If the sub-agent fails or does not return a result, report the status as `fail` and provide the reason.
    ```json
    {{
        "status": "pass" | "fail",
        "reason": "A detailed explanation of why the test passed or failed."
    }}
    ```
"""
