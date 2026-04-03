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

class VerificationFailedError(Exception):
    """Custom exception for when a verification fails."""
    pass

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
        self._agent_file_path = self._create_agent_file()

    def __del__(self):
        """Cleans up the agent file."""
        if os.path.exists(self._agent_file_path):
            os.remove(self._agent_file_path)

    def _create_agent_file(self) -> str:
        """Creates the agent file with critical instructions."""
        agent_dir = os.path.join(os.getcwd(), ".pochi", "agents")
        os.makedirs(agent_dir, exist_ok=True)
        agent_file_path = os.path.join(agent_dir, "pochi_verifier.md")

        instructions = """---
name: pochi_verifier
description: A software tester agent for browser-based verification.
tools: [newTask, attemptCompletion]
---
## Role
You are a software tester responsible for verifying test cases using a browser-based agent.

## Instructions
1.  You **must** use the `newTask` tool to create a `browser` sub-agent to perform all verification steps.
2.  The `prompt` for the `newTask` tool should contain the "Reason for this test" and the "Verification Steps".
3.  The `prompt` for the `newTask` tool **must** contain all the `Rules` below.
4.  After the sub-agent completes its task, you **must** call the `attemptCompletion` tool with the final status and a detailed reason.

## Rules
- **Do NOT** attempt to execute any system commands, scripts, or actions that start, open, terminate, or kill any process or service. This includes, but is not limited to, starting servers, opening ports, or killing background jobs.
- **Do NOT** attempt to access, open, or modify network ports through any means, including indirect methods such as running programs that bind to ports.
- If the verification steps reference a URL or port that cannot be accessed via the agent browser, you must stop and immediately return a `"fail"` status. For example, this can happen if the agent-browser returns `net::ERR_CONNECTION_REFUSED`, which could indicate that a required server is not running or a previous command has caused it to crash.
- If you encounter any verification step that you cannot execute, or if you are unsure how to proceed at any point, you must stop and immediately return a `"fail"` status. For instance, if you encounter a login page but the instructions do not include a login step, you should stop and return a `"fail"` status.

## Result Format
Call the `attemptCompletion` tool with a JSON object matching this schema:
```json
{
    "status": "pass" | "fail",
    "reason": "A detailed explanation describing why the test passed or failed."
}
```
"""
        with open(agent_file_path, "w", encoding="utf-8") as f:
            f.write(instructions)
        return agent_file_path

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
        model: str = "google/gemini-3-flash",
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

        Raises:
            VerificationFailedError: If the verification fails.
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
            f"--agent pochi_verifier "
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
        # Find the JSON object that follows the `Task Completed` marker (with possible whitespace)
        match = re.search(r"Task Completed.*?(\{.*)", stdout, re.DOTALL)
        if not match:
            raise PochiOutputError("Cannot extract the JSON from the output message.")
        json_str = match.group(1)
        # Extract the full JSON object, accounting for nested/escaped braces
        brace_count = 0
        in_string = False
        escape = False
        for i, c in enumerate(json_str):
            if escape:
                escape = False
                continue
            if c == '\\':
                escape = True
                continue
            if c == '"':
                in_string = not in_string
            if not in_string:
                if c == '{':
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = json_str[:i+1]
                        break
        try:
            parsed_json = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise PochiOutputError(f"Failed to parse JSON from pochi output: {e}") from e

        status = parsed_json.get("status")
        reason = parsed_json.get("reason")

        if not status or not reason:
            raise PochiOutputError("The result JSON is missing 'status' or 'reason'.")

        if status == "fail":
            raise VerificationFailedError(f"Verification failed: {reason}")

        return VerificationResult(status=status, reason=reason, stdout=stdout, stderr=stderr)

    def _create_prompt(self, reason: str, truth: str) -> str:
        """Creates the prompt for the pochi CLI for browser verification."""
        
        return f"""
## Reason for this test
{reason}

## Verification Steps
Follow these steps precisely and in order:
{truth}
"""
