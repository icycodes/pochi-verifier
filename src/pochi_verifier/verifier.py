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

        Args:
            pochi_path (str): The path to the pochi executable. 
                                Defaults to "pochi", assuming it's in the system's PATH.
            ffmpeg_path (Optional[str]): The path to the ffmpeg executable.
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

        command = [
            self.pochi_path,
            "--model", model,
            "--attempt-completion-schema", schema,
            "--stream-json", stream_json_path,
            "--blobs-dir", blobs_dir_path,
            "--prompt", prompt,
        ]
        
        if self.ffmpeg_path:
            command.extend(["--ffmpeg", self.ffmpeg_path])

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8'
            )
            with open(stdout_file_path, "w", encoding='utf-8') as f:
                f.write(result.stdout)
            if result.stderr:
                with open(stderr_file_path, "w", encoding='utf-8') as f:
                    f.write(result.stderr)
            
            return self._parse_result(result.stdout, result.stderr)

        except FileNotFoundError:
            raise FileNotFoundError(
                f"The 'pochi' executable was not found at '{self.pochi_path}'. "
                "Please ensure that the pochi CLI is installed and that its location "
                "is in your system's PATH, or provide the correct path to it."
            )
        except subprocess.CalledProcessError as e:
            with open(stdout_file_path, "w", encoding='utf-8') as f:
                f.write(e.stdout)
            with open(stderr_file_path, "w", encoding='utf-8') as f:
                f.write(e.stderr)
            raise subprocess.CalledProcessError(
                e.returncode, e.cmd, output=e.output, stderr=e.stderr
            ) from e

    def _parse_result(self, stdout: str, stderr: str) -> VerificationResult:
        """Parses the stdout from pochi to extract the verification result."""
        try:
            # Find the JSON object that follows "Task Completed"
            match = re.search(r"Task Completed.*?(\{.*)", stdout, re.DOTALL)
            if not match:
                raise PochiOutputError("Could not find 'Task Completed' marker in the output.")
            
            json_str = match.group(1)
            parsed_json = json.loads(json_str)

            status = parsed_json.get("status")
            reason = parsed_json.get("reason")

            if not status or not reason:
                raise PochiOutputError("The result JSON is missing 'status' or 'reason'.")

            if status == "fail":
                raise VerificationFailedError(f"Verification failed: {reason}")

            return VerificationResult(status=status, reason=reason, stdout=stdout, stderr=stderr)

        except json.JSONDecodeError as e:
            raise PochiOutputError(f"Failed to parse JSON from pochi output: {e}") from e

    def _create_prompt(self, reason: str, truth: str) -> str:
        """Creates the prompt for the pochi CLI for browser verification."""
        
        return f"""You are a software tester. Your task is to use the browser agent to verify the following test case.

Reason for this test:
{reason}

Verification Steps:
{truth}

Important: If the target URL in the verification steps cannot be opened, you must immediately return a "fail" status.

Complete this task and provide the result in the following JSON format:
{{
  "status": "pass" | "fail",
  "reason": "A detailed explanation of why the test passed or failed."
}}
"""
