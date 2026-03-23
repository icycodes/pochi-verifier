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

    def __init__(self, pochi_path: str = "pochi"):
        """
        Initializes the PochiVerifier.

        Args:
            pochi_path (str): The path to the pochi executable. 
                                Defaults to "pochi", assuming it's in the system's PATH.
        """
        if not shutil.which(pochi_path):
            raise FileNotFoundError(
                f"The 'pochi' executable was not found at '{pochi_path}'. "
                "Please ensure that the pochi CLI is installed and that its location "
                "is in your system's PATH, or provide the correct path to it."
            )
        self.pochi_path = pochi_path

    def verify(
        self, 
        spec: Dict[str, Any], 
        model: str = "google/gemini-3-flash",
        trajectory_dir: Optional[str] = None,
    ) -> VerificationResult:
        """
        Runs the pochi CLI's verify command with the given specification.

        Args:
            spec (Dict[str, Any]): A dictionary containing the verification specification.
                                 It should contain the following keys:
                                 - 'name' (str): The name of the test case.
                                 - 'type' (str): The type of verification. Currently only
                                   'browser_verification' is supported.
                                 - 'reason' (str): The reason for this verification.
                                 - 'verification' (str): The steps to perform for the verification.
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
        if "type" not in spec or spec["type"] != "browser_verification":
            raise ValueError("Currently, only 'browser_verification' is supported for the 'type' field.")

        if "name" not in spec or not spec["name"]:
            raise ValueError("The 'name' field is missing or empty in the spec.")
        
        if "reason" not in spec or not spec["reason"]:
            raise ValueError("The 'reason' field is missing or empty in the spec.")

        if "verification" not in spec or not spec["verification"]:
            raise ValueError("The 'verification' field is missing or empty in the spec.")

        prompt = self._create_prompt(spec)
        
        if trajectory_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            pid = os.getpid()
            safe_name = "".join(c if c.isalnum() else "_" for c in spec["name"])
            trajectory_dir = os.path.join(os.getcwd(), "pochi-verifier", "trajectories", f"{safe_name}-{timestamp}-{pid}")
        
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

    def _create_prompt(self, spec: Dict[str, Any]) -> str:
        """Creates the prompt for the pochi CLI for browser verification."""
        reason = spec.get("reason", "No reason provided.")
        verification = spec.get("verification")
        
        return f"""You are a software tester. Your task is to use the browser agent to verify the following test case.

Reason for this test:
{reason}

Verification Steps:
{verification}

Important: If the target URL in the verification steps cannot be opened, you must immediately return a "fail" status.

Complete this task and provide the result in the following JSON format:
{{
  "status": "pass" | "fail",
  "reason": "A detailed explanation of why the test passed or failed."
}}
"""
