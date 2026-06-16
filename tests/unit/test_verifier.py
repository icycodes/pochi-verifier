import io
import os
import pytest
import subprocess
from unittest.mock import patch, MagicMock
from pochi_verifier.verifier import PochiVerifier, PochiOutputError

@pytest.fixture
def verifier():
    """Fixture to provide a PochiVerifier instance with a mocked pochi executable."""
    with patch('shutil.which', side_effect=lambda x: '/fake/path/to/' + x):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            return PochiVerifier()

@pytest.fixture
def valid_reason():
    """Fixture to provide a valid test reason."""
    return "A valid test reason."

@pytest.fixture
def valid_truth():
    """Fixture to provide a valid verification step."""
    return "A valid verification step."

def test_verify_success(verifier, valid_reason, valid_truth, tmp_path):
    """Test a successful verification."""
    mock_stdout = """Some output...
🎉 Task Completed
{
  "status": "pass",
  "reason": "Everything is awesome."
}
"""
    mock_stderr = ''
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = mock_stderr
    mock_process.returncode = 0
    with patch('subprocess.run', return_value=mock_process) as mock_run:
        result = verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))
        assert result.status == "pass"
        assert result.reason == "Everything is awesome."
        mock_run.assert_called_once()
        command = mock_run.call_args[0][0]
        assert "pochi" in command
        assert "--model" in command
        assert "google/gemini-3.5-flash" in command
        assert "--attempt-completion-schema" in command
        assert "--experimental-stream-trajectory" in command

def test_verify_error_without_emoji(verifier, valid_reason, valid_truth, tmp_path):
    """Test that verification fails if the 'Task Completed' marker is missing the emoji."""
    mock_stdout = """Some output...
Task Completed
{
  "status": "pass",
  "reason": "No emoji."
}
"""
    mock_stderr = ''
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = mock_stderr
    mock_process.returncode = 0
    with patch('subprocess.run', return_value=mock_process):
        with pytest.raises(PochiOutputError, match="Cannot find the start of the JSON block."):
            verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))

def test_verify_success_with_suffix_output(verifier, valid_reason, valid_truth, tmp_path):
    """Test a successful verification."""
    mock_stdout = """Some output...

🎉 Task Completed
{
  "status": "pass",
  "reason": "Test passed with suffix."
}
More output..."""
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = ''
    mock_process.returncode = 0
    with patch('subprocess.run', return_value=mock_process):
        result = verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))
        assert result.status == "pass"
        assert result.reason == "Test passed with suffix."

def test_verify_success_with_escaped_braces(verifier, valid_reason, valid_truth, tmp_path):
    """Test a successful verification with escaped braces in JSON."""
    mock_stdout = """Some output...
🎉 Task Completed
{
  "status": "pass",
  "reason": "This string contains a brace: { and another: } and \\"nested\\": {\\"a\\": 1}"
}
More output..."""
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = ''
    mock_process.returncode = 0
    with patch('subprocess.run', return_value=mock_process):
        result = verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))
        assert result.status == "pass"
        assert "This string contains a brace: { and another: }" in result.reason

def test_verify_failure_status(verifier, valid_reason, valid_truth, tmp_path):
    """Test a verification that returns a structured 'fail' result."""
    mock_stdout = """Some output...
🎉 Task Completed
{
  "status": "fail",
  "reason": "Something went wrong."
}
"""
    mock_stderr = ''
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = mock_stderr
    mock_process.returncode = 0
    with patch('subprocess.run', return_value=mock_process):
        result = verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))
        assert result.status == "fail"
        assert result.reason == "Something went wrong."
        assert result.stdout == mock_stdout
        assert result.stderr == mock_stderr


def test_verify_error_with_truncated_json(verifier, valid_reason, valid_truth, tmp_path):
    """Test handling of truncated JSON in the output."""
    mock_stdout = """Some output...
🎉 Task Completed
{
  "status": "pass",
  "reason": "Unclosed string..."""
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = ''
    mock_process.returncode = 0
    with patch('subprocess.run', return_value=mock_process):
        with pytest.raises(PochiOutputError, match="Cannot find the end of the JSON block."):
            verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))

def test_verify_error_malformed_json(verifier, valid_reason, valid_truth, tmp_path):
    """Test handling of malformed JSON in the output."""
    mock_stdout = """Some output...
🎉 Task Completed
{
  "status": "pass",
  "reason" "Malformed JSON."
}
"""
    mock_stderr = ''
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = mock_stderr
    mock_process.returncode = 0
    with patch('subprocess.run', return_value=mock_process):
        with pytest.raises(PochiOutputError, match="Failed to parse JSON from pochi output"):
            verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))

def test_missing_task_completed_marker(verifier, valid_reason, valid_truth, tmp_path):
    """Test handling of output missing the 'Task Completed' marker."""
    mock_stdout = """Some output...
{
  "status": "pass",
  "reason": "Cannot find the starting line."
}
"""
    mock_stderr = ''
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = mock_stderr
    mock_process.returncode = 0
    with patch('subprocess.run', return_value=mock_process):
        with pytest.raises(PochiOutputError, match="Cannot find the start of the JSON block."):
            verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))

def test_pochi_executable_not_found():
    """Test that FileNotFoundError is raised if pochi is not found."""
    with patch('shutil.which', return_value=None):
        with pytest.raises(FileNotFoundError):
            PochiVerifier()

def test_called_process_error(verifier, valid_reason, valid_truth, tmp_path):
    """Test that CalledProcessError is handled correctly."""
    mock_stdout = "Some output...\n"
    mock_stderr = "Some error...\n"
    def raise_cpe(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], output=mock_stdout, stderr=mock_stderr)
    with patch('subprocess.run', side_effect=raise_cpe):
        with pytest.raises(subprocess.CalledProcessError):
            verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))

def test_default_trajectory_dir_creation(verifier, valid_reason, valid_truth, tmp_path):
    """Test that a default trajectory directory is created and used."""
    mock_stdout = """Some output...
🎉 Task Completed
{
  "status": "pass",
  "reason": "Success"
}
"""
    mock_stderr = ''
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = mock_stderr
    mock_process.returncode = 0
    with patch('subprocess.run', return_value=mock_process):
        with patch('os.getcwd', return_value=str(tmp_path)):
            result = verifier.verify(valid_reason, valid_truth)
            created_dirs = [d for d in tmp_path.joinpath("pochi-verifier", "trajectories").iterdir() if d.is_dir()]
            assert len(created_dirs) == 1
    assert result.status == "pass"

def test_agent_browser_not_available():
    """Test that ValueError is raised if agent-browser is not available."""
    with patch('shutil.which', side_effect=lambda x: '/fake/path/to/' + x if x == "pochi" else None):
        with patch('subprocess.run', side_effect=FileNotFoundError()):
            verifier = PochiVerifier()
            with pytest.raises(ValueError, match="Browser agent is not available."):
                verifier.verify("reason", "truth")

def test_ffmpeg_path_used(valid_reason, valid_truth, tmp_path):
    """Test that the ffmpeg path is used in the command."""
    with patch('shutil.which', side_effect=lambda x: '/fake/path/to/' + x):
        with patch('subprocess.run') as mock_run_init:
            mock_run_init.return_value = MagicMock(returncode=0)
            verifier = PochiVerifier(ffmpeg_path="/custom/ffmpeg")
            mock_stdout = """Some output...
🎉 Task Completed
{
  "status": "pass",
  "reason": "Success"
}
"""
            mock_stderr = ''
            mock_process = MagicMock()
            mock_process.stdout = mock_stdout
            mock_process.stderr = mock_stderr
            mock_process.returncode = 0
            with patch('subprocess.run', return_value=mock_process) as mock_run:
                verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))
                command = mock_run.call_args[0][0]
                assert "--ffmpeg" in command
                assert "/custom/ffmpeg" in command
