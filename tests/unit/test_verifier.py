
import io
import os
import pytest
import subprocess
from unittest.mock import patch, MagicMock
from pochi_verifier.verifier import PochiVerifier, VerificationFailedError, PochiOutputError

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
    mock_stdout = io.StringIO(
        "Some output...\nTask Completed\n{\n  \"status\": \"pass\",\n  \"reason\": \"Everything is awesome.\"\n}\n"
    )
    mock_stderr = io.StringIO("")
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = mock_stderr
    mock_process.poll.side_effect = [None, 0]
    mock_process.wait.return_value = 0
    mock_process.returncode = 0

    with patch('subprocess.Popen', return_value=mock_process) as mock_popen:
        result = verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))
        assert result.status == "pass"
        assert result.reason == "Everything is awesome."
        assert os.path.exists(os.path.join(tmp_path, "stdout.txt"))
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        command = args[0]
        assert "pochi" in command
        assert "--model" in command
        assert "google/gemini-3-flash" in command
        assert "--attempt-completion-schema" in command
        assert "--stream-json" in command
        assert "--prompt" in command

def test_verify_success_with_suffix_output(verifier, valid_reason, valid_truth, tmp_path):
    """Test a successful verification."""
    mock_stdout = io.StringIO(
        "Some output...\n\n🎉 Task Completed\n{\n  \"status\": \"pass\",\n  \"reason\": \"Test passed with suffix.\"\n}\nMore output..."
    )
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = io.StringIO("")
    mock_process.poll.side_effect = [None, 0]
    mock_process.wait.return_value = 0
    mock_process.returncode = 0
    with patch('subprocess.Popen', return_value=mock_process):
        result = verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))
        assert result.status == "pass"
        assert result.reason == "Test passed with suffix."

def test_verify_failure_status(verifier, valid_reason, valid_truth, tmp_path):
    """Test a verification that returns a 'fail' status."""
    mock_stdout = io.StringIO(
        "Some output...\nTask Completed\n{\n  \"status\": \"fail\",\n  \"reason\": \"Something went wrong.\"\n}\n"
    )
    mock_stderr = io.StringIO("")
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = mock_stderr
    mock_process.poll.side_effect = [None, 0]
    mock_process.wait.return_value = 0
    mock_process.returncode = 0

    with patch('subprocess.Popen', return_value=mock_process):
        with pytest.raises(VerificationFailedError, match="Verification failed: Something went wrong."):
            verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))

def test_pochi_executable_not_found():
    """Test that FileNotFoundError is raised if pochi is not found."""
    with patch('shutil.which', return_value=None):
        with pytest.raises(FileNotFoundError):
            PochiVerifier()

def test_called_process_error(verifier, valid_reason, valid_truth, tmp_path):
    """Test that CalledProcessError is handled correctly."""
    mock_stdout = io.StringIO("Some output...\n")
    mock_stderr = io.StringIO("Some error...\n")
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = mock_stderr
    mock_process.poll.side_effect = [None, 0]
    mock_process.wait.return_value = 1
    mock_process.returncode = 1

    def raise_cpe(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], output="Some output...\n", stderr="Some error...\n")

    with patch('subprocess.Popen', return_value=mock_process):
        with pytest.raises(subprocess.CalledProcessError):
            verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))
        assert os.path.exists(os.path.join(tmp_path, "stdout.txt"))
        assert os.path.exists(os.path.join(tmp_path, "stderr.txt"))

def test_malformed_json_output(verifier, valid_reason, valid_truth, tmp_path):
    """Test handling of malformed JSON in the output."""
    mock_stdout = io.StringIO(
        "Some output...\nTask Completed\n{\n  \"status\": \"pass\",\n  \"reason\": \"Everything is awesome.\""
    ) # Malformed JSON
    mock_stderr = io.StringIO("")
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = mock_stderr
    mock_process.poll.side_effect = [None, 0]
    mock_process.wait.return_value = 0
    mock_process.returncode = 0

    with patch('subprocess.Popen', return_value=mock_process):
        with pytest.raises(PochiOutputError, match="Cannot extract the JSON from the output message."):
            verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))

def test_missing_task_completed_marker(verifier, valid_reason, valid_truth, tmp_path):
    """Test handling of output missing the 'Task Completed' marker."""
    mock_stdout = io.StringIO(
        "Some output...\n{\n  \"status\": \"pass\",\n  \"reason\": \"Everything is awesome.\"\n}\n"
    )
    mock_stderr = io.StringIO("")
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = mock_stderr
    mock_process.poll.side_effect = [None, 0]
    mock_process.wait.return_value = 0
    mock_process.returncode = 0

    with patch('subprocess.Popen', return_value=mock_process):
        with pytest.raises(PochiOutputError, match="Cannot extract the JSON from the output message."):
            verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))

def test_default_trajectory_dir_creation(verifier, valid_reason, valid_truth, tmp_path):
    """Test that a default trajectory directory is created and used."""
    mock_stdout = io.StringIO('Task Completed\n{"status": "pass", "reason": "Success"}\n')
    mock_stderr = io.StringIO("")
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = mock_stderr
    mock_process.poll.side_effect = [None, 0]
    mock_process.wait.return_value = 0
    mock_process.returncode = 0

    with patch('subprocess.Popen', return_value=mock_process):
        with patch('os.getcwd', return_value=str(tmp_path)):
            result = verifier.verify(valid_reason, valid_truth)
            created_dirs = [d for d in tmp_path.joinpath("pochi-verifier", "trajectories").iterdir() if d.is_dir()]
            assert len(created_dirs) == 1
            trajectory_dir = created_dirs[0]
            assert trajectory_dir.joinpath("stdout.txt").is_file()
            assert trajectory_dir.joinpath("stdout.txt").read_text(encoding='utf-8') == 'Task Completed\n{"status": "pass", "reason": "Success"}\n'
            # stderr.txt will be created but empty
            assert trajectory_dir.joinpath("stderr.txt").is_file()
            assert trajectory_dir.joinpath("stderr.txt").read_text(encoding='utf-8') == ""
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
            mock_stdout = io.StringIO('Task Completed\n{"status": "pass", "reason": "Success"}\n')
            mock_stderr = io.StringIO("")
            mock_process = MagicMock()
            mock_process.stdout = mock_stdout
            mock_process.stderr = mock_stderr
            mock_process.poll.side_effect = [None, 0]
            mock_process.wait.return_value = 0
            mock_process.returncode = 0
            with patch('subprocess.Popen', return_value=mock_process) as mock_popen:
                verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))
                args, kwargs = mock_popen.call_args
                command = args[0]
                assert "--ffmpeg" in command
                assert "/custom/ffmpeg" in command
