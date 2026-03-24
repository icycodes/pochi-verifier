import pytest
from unittest.mock import patch, MagicMock
import os
import json
import subprocess
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
    mock_stdout = """
    Some output...
    Task Completed
    {
      "status": "pass",
      "reason": "Everything is awesome."
    }
    """
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = ""

    # The first call to subprocess.run is in __init__ (mocked in the fixture), 
    # but since we're patching it here, we need to account for it or re-mock it.
    with patch('subprocess.run', return_value=mock_process) as mock_run:
        result = verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))
        
        assert result.status == "pass"
        assert result.reason == "Everything is awesome."
        
        # Check that stdout and stderr files were created
        assert os.path.exists(os.path.join(tmp_path, "stdout.txt"))
        
        # Check that the command was called correctly
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        command = args[0]
        assert "pochi" in command
        assert "--model" in command
        assert "google/gemini-3-flash" in command
        assert "--attempt-completion-schema" in command
        assert "--stream-json" in command
        assert "--prompt" in command
        assert valid_reason in command[-1]
        assert valid_truth in command[-1]

def test_verify_failure_status(verifier, valid_reason, valid_truth, tmp_path):
    """Test a verification that returns a 'fail' status."""
    mock_stdout = """
    Some output...
    Task Completed
    {
      "status": "fail",
      "reason": "Something went wrong."
    }
    """
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = ""

    with patch('subprocess.run', return_value=mock_process):
        with pytest.raises(VerificationFailedError, match="Verification failed: Something went wrong."):
            verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))

def test_pochi_executable_not_found():
    """Test that FileNotFoundError is raised if pochi is not found."""
    with patch('shutil.which', return_value=None):
        with pytest.raises(FileNotFoundError):
            PochiVerifier()

def test_called_process_error(verifier, valid_reason, valid_truth, tmp_path):
    """Test that CalledProcessError is handled correctly."""
    with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, "cmd", "stdout", "stderr")):
        with pytest.raises(subprocess.CalledProcessError):
            verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))
        
        # Check that stdout and stderr files were written even on error
        assert os.path.exists(os.path.join(tmp_path, "stdout.txt"))
        assert os.path.exists(os.path.join(tmp_path, "stderr.txt"))

def test_malformed_json_output(verifier, valid_reason, valid_truth, tmp_path):
    """Test handling of malformed JSON in the output."""
    mock_stdout = """
    Some output...
    Task Completed
    {
      "status": "pass",
      "reason": "Everything is awesome."
    """ # Malformed JSON
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = ""

    with patch('subprocess.run', return_value=mock_process):
        with pytest.raises(PochiOutputError, match="Failed to parse JSON from pochi output"):
            verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))

def test_missing_task_completed_marker(verifier, valid_reason, valid_truth, tmp_path):
    """Test handling of output missing the 'Task Completed' marker."""
    mock_stdout = """
    Some output...
    {
      "status": "pass",
      "reason": "Everything is awesome."
    }
    """
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = ""

    with patch('subprocess.run', return_value=mock_process):
        with pytest.raises(PochiOutputError, match="Could not find 'Task Completed' marker in the output."):
            verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))

def test_default_trajectory_dir_creation(verifier, valid_reason, valid_truth, tmp_path):
    """Test that a default trajectory directory is created and used."""
    mock_stdout = """
    Task Completed
    {"status": "pass", "reason": "Success"}
    """
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = ""

    # Patch subprocess.run and os.getcwd to control the environment
    with patch('subprocess.run', return_value=mock_process):
        with patch('os.getcwd', return_value=str(tmp_path)):
            result = verifier.verify(valid_reason, valid_truth)

            # Find the created trajectory directory
            created_dirs = [d for d in tmp_path.joinpath("pochi-verifier", "trajectories").iterdir() if d.is_dir()]
            assert len(created_dirs) == 1
            trajectory_dir = created_dirs[0]

            # Check that output files were written
            assert trajectory_dir.joinpath("stdout.txt").is_file()
            assert trajectory_dir.joinpath("stdout.txt").read_text(encoding='utf-8') == mock_stdout
            assert not trajectory_dir.joinpath("stderr.txt").exists() # stderr was empty

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
            
            mock_stdout = """
            Task Completed
            {"status": "pass", "reason": "Success"}
            """
            mock_process = MagicMock()
            mock_process.stdout = mock_stdout
            mock_process.stderr = ""
            
            with patch('subprocess.run', return_value=mock_process) as mock_run_verify:
                verifier.verify(valid_reason, valid_truth, trajectory_dir=str(tmp_path))
                
                args, kwargs = mock_run_verify.call_args
                command = args[0]
                assert "--ffmpeg" in command
                assert "/custom/ffmpeg" in command
