import pytest
from unittest.mock import patch, MagicMock
import os
import json
import subprocess
from pochi_verifier.verifier import PochiVerifier, VerificationFailedError, PochiOutputError

@pytest.fixture
def verifier():
    """Fixture to provide a PochiVerifier instance with a mocked pochi executable."""
    with patch('shutil.which', return_value='/fake/path/to/pochi'):
        return PochiVerifier()

@pytest.fixture
def valid_spec():
    """Fixture to provide a valid spec dictionary."""
    return {
        "name": "test_valid_spec",
        "type": "browser_verification",
        "reason": "A valid test reason.",
        "verification": "A valid verification step."
    }

def test_verify_success(verifier, valid_spec, tmp_path):
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

    with patch('subprocess.run', return_value=mock_process) as mock_run:
        result = verifier.verify(valid_spec, trajectory_dir=str(tmp_path))
        
        assert result.status == "pass"
        assert result.reason == "Everything is awesome."
        
        # Check that stdout and stderr files were created
        assert os.path.exists(os.path.join(tmp_path, "stdout.txt"))
        
        # Check that the command was called correctly
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert "--model" in args[0]

def test_verify_failure_status(verifier, valid_spec, tmp_path):
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
            verifier.verify(valid_spec, trajectory_dir=str(tmp_path))

def test_pochi_executable_not_found():
    """Test that FileNotFoundError is raised if pochi is not found."""
    with patch('shutil.which', return_value=None):
        with pytest.raises(FileNotFoundError):
            PochiVerifier()

def test_invalid_spec_missing_field(verifier):
    """Test that ValueError is raised for an invalid spec."""
    with pytest.raises(ValueError, match="The 'name' field is missing or empty in the spec."):
        verifier.verify({"type": "browser_verification", "reason": "r", "verification": "v"})

def test_called_process_error(verifier, valid_spec, tmp_path):
    """Test that CalledProcessError is handled correctly."""
    with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, "cmd", "stdout", "stderr")):
        with pytest.raises(subprocess.CalledProcessError):
            verifier.verify(valid_spec, trajectory_dir=str(tmp_path))
        
        # Check that stdout and stderr files were written even on error
        assert os.path.exists(os.path.join(tmp_path, "stdout.txt"))
        assert os.path.exists(os.path.join(tmp_path, "stderr.txt"))

def test_malformed_json_output(verifier, valid_spec, tmp_path):
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
            verifier.verify(valid_spec, trajectory_dir=str(tmp_path))

def test_missing_task_completed_marker(verifier, valid_spec, tmp_path):
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
            verifier.verify(valid_spec, trajectory_dir=str(tmp_path))

def test_default_trajectory_dir_creation(verifier, valid_spec, tmp_path):
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
            result = verifier.verify(valid_spec)

            # Find the created trajectory directory
            created_dirs = [d for d in tmp_path.joinpath("pochi-verifier", "trajectories").iterdir() if d.is_dir()]
            assert len(created_dirs) == 1
            trajectory_dir = created_dirs[0]

            # Check that output files were written
            assert trajectory_dir.joinpath("stdout.txt").is_file()
            assert trajectory_dir.joinpath("stdout.txt").read_text(encoding='utf-8') == mock_stdout
            assert not trajectory_dir.joinpath("stderr.txt").exists() # stderr was empty

    assert result.status == "pass"
