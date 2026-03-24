import pytest
import shutil
import os
from pochi_verifier import PochiVerifier, VerificationResult

# Mark the entire module as 'integration'
pytestmark = pytest.mark.integration

def test_pochi_integration_success():
    """
    An integration test that calls the real pochi CLI to verify a simple success case.
    This test requires the 'pochi' CLI to be installed and configured.
    It will navigate to "about:blank" and check for an empty title.
    """
    trajectory_dir = "tests/integration/trajectories"

    # Clean up previous run
    if os.path.exists(trajectory_dir):
        shutil.rmtree(trajectory_dir)

    verifier = PochiVerifier()
    
    reason = "To confirm the integration with the real pochi CLI works."
    truth = "Go to 'about:blank' and verify that the page title is an empty string."

    # This call will execute the actual 'pochi' command.
    # It may be slow and requires proper configuration (e.g., API keys).
    result = verifier.verify(
        reason=reason,
        truth=truth,
        trajectory_dir=trajectory_dir
    )

    assert isinstance(result, VerificationResult)
    assert result.status == "pass"
