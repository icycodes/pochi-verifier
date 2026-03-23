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
    spec = {
        "name": "integration_test_blank_page",
        "type": "browser_verification",
        "reason": "To confirm the integration with the real pochi CLI works.",
        "verification": "Go to 'about:blank' and verify that the page title is an empty string."
    }

    # This call will execute the actual 'pochi' command.
    # It may be slow and requires proper configuration (e.g., API keys).
    result = verifier.verify(
        spec,
        trajectory_dir=trajectory_dir
    )

    assert isinstance(result, VerificationResult)
    assert result.status == "pass"
