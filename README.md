# Pochi Verifier

A Python interface for the `pochi` CLI to streamline verification processes.

## Description

`pochi-verifier` provides a convenient way to run `pochi` CLI commands from your Python code. This allows you to integrate `pochi`'s verification capabilities directly into your Python-based testing frameworks.

## Requirements

- **Python 3.12+**
- **[`pochi` CLI](https://docs.getpochi.com/cli/)**: This package is a wrapper around the `pochi` command-line tool. You must have the `pochi` CLI installed.
- **[`agent-browser`](https://www.npmjs.com/package/agent-browser)**: Required for running browser verifications.
- **[`FFmpeg`](https://ffmpeg.org/)**: Required for recording the browser verification trajectory as a video file.

## Usage

`pochi-verifier` works with a specification that describes the verification task. Here is an example of a `browser_verification` spec for a to-do list application:

**`verify_todo.json`**
```json
{
  "name": "verify_todo_list_functionality",
  "reason": "The application should feature a fully functional to-do list on its homepage. Users must be able to add, edit, and remove items, as well as mark items as complete and incomplete.",
  "truth": "Navigate to http://localhost:8080. Verify that an input field for adding new to-do items is visible. Click the input field, type 'Buy milk', and press Enter. Verify that 'Buy milk' appears in the to-do list. Find the 'Buy milk' item and click the checkbox next to it to mark it as complete. Verify that the 'Buy milk' item is now marked as done (e.g., has a line-through). Find the 'Buy milk' item and click the 'delete' button next to it. Verify that 'Buy milk' is no longer present in the to-do list."
}
```

You can then use `pochi-verifier` to run this verification from your Python code. You can also specify a directory to save trajectory files.

```python
import json
from pochi_verifier import PochiVerifier

# Create a verifier instance
verifier = PochiVerifier()

# Example: Run a verification command
with open("path/to/your/verify_todo.json", "r") as f:
    spec = json.load(f)

reason = spec["reason"]
truth = spec["truth"]

try:
    result = verifier.verify(
        reason=reason,
        truth=truth,
        use_browser_agent=True,
        trajectory_dir="path/to/save/trajectory/files"
    )
except (FileNotFoundError, ValueError) as e:
    print(f"Configuration error: {e}")
    raise
except Exception as e:
    print(f"Verification could not be completed: {e}")
    raise

if result.status == "pass":
    print("Verification successful!")
else:
    print("Verification failed.")

print(f"Status: {result.status}")
print(f"Reason: {result.reason}")
```

