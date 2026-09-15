"""Optional task-specific prompt hooks.

The public package deliberately ships no private task history or deployment-specific
prompt profiles. Applications may provide their own policy layer around the stable
robot tool protocol.
"""

PLUG_PROFILE = None
PLUG_CONTROL = ""
PLUG_DONE = "Declare completion only after fresh observations establish the physical goal."


def control_prompt_profile(task_instruction: str = "") -> str | None:
    """Return a local profile name, if an application has registered one."""
    return None
