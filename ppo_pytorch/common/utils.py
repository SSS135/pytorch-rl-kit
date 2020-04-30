import subprocess
from pathlib import Path


def save_git_diff(save_folder, tag, repo_path):
    diff = subprocess.check_output(['git', 'diff'], cwd=repo_path)
    with open(f'{save_folder}/{Path(repo_path).name}_{tag}.diff', 'wb') as file:
        file.write(diff)


def is_running_on_windows():
    import sys
    try:
        sys.getwindowsversion()
    except AttributeError:
        return False
    else:
        return True


def set_low_priority():
    """
    Set the priority of the process to below-normal.
    https://stackoverflow.com/questions/1023038/change-process-priority-in-python-cross-platform
    """
    is_windows = is_running_on_windows()

    if is_windows:
        # Based on:
        #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
        #   http://code.activestate.com/recipes/496767/
        import win32api, win32process, win32con

        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        import os

        os.nice(1)