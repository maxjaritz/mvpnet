import os
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.collect_env import run, run_and_read_all

_ROOT_DIR = os.path.abspath(os.path.dirname(__file__) + '/../..')


def git_available():
    try:
        run('git version')
        return True
    except:
        return False


def get_git_rev(root_dir=_ROOT_DIR, first=8):
    git_rev = run_and_read_all(run, 'cd {:s} && git rev-parse HEAD'.format(root_dir))
    return git_rev[:first] if git_rev else git_rev


def get_git_modifed(root_dir=_ROOT_DIR, git_dir=_ROOT_DIR):
    # Note that paths returned by git ls-files are relative to the script.
    return run_and_read_all(run, 'cd {:s} && git ls-files {:s} -m'.format(root_dir, git_dir))


def get_git_untracked(root_dir=_ROOT_DIR, git_dir=_ROOT_DIR):
    # Note that paths returned by git ls-files are relative to the script.
    return run_and_read_all(run, 'cd {:s} && git ls-files {:s} --exclude-standard --others'.format(root_dir, git_dir))


def get_PIL_version():
    try:
        import PIL
    except ImportError:
        return '\n No Pillow is found.'
    else:
        return '\nPillow ({})'.format(PIL.__version__)


def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += get_PIL_version()
    if git_available():
        env_str += '\nGit revision number: {}'.format(get_git_rev())
        env_str += '\nGit Modified\n{}'.format(get_git_modifed())
        # env_str += '\nGit Untrakced\n {}'.format(get_git_untracked())
    return env_str
