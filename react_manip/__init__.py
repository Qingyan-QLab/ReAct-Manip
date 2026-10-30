"""CL-Manip external-policy evaluation runtime."""
# Set this before any task imports robosuite/Numba. Some historical task
# modules use setdefault('NUMBA_DISABLE_JIT', '1') after robosuite is loaded;
# switching modes mid-process breaks controller compilation and replay IDs.
import os
os.environ.setdefault('NUMBA_DISABLE_JIT', '0')
__version__ = '0.4.0'
from .core import BenchmarkRunner, SuccessRule, TaskSpec
from .builtin import discover_task_definitions, register_builtin_tasks
from .instances.public import InstanceBundle, create_instance, open_instance, task_score
__all__ = ['BenchmarkRunner', 'SuccessRule', 'TaskSpec', 'discover_task_definitions',
           'register_builtin_tasks', 'InstanceBundle', 'create_instance', 'open_instance', 'task_score']
