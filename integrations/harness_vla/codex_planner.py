"""RPent Codex backend with benchmark-specific command restrictions.

Authentication belongs to an operator-configured independent CODEX_HOME.
The simulator and scoped memory are accessed through RPent's MCP toolkit.
"""
from dataclasses import replace
from pathlib import Path
import os


def build_codex_planner(*, output_dir, recipe_tag, robot_name, model,
                        base_url, memory_dir, reasoning_effort,
                        planner_timeout_s, dashboard_events):
    if base_url:
        raise ValueError('Set CODEX_BASE_URL for the Codex backend')
    if not os.environ.get('CODEX_HOME'):
        raise ValueError('Set an independent CODEX_HOME for benchmark evaluation')
    import openai_codex
    from rpent.planner.codex import CodexPlanner

    class BenchmarkCodexPlanner(CodexPlanner):
        def _build_config(self, mcp_url):
            config = super()._build_config(mcp_url)
            # These are the evaluator-owned simulation tools the operator has
            # authorized. Keep unrelated tools subject to deny_all.
            tools = ('observe', 'locate', 'read_memory', 'write_memory', 'finish',
                     'move_to', 'move_pose', 'rotate_wrist', 'rotate_pitch',
                     'set_gripper', 'release', 'vla_act', 'move_base')
            return replace(config, config_overrides=config.config_overrides + (
                'features.shell_tool=false',
                'web_search="disabled"',
                'features.multi_agent=false',
                'features.apps=false',
                'mcp_servers.rpent.required=true',
            ) + tuple(f'mcp_servers.rpent.tools.{name}.approval_mode="approve"'
                      for name in tools))

    work = Path(output_dir) / 'planner_work'
    work.mkdir(parents=True, exist_ok=False)
    planner = BenchmarkCodexPlanner(
        output_dir=output_dir, repo_root=work, model=model,
        timeout_s=planner_timeout_s, extra_dirs=[str(memory_dir)],
        output_path=Path(output_dir) / f'codex_{recipe_tag}.txt',
        dashboard_events=dashboard_events, reasoning_effort=reasoning_effort)
    # The upstream backend defaults to full filesystem access. Both thread and
    # turn settings must be constrained; turns otherwise override the thread.
    planner._thread_options['sandbox'] = openai_codex.Sandbox.read_only
    # RPent owns episode transcripts and memory. Avoid a second persistent
    # Codex session store, which also fails on object-storage mounts.
    planner._thread_options['ephemeral'] = True
    planner._turn_options['sandbox'] = openai_codex.Sandbox.read_only
    return planner
