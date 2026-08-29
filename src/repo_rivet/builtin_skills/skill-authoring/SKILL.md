---
schema_version: 1
id: skill-authoring
name: RepoRivet Skill Authoring
version: 1.1.2
summary: Create or convert a single-file RepoRivet Skill with semantically consistent capabilities and requirements.
category: workflow
activation:
  explicit: false
  automatic: true
triggers:
  task_types: [design-skill, convert-skill, author-skill]
  file_globs: ["**/SKILL.md"]
  keywords: [create skill, convert skill, skill authoring, 创建 skill, 生成 skill, 转换 skill, 编写 skill]
compatible_modes: [plan, execute]
requested_tools:
  - list_files
  - read_file
  - search_text
  - write_file
  - edit_file
  - run_command
  - run_verification
  - git_diff
requirements:
  before_edit: []
  before_finish: []
verification_profiles: []
limits:
  max_prompt_tokens: 2000
  max_active_support_skills: 0
---

# Objective

Create or convert a native RepoRivet `SKILL.md` whose instructions, requested capabilities, and
runtime completion requirements describe the same reusable workflow.

# Platform Boundary

RepoRivet version 1 installs and loads only `SKILL.md`. Supporting scripts, templates, hooks,
callbacks, dynamic imports, and executable extensions are not part of an installed Skill. Do not
create or reference companion resources as if installation will preserve them. When converting a
source Skill, omit such resources and report the lost capability.

A Skill may request existing tools, but it cannot grant capabilities, approve actions, widen the
workspace, or override Controller policy. Installation into user-global scope remains a separate,
explicit user action.

# Authoring Workflow

1. Identify the reusable task boundary and exclude adjacent tasks that should not activate it.
2. Determine the modes and existing RepoRivet task tools actually needed by the instructions.
3. Define observable runtime completion conditions before choosing any manifest requirement.
4. Create or edit one `SKILL.md` under the workspace, normally
   `reporivet-skills/<skill-id>/SKILL.md`.
5. Use schema version 1, a lowercase hyphenated ID, and a `major.minor.patch` version.
6. Reread the complete file and perform the semantic review below.
7. Run `reporivet skill validate <draft>` as artifact validation. When it must count as completion
   evidence, register it as a `lint` or `custom` check with a deterministic exit-code and output
   oracle, then use `run_verification`.

In Plan Mode, inspect and produce a plan only. Do not create or edit the draft until Execute Mode.

# Runtime Requirement Semantics

The checks used while authoring the Skill are not requirements of tasks that later use the Skill.
In particular, validating `SKILL.md` must not cause the generated Skill to require a behavior check.

Leave runtime requirements empty unless every successful use of the generated Skill necessarily
produces that exact evidence:

- `required_build_passed`: every required `build` check in the active verification plan passed at
  the current workspace revision.
- `required_tests_passed`: every required `test` check passed at the current revision.
- `required_behavior_checks_passed`: every required `behavior` check passed at the current
  revision. Do not use this as a generic synonym for "verified".
- `no_stale_verification`: no retained verification result is stale or belongs to another revision.
- `git_diff_reviewed`: `git_diff` succeeded after the latest successful file edit or creation.
- `no_active_processes`: no recorded process remains unfinished.
- `plan_approved`: an executable Plan Artifact is currently approved.
- `target_snapshot_current` and `target_range_seen`: existing-file edits remain anchored to current,
  previously observed content.

The requirement list is conjunctive. RepoRivet cannot express "build, test, or behavior" with these
fields. For workflows with alternative valid outcomes, narrow the Skill to one invariant or leave
the verification-kind requirements empty and state the conditional evidence in the instructions.

# Capability Review

- Every task action prescribed by the body must be possible with `requested_tools`.
- Use `run_command` for arbitrary shell, debugger, compiler, or diagnostic commands.
- Use `run_verification` only for checks registered in the verification plan; it is not a substitute
  for general command execution.
- Do not request tools merely because they might be useful.
- Keep Plan Mode compatible only when the workflow has a meaningful read-and-plan phase.

# RepoRivet Skill Commands

Use the direct `reporivet` executable rather than wrapping it in a package runner. RepoRivet can
deterministically approve these bounded commands when the executable belongs to its trusted runtime:

- `reporivet skill list`, `show`, and `validate` only read Skill data.
- `reporivet skill init` creates one non-overwriting draft inside the workspace.
- `reporivet skill convert` is eligible for automatic approval only with an explicit `--id`; its
  source and output must remain inside the workspace and the target must be new.

Do not register `init` or `convert` as verification merely to execute them; use `run_command`.
`install`, `uninstall`, `use`, and `clear` change user-global or session state and continue through
normal approval. A Skill must never describe those commands as implicitly authorized.

# Conversion Rules

- Map only to tools that RepoRivet currently exposes. Omit unknown tools and report them.
- Remove provider-specific permissions, hooks, scripts, commands, and installation claims.
- Remove body references to resources that the single-file package cannot install.
- Preserve useful declarative guidance without importing hidden reasoning or machine-specific data.
- Do not silently turn conditional source guidance into an unconditional runtime requirement.

# Validation and Completion

`reporivet skill validate` proves structural validity, known tool names, and size limits; it does not
prove semantic correctness. Before reporting completion, also confirm that:

- objective, triggers, modes, tools, procedure, and completion conditions agree;
- every runtime requirement has an unconditional matching outcome and exact verification kind;
- no missing companion file or unsupported executable behavior is referenced;
- the final file was reread after the latest edit;
- conversion losses and unresolved choices are reported explicitly.

If installation is requested, present or run the explicit `reporivet skill install <draft>` action
through the normal approval path. Do not imply that successful validation installed or selected it.
