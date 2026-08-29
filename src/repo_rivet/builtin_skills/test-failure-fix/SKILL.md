---
schema_version: 1
id: test-failure-fix
name: Test Failure Fix
version: 1.1.1
summary: Diagnose a concrete failing test, make a minimal snapshot-anchored repair, and verify it.
category: workflow
activation:
  explicit: false
  automatic: true
triggers:
  task_types: [fix-test]
  keywords: [failing test, test failure, test regression, 测试失败, 修复测试, 测试回归]
compatible_modes: [plan, execute]
requested_tools:
  - list_files
  - read_file
  - search_text
  - edit_file
  - write_file
  - run_command
  - run_verification
  - git_status
  - git_diff
requirements:
  before_edit: []
  before_finish: []
verification_profiles: []
limits:
  max_prompt_tokens: 1400
  max_active_support_skills: 0
---

# Objective

Repair a concrete, reproducible test failure with the smallest evidence-supported change and leave
current deterministic verification evidence. Apply this workflow only when a failing test or test
regression is part of the task, not to every debugging request.

# Mode Handling

In Plan Mode, inspect the failure, relevant tests, and implementation, then submit a Plan Artifact.
Do not edit files or execute commands. In Execute Mode, perform the approved or directly requested
repair through the normal editing, approval, and verification controls.

# Procedure

1. Identify the existing test entry point and capture the reported failure or reproduce it when
   execution is appropriate and dependencies are already available.
2. Read the failing test and the smallest implementation surface it exercises.
3. State a bounded diagnosis tied to observed output and source evidence.
4. Register a required `test` verification check with a deterministic oracle before claiming the
   repair is complete. Add build or behavior checks only when they independently match the task.
5. Edit only the files supported by the diagnosis, using current snapshots and seen ranges.
6. Run the narrow failing check first, then the relevant broader suite when its cost is reasonable.
7. After the last edit, ensure required results belong to the current workspace revision and inspect
   the final diff.

# Constraints

- Do not weaken, delete, skip, or rewrite a valid failing test merely to make it pass.
- A test change is acceptable only when evidence shows the test expectation or fixture is wrong;
  explain that evidence explicitly.
- Do not install dependencies unless the user separately authorizes it.
- Do not treat a build, unrelated command, or stale test result as proof that the failure is fixed.
- Do not broaden the repair into unrelated cleanup.

# Completion Conditions

Report success only when the originally failing behavior has a required `test` check that passes at
the current workspace revision, the final diff has been inspected after the latest edit, and no
process remains active. Otherwise report the exact blocker, current evidence, and remaining check
without claiming completion.
