---
name: skill-authoring
description: Create, revise, or convert a portable Agent Skill package with a standard SKILL.md and optional references, scripts, or assets. Use for create skill, convert skill, skill authoring, 创建 skill, 生成 skill, 转换 skill, or 编写 skill requests.
metadata:
  version: "2.0.0"
---

# Portable Skill Authoring

Create an Agent Skill that remains useful outside RepoRivet. A Skill is a directory whose entry
point is `SKILL.md`; optional supporting material belongs in `references/`, `scripts/`, or `assets/`.

## Format

Use this minimal front matter:

```yaml
---
name: example-skill
description: Explain what the Skill does and when an agent should use it.
---
```

The `name` must match the parent directory, contain only lowercase letters, digits, and hyphens,
and be at most 64 characters. Keep `description` under 1024 characters and make its routing boundary
specific. Optional standard fields are `license`, `compatibility`, `metadata`, and experimental
`allowed-tools`.

Do not put RepoRivet modes, approval policy, verification contracts, trigger objects, hooks, or
other host-specific behavior in front matter. A Skill is guidance and never grants permissions.

## Workflow

1. Define one reusable task boundary and the requests that should activate it.
2. Write a precise `description` covering both purpose and use conditions.
3. Keep the main instructions short and operational. Move lengthy optional details to
   `references/` and say exactly when to read each file.
4. Put reusable executable helpers in `scripts/` only when needed. They are package resources, not
   hooks, and must never be described as automatically executed or implicitly trusted.
5. Put templates and other non-instruction artifacts in `assets/`.
6. Validate with `reporivet skill validate <skill-directory>`.
7. Reread the complete package and confirm that every referenced path exists.

In a read-only planning workflow, inspect and describe the package without creating it. File writes,
script execution, installation, and verification remain subject to the host agent's normal mode,
workspace, and approval controls.

## Conversion

When converting another agent's Skill:

- Preserve its reusable Markdown instructions and portable resources.
- Normalize the directory and `name` together.
- Supply an explicit standard `description` when the source does not have one.
- Omit provider models, lifecycle hooks, callbacks, permission claims, and other host-only fields.
- Report meaningful behavior that could not be represented instead of hiding the loss.

`reporivet skill convert` performs structural normalization; it does not prove the instructions are
correct or execute any bundled script.

## Completion

A finished package has a valid standard `SKILL.md`, a matching directory name, focused instructions,
and no missing resource references. Installation into the user-global Skill directory is a separate
explicit action.
