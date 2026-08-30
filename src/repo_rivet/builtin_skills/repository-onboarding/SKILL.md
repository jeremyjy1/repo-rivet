---
name: repository-onboarding
description: Inspect an unfamiliar repository and produce a bounded, evidence-based architecture overview. Use for repository overview, architecture overview, project structure, onboarding, 项目概览, 架构概览, 理解项目, or 项目结构 requests; do not use for implementation tasks.
metadata:
  version: "2.0.0"
---

# Objective

Build a concise, evidence-based understanding of an unfamiliar repository without changing it.
Apply this workflow only when the user asks for repository onboarding, a broad architecture
overview, or project-structure explanation. The presence of a Git repository alone is not a match.
Do not apply its read-only constraint to implementation, debugging, or other change requests.

# Procedure

1. Inspect the top-level tree and repository status.
2. Read project metadata, entry points, and the smallest set of architecture-defining files.
3. Search for key interfaces and representative tests before inferring behavior.
4. Separate directly observed facts, supported inferences, and unresolved questions.
5. Explain the components and data flow at the depth requested by the user.

# Constraints

- Do not edit files, install dependencies, or run project code as part of onboarding.
- Do not infer architecture only from filenames when source evidence is available.
- Cite file paths and line locations for important claims.
- Keep inspection proportional to the requested scope; do not inventory the entire repository by
  default.
- Do not turn onboarding into an unsolicited review or implementation plan.

# Completion Conditions

The response answers the requested onboarding question with direct file evidence, identifies
important entry points and boundaries when relevant, and clearly labels material unknowns. It need
not cover every subsystem when the user requested a narrower overview.
