#!/usr/bin/env python3
"""
Skill Bank Compiler

Compiles base.md + active patches into final SKILL.md files.

Usage:
    python skill-bank/compile.py rllm-config              # single skill
    python skill-bank/compile.py rllm-config -p small-model # with profile
    python skill-bank/compile.py --group rllm              # entire group
    python skill-bank/compile.py --all                     # everything
    python skill-bank/compile.py --dry-run rllm-config     # preview only
    python skill-bank/compile.py --diff rllm-config        # show diff
    python skill-bank/compile.py --status                  # patch summary
    python skill-bank/compile.py --squash rllm-config      # merge patches into base
"""

import argparse
import copy
import difflib
import os
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import yaml


BANK_DIR = Path(__file__).parent
SECTION_OPEN = re.compile(r"<!--\s*section:([a-z0-9-]+)\s*-->")
SECTION_CLOSE = re.compile(r"<!--\s*/section:([a-z0-9-]+)\s*-->")
FRONTMATTER_DELIM = re.compile(r"^---\s*$")


def load_bank_yaml():
    path = BANK_DIR / "bank.yaml"
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def find_skill(bank, skill_name):
    for group_name, group in bank.get("groups", {}).items():
        skills = group.get("skills", {})
        if skill_name in skills:
            return group_name, skills[skill_name]
    return None, None


def get_skills_in_group(bank, group_name):
    group = bank.get("groups", {}).get(group_name)
    if not group:
        return {}
    return group.get("skills", {})


def get_all_skills(bank):
    result = {}
    for group_name, group in bank.get("groups", {}).items():
        for skill_name, skill_cfg in group.get("skills", {}).items():
            result[skill_name] = (group_name, skill_cfg)
    return result


def resolve_output_path(bank, output_rel):
    output_base = bank.get("settings", {}).get("output_base", "../")
    return (BANK_DIR / output_base / output_rel).resolve()


def parse_frontmatter(text):
    lines = text.split("\n")
    if not lines or not FRONTMATTER_DELIM.match(lines[0]):
        return {}, text
    end = None
    for i in range(1, len(lines)):
        if FRONTMATTER_DELIM.match(lines[i]):
            end = i
            break
    if end is None:
        return {}, text
    fm_text = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1 :])
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        fm = {}
    return fm, body


def parse_sections(body):
    sections = OrderedDict()
    gaps = []
    current_section = None
    current_lines = []
    gap_lines = []
    section_order = []

    for line in body.split("\n"):
        open_m = SECTION_OPEN.match(line.strip())
        close_m = SECTION_CLOSE.match(line.strip())

        if open_m:
            if current_section is not None:
                print(f"Error: nested section '{open_m.group(1)}' inside '{current_section}'", file=sys.stderr)
                sys.exit(1)
            if gap_lines:
                gaps.append(("\n".join(gap_lines), len(section_order)))
                gap_lines = []
            current_section = open_m.group(1)
            current_lines = []
            section_order.append(current_section)
        elif close_m:
            name = close_m.group(1)
            if current_section != name:
                print(f"Error: closing '{name}' but current section is '{current_section}'", file=sys.stderr)
                sys.exit(1)
            content = "\n".join(current_lines)
            if content.startswith("\n"):
                content = content[1:]
            if content.endswith("\n"):
                content = content[:-1]
            sections[current_section] = content
            current_section = None
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)
        else:
            gap_lines.append(line)

    if current_section is not None:
        print(f"Error: unclosed section '{current_section}'", file=sys.stderr)
        sys.exit(1)

    if gap_lines:
        gaps.append(("\n".join(gap_lines), len(section_order)))

    return sections, gaps, section_order


def load_patch(patch_path):
    text = patch_path.read_text()
    fm, body = parse_frontmatter(text)
    fm["_body"] = body.strip()
    fm["_path"] = str(patch_path)
    return fm


def topo_sort(patches):
    graph = {p["id"]: set() for p in patches}
    patch_map = {p["id"]: p for p in patches}
    local_ids = set(graph.keys())

    for p in patches:
        for dep in p.get("depends_on", []):
            if ":" not in dep and dep in local_ids:
                graph[p["id"]].add(dep)

    visited = set()
    order = []
    visiting = set()

    def dfs(node):
        if node in visiting:
            print(f"Error: circular dependency involving '{node}'", file=sys.stderr)
            sys.exit(1)
        if node in visited:
            return
        visiting.add(node)
        for dep in graph.get(node, set()):
            dfs(dep)
        visiting.discard(node)
        visited.add(node)
        order.append(node)

    for node in graph:
        dfs(node)

    return [patch_map[pid] for pid in order]


def check_conflicts(patches):
    active_ids = {p["id"] for p in patches}
    for p in patches:
        for conflict in p.get("conflicts_with", []):
            if conflict in active_ids:
                print(
                    f"Error: patch '{p['id']}' conflicts with active patch '{conflict}'",
                    file=sys.stderr,
                )
                sys.exit(1)


def check_cross_skill_deps(patches, bank):
    for p in patches:
        for dep in p.get("depends_on", []):
            if ":" not in dep:
                continue
            skill_ref, patch_ref = dep.split(":", 1)
            group_name, _ = find_skill(bank, skill_ref)
            if group_name is None:
                print(f"Error: cross-skill dep '{dep}' — skill '{skill_ref}' not found", file=sys.stderr)
                sys.exit(1)
            manifest_path = BANK_DIR / group_name / skill_ref / "manifest.yaml"
            if not manifest_path.exists():
                print(f"Error: cross-skill dep '{dep}' — manifest not found at {manifest_path}", file=sys.stderr)
                sys.exit(1)
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f) or {}
            if patch_ref not in manifest.get("active", []):
                print(
                    f"Error: cross-skill dep '{dep}' — patch '{patch_ref}' is not active in '{skill_ref}'",
                    file=sys.stderr,
                )
                sys.exit(1)


def apply_patches(sections, section_order, patches):
    sections = OrderedDict(sections)
    section_order = list(section_order)

    for p in patches:
        target = p.get("target_section", "")
        action = p.get("action", "replace")
        body = p["_body"]

        if action == "insert_after":
            if target and target not in sections:
                print(f"Error: patch '{p['id']}' targets non-existent section '{target}'", file=sys.stderr)
                sys.exit(1)
            new_name = p["id"]
            sections[new_name] = body
            if target:
                idx = section_order.index(target)
                section_order.insert(idx + 1, new_name)
            else:
                section_order.append(new_name)
            continue

        if target not in sections:
            print(f"Error: patch '{p['id']}' targets non-existent section '{target}'", file=sys.stderr)
            sys.exit(1)

        if action == "replace":
            sections[target] = body
        elif action == "append":
            sections[target] = sections[target] + "\n\n" + body
        elif action == "prepend":
            sections[target] = body + "\n\n" + sections[target]
        else:
            print(f"Error: unknown action '{action}' in patch '{p['id']}'", file=sys.stderr)
            sys.exit(1)

    return sections, section_order


def reassemble(frontmatter, sections, section_order, gaps):
    parts = []

    if frontmatter:
        fm_text = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True).strip()
        parts.append(f"---\n{fm_text}\n---\n")

    gap_map = {}
    for text, pos in gaps:
        gap_map.setdefault(pos, []).append(text)

    for pos in sorted(gap_map):
        if pos == 0:
            for g in gap_map[pos]:
                stripped = g.strip("\n")
                if stripped:
                    parts.append(stripped)

    for i, sec_name in enumerate(section_order):
        content = sections.get(sec_name, "")
        parts.append(content)

        after_pos = i + 1
        if after_pos in gap_map:
            for g in gap_map[after_pos]:
                stripped = g.strip("\n")
                if stripped:
                    parts.append(stripped)

    remaining_pos = len(section_order)
    for pos in sorted(gap_map):
        if pos > remaining_pos:
            for g in gap_map[pos]:
                stripped = g.strip("\n")
                if stripped:
                    parts.append(stripped)

    return "\n\n".join(parts) + "\n"


def get_next_version(compiled_dir):
    if not compiled_dir.exists():
        return 1
    existing = [d.name for d in compiled_dir.iterdir() if d.is_dir() and d.name.startswith("v")]
    if not existing:
        return 1
    nums = []
    for name in existing:
        try:
            nums.append(int(name[1:]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 1


def compile_skill(skill_name, bank, profile=None, dry_run=False, diff_mode=False):
    group_name, skill_cfg = find_skill(bank, skill_name)
    if group_name is None:
        print(f"Error: skill '{skill_name}' not found in bank.yaml", file=sys.stderr)
        return False

    skill_dir = BANK_DIR / group_name / skill_name
    if not skill_dir.exists():
        print(f"Error: skill directory not found: {skill_dir}", file=sys.stderr)
        return False

    manifest_path = skill_dir / "manifest.yaml"
    if not manifest_path.exists():
        print(f"Error: manifest.yaml not found in {skill_dir}", file=sys.stderr)
        return False

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f) or {}

    base_file = skill_dir / manifest.get("base", "base.md")
    if not base_file.exists():
        print(f"Error: base file not found: {base_file}", file=sys.stderr)
        return False

    base_text = base_file.read_text()
    frontmatter, body = parse_frontmatter(base_text)
    sections, gaps, section_order = parse_sections(body)

    if profile:
        profiles = manifest.get("profiles", {})
        if profile not in profiles:
            print(f"Error: profile '{profile}' not found in {manifest_path}", file=sys.stderr)
            return False
        active_ids = profiles[profile].get("active", [])
    else:
        active_ids = manifest.get("active", [])

    if not active_ids:
        compiled = reassemble(frontmatter, sections, section_order, gaps)
    else:
        patches_dir = skill_dir / "patches"
        patches = []
        warnings = []

        for pid in active_ids:
            patch_file = patches_dir / f"{pid}.md"
            if not patch_file.exists():
                print(f"Error: patch file not found: {patch_file}", file=sys.stderr)
                return False
            patch = load_patch(patch_file)
            if "id" not in patch:
                patch["id"] = pid

            status = patch.get("status", "active")
            if status == "archived":
                print(f"Error: patch '{pid}' is archived and cannot be activated", file=sys.stderr)
                return False
            if status == "deprecated":
                superseded = patch.get("superseded_by", "")
                warnings.append(f"Warning: patch '{pid}' is deprecated (superseded by '{superseded}')")

            patches.append(patch)

        for w in warnings:
            print(w, file=sys.stderr)

        check_conflicts(patches)
        check_cross_skill_deps(patches, bank)
        sorted_patches = topo_sort(patches)
        sections, section_order = apply_patches(sections, section_order, sorted_patches)
        compiled = reassemble(frontmatter, sections, section_order, gaps)

    output_path = resolve_output_path(bank, skill_cfg["output"])

    if diff_mode:
        if output_path.exists():
            current = output_path.read_text().splitlines(keepends=True)
            new = compiled.splitlines(keepends=True)
            diff = difflib.unified_diff(current, new, fromfile=str(output_path), tofile="compiled")
            diff_text = "".join(diff)
            if diff_text:
                print(diff_text)
            else:
                print(f"  {skill_name}: no changes")
        else:
            print(f"  {skill_name}: output file does not exist yet (new)")
        return True

    if dry_run:
        print(f"--- {skill_name} (dry-run) ---")
        print(compiled[:500])
        if len(compiled) > 500:
            print(f"... ({len(compiled)} chars total)")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(compiled)
    print(f"  {skill_name} -> {output_path}")
    return True


def save_snapshot(bank):
    compiled_dir = BANK_DIR / bank.get("settings", {}).get("compiled_dir", "compiled")
    version = get_next_version(compiled_dir)
    snapshot_dir = compiled_dir / f"v{version:03d}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    all_skills = get_all_skills(bank)
    for skill_name, (group_name, skill_cfg) in all_skills.items():
        output_path = resolve_output_path(bank, skill_cfg["output"])
        if output_path.exists():
            dest = snapshot_dir / group_name / f"{skill_name}.md"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, dest)

    print(f"  Snapshot saved: {snapshot_dir}")
    return snapshot_dir


def show_status(bank, group_filter=None):
    all_skills = get_all_skills(bank)
    current_group = None

    for skill_name, (group_name, skill_cfg) in sorted(all_skills.items(), key=lambda x: (x[1][0], x[0])):
        if group_filter and group_name != group_filter:
            continue

        if group_name != current_group:
            current_group = group_name
            print(f"\n[{group_name}]")

        skill_dir = BANK_DIR / group_name / skill_name
        manifest_path = skill_dir / "manifest.yaml"

        if not manifest_path.exists():
            print(f"  {skill_name}: no manifest")
            continue

        with open(manifest_path) as f:
            manifest = yaml.safe_load(f) or {}

        active = manifest.get("active", [])
        disabled = manifest.get("disabled", [])
        profiles = list(manifest.get("profiles", {}).keys())

        patches_dir = skill_dir / "patches"
        total_patches = len(list(patches_dir.glob("*.md"))) if patches_dir.exists() else 0

        status_parts = [f"{len(active)} active"]
        if disabled:
            status_parts.append(f"{len(disabled)} disabled")
        if total_patches > len(active) + len(disabled):
            other = total_patches - len(active) - len(disabled)
            status_parts.append(f"{other} other")
        if profiles:
            status_parts.append(f"profiles: {', '.join(profiles)}")

        print(f"  {skill_name}: {total_patches} patches ({', '.join(status_parts)})")


def squash_skill(skill_name, bank):
    group_name, skill_cfg = find_skill(bank, skill_name)
    if group_name is None:
        print(f"Error: skill '{skill_name}' not found", file=sys.stderr)
        return False

    skill_dir = BANK_DIR / group_name / skill_name
    manifest_path = skill_dir / "manifest.yaml"

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f) or {}

    active_ids = manifest.get("active", [])
    if not active_ids:
        print(f"  {skill_name}: no active patches to squash")
        return True

    save_snapshot(bank)

    if not compile_skill(skill_name, bank):
        return False

    output_path = resolve_output_path(bank, skill_cfg["output"])
    compiled_text = output_path.read_text()

    base_file = skill_dir / manifest.get("base", "base.md")
    base_text = base_file.read_text()
    old_fm, _ = parse_frontmatter(base_text)

    fm_text = yaml.dump(old_fm, default_flow_style=False, allow_unicode=True).strip() if old_fm else ""
    new_base_lines = []
    if fm_text:
        new_base_lines.append(f"---\n{fm_text}\n---\n")

    compiled_fm, compiled_body = parse_frontmatter(compiled_text)
    sections_from_compiled = compiled_body.split("\n\n")

    new_base_lines.append(compiled_body.strip())
    base_file.write_text("\n".join(new_base_lines) + "\n")

    patches_dir = skill_dir / "patches"
    for pid in active_ids:
        patch_file = patches_dir / f"{pid}.md"
        if patch_file.exists():
            text = patch_file.read_text()
            text = re.sub(r"(?m)^status:\s*active", "status: archived", text)
            patch_file.write_text(text)

    manifest["active"] = []
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, allow_unicode=True)

    print(f"  {skill_name}: squashed {len(active_ids)} patches into base.md")
    return True


def main():
    parser = argparse.ArgumentParser(description="Skill Bank Compiler")
    parser.add_argument("skill", nargs="?", help="Skill name to compile")
    parser.add_argument("-p", "--profile", help="Profile name")
    parser.add_argument("--group", help="Compile all skills in a group")
    parser.add_argument("--all", action="store_true", help="Compile all skills")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--diff", action="store_true", help="Show diff with current output")
    parser.add_argument("--status", action="store_true", help="Show patch status summary")
    parser.add_argument("--squash", action="store_true", help="Squash active patches into base")

    args = parser.parse_args()
    bank = load_bank_yaml()

    if args.status:
        show_status(bank, group_filter=args.group)
        return

    if args.squash:
        if not args.skill:
            print("Error: --squash requires a skill name", file=sys.stderr)
            sys.exit(1)
        ok = squash_skill(args.skill, bank)
        sys.exit(0 if ok else 1)

    skills_to_compile = []

    if args.all:
        all_skills = get_all_skills(bank)
        skills_to_compile = list(all_skills.keys())
    elif args.group:
        group_skills = get_skills_in_group(bank, args.group)
        if not group_skills:
            print(f"Error: group '{args.group}' not found or empty", file=sys.stderr)
            sys.exit(1)
        skills_to_compile = list(group_skills.keys())
    elif args.skill:
        skills_to_compile = [args.skill]
    else:
        parser.print_help()
        sys.exit(1)

    print(f"Compiling {len(skills_to_compile)} skill(s)...")
    success = True
    for skill_name in skills_to_compile:
        if not compile_skill(skill_name, bank, profile=args.profile, dry_run=args.dry_run, diff_mode=args.diff):
            success = False

    if not args.dry_run and not args.diff and success:
        save_snapshot(bank)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
