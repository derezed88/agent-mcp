import os
from typing import List, Dict, Optional, Tuple, Set
from config import log, SYSTEM_PROMPT_FILE

# ---------------------------------------------------------------------------
# SECTIONAL SYSTEM PROMPT DESIGN
# ---------------------------------------------------------------------------
# The system prompt is a tree of section files:
#
# .system_prompt                  ← root: main paragraph + [SECTIONS] list
#   .system_prompt_tools          ← leaf: body text  OR container with [SECTIONS]
#     .system_prompt_tool_url_extract   ← leaf: body text
#     .system_prompt_tool_db_query      ← leaf: body text
#   .system_prompt_behavior       ← leaf: body text
#
# Rules:
# - A section file is either a LEAF (body text) or a CONTAINER ([SECTIONS] list).
#   Not both. If [SECTIONS] is present, any text before it is ignored.
# - All sections at all depths are registered in the flat _sections list.
# - Duplicate section names across the tree are rejected (loop detection also
#   catches cross-branch reuse of the same name).
# - Loop detection: if a section name appears in its own ancestor chain,
#   it is skipped and an error placeholder is inserted.
#
# LLM tools:
# - update_system_prompt(section_name, operation, ...) edits specific sections
# - read_system_prompt() returns full assembled prompt
# - read_system_prompt(section) returns specific section
# ---------------------------------------------------------------------------

DEFAULT_MAIN_PROMPT = """\
You are Robot, an AI assistant with persistent memory via a MySQL database (mymcp).
You are an autonomous agent. When you need information, you MUST use a tool call. Do not explain your steps. Do not provide Markdown code blocks. If you have a tool available for a task, use it immediately.

[SECTIONS]
memory-hierarchy: Staged Memory Hierarchy (PDDS chain)
tool-guardrails: Refined Tool Usage Guardrails
tools: Available Tool Definitions
tool-logging: Tool Usage Logging & Review
time-bypass: Direct Time Bypass
db-guardrails: Refined db_query Guardrails
behavior: Behaviour rules
"""

# In-memory cache of prompt structure.
# Flat list — all sections at all depths, in tree-traversal order.
# Each entry: {short-section-name, description, body, depth, parent}
_main_paragraph: str = ""
_sections: List[Dict] = []
_cached_full_prompt: Optional[str] = None


def _get_section_file_path(section_name: str) -> str:
    """Return the file path for a given section name."""
    base_dir = os.path.dirname(SYSTEM_PROMPT_FILE)
    return os.path.join(base_dir, f".system_prompt_{section_name}")


def _parse_sections_block(content: str) -> Tuple[Optional[List[Tuple[str, str]]], str]:
    """
    Parse content that may contain a [SECTIONS] marker.

    Returns:
      (section_list, body_text)
      - If [SECTIONS] found: section_list = [(name, desc), ...], body_text = ""
      - If no [SECTIONS]: section_list = None, body_text = content
    """
    lines = content.split('\n')
    sections_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == '[SECTIONS]':
            sections_idx = i
            break

    if sections_idx == -1:
        return None, content

    section_list = []
    for line in lines[sections_idx + 1:]:
        line = line.strip()
        if not line:
            continue
        if ':' in line:
            parts = line.split(':', 1)
            short_name = parts[0].strip()
            description = parts[1].strip()
            section_list.append((short_name, description))

    return section_list, ""


def _load_section_recursive(
    section_name: str,
    description: str,
    depth: int,
    parent: Optional[str],
    ancestors: Set[str],
    all_seen: Set[str],
) -> List[Dict]:
    """
    Load a section file recursively.

    Returns a flat list of section dicts in tree-traversal order.
    Each dict: {short-section-name, description, body, depth, parent}

    ancestors: set of section names in the current call stack (loop detection)
    all_seen: set of all section names registered so far (duplicate detection)
    """
    # Loop detection
    if section_name in ancestors:
        log.error(
            f"Circular reference detected: '{section_name}' is already in ancestor chain "
            f"{sorted(ancestors)}. Skipping."
        )
        return [{
            'short-section-name': section_name,
            'description': description,
            'body': f"[SECTION ERROR: circular reference to '{section_name}']",
            'depth': depth,
            'parent': parent,
        }]

    # Duplicate name detection (same name in two different branches)
    if section_name in all_seen:
        log.error(
            f"Duplicate section name '{section_name}' detected in prompt tree. "
            f"Each section name must be unique. Skipping second occurrence."
        )
        return [{
            'short-section-name': section_name,
            'description': description,
            'body': f"[SECTION ERROR: duplicate section name '{section_name}']",
            'depth': depth,
            'parent': parent,
        }]

    all_seen.add(section_name)

    file_path = _get_section_file_path(section_name)
    if not os.path.exists(file_path):
        log.warning(f"Section file not found: {file_path}")
        return [{
            'short-section-name': section_name,
            'description': description,
            'body': "",
            'depth': depth,
            'parent': parent,
        }]

    try:
        with open(file_path, 'r', encoding='utf-8') as fh:
            raw = fh.read()
    except Exception as exc:
        log.warning(f"Could not read section file {file_path}: {exc}")
        return [{
            'short-section-name': section_name,
            'description': description,
            'body': "",
            'depth': depth,
            'parent': parent,
        }]

    # Strip the ## header line if present
    lines = raw.split('\n', 1)
    content = lines[1] if len(lines) > 1 and lines[0].startswith('## ') else raw

    # Check if this is a container (has [SECTIONS]) or a leaf (body text)
    sub_section_list, body = _parse_sections_block(content)

    if sub_section_list is None:
        # Leaf node — plain body text
        return [{
            'short-section-name': section_name,
            'description': description,
            'body': body,
            'depth': depth,
            'parent': parent,
        }]
    else:
        # Container node — recurse into children
        # The container itself gets an empty body; its content comes from children
        result = [{
            'short-section-name': section_name,
            'description': description,
            'body': "",
            'depth': depth,
            'parent': parent,
            'is_container': True,
        }]
        new_ancestors = ancestors | {section_name}
        for child_name, child_desc in sub_section_list:
            child_sections = _load_section_recursive(
                child_name, child_desc, depth + 1, section_name,
                new_ancestors, all_seen
            )
            result.extend(child_sections)
        return result


def _parse_main_prompt(content: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Parse main .system_prompt file into:
    - main_paragraph (before [SECTIONS])
    - section_list [(short-name, description), ...]
    """
    lines = content.split('\n')
    sections_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == '[SECTIONS]':
            sections_idx = i
            break

    if sections_idx == -1:
        return content.strip(), []

    main_paragraph = '\n'.join(lines[:sections_idx]).strip()
    section_list = []
    for line in lines[sections_idx + 1:]:
        line = line.strip()
        if not line:
            continue
        if ':' in line:
            parts = line.split(':', 1)
            short_name = parts[0].strip()
            description = parts[1].strip()
            section_list.append((short_name, description))

    return main_paragraph, section_list


def load_system_prompt() -> str:
    """
    Load the system prompt from disk recursively.
    Builds a flat _sections list from the full section tree.
    """
    global _main_paragraph, _sections, _cached_full_prompt

    if not os.path.exists(SYSTEM_PROMPT_FILE):
        try:
            with open(SYSTEM_PROMPT_FILE, 'w', encoding='utf-8') as fh:
                fh.write(DEFAULT_MAIN_PROMPT)
        except Exception as exc:
            log.warning(f"Could not write default prompt: {exc}")
        content = DEFAULT_MAIN_PROMPT
    else:
        try:
            with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as fh:
                content = fh.read()
        except Exception as exc:
            log.warning(f"Could not read prompt: {exc}")
            content = DEFAULT_MAIN_PROMPT

    main_paragraph, section_list = _parse_main_prompt(content)
    _main_paragraph = main_paragraph

    all_seen: Set[str] = set()
    _sections = []
    for short_name, description in section_list:
        sections = _load_section_recursive(
            short_name, description, depth=0, parent=None,
            ancestors=set(), all_seen=all_seen
        )
        _sections.extend(sections)

    _cached_full_prompt = _assemble_full_prompt()

    total = len(_sections)
    leaves = sum(1 for s in _sections if not s.get('is_container'))
    log.info(
        f"Loaded system prompt: {len(_main_paragraph)} chars main, "
        f"{total} sections ({leaves} leaf, {total - leaves} container)"
    )
    return _cached_full_prompt


def _assemble_full_prompt() -> str:
    """Assemble the full prompt from main paragraph and flat sections list."""
    parts = [_main_paragraph]
    for section in _sections:
        depth = section.get('depth', 0)
        prefix = "##" + "#" * depth  # ## at depth 0, ### at depth 1, etc.
        header = f"\n\n{prefix} {section['short-section-name']}: {section['description']}"
        body = section['body']
        if body:
            parts.append(header + "\n" + body)
        else:
            parts.append(header)
    return '\n'.join(parts)


def get_current_prompt() -> str:
    """Return the currently cached full system prompt."""
    global _cached_full_prompt
    if _cached_full_prompt is None:
        return load_system_prompt()
    return _cached_full_prompt


def get_section(identifier: str) -> Optional[str]:
    """
    Get a specific section by index (e.g. "0") or name (e.g. "tool_url_extract").
    Searches all sections at all depths.
    Returns section content with header, or None if not found.
    """
    if _cached_full_prompt is None:
        load_system_prompt()

    try:
        idx = int(identifier)
        if 0 <= idx < len(_sections):
            section = _sections[idx]
            depth = section.get('depth', 0)
            prefix = "##" + "#" * depth
            header = f"{prefix} {section['short-section-name']}: {section['description']}"
            return header + "\n" + section['body']
        return None
    except ValueError:
        pass

    for section in _sections:
        if section['short-section-name'] == identifier:
            depth = section.get('depth', 0)
            prefix = "##" + "#" * depth
            header = f"{prefix} {section['short-section-name']}: {section['description']}"
            return header + "\n" + section['body']

    return None


def list_sections() -> List[Dict]:
    """Return metadata for all sections at all depths (without bodies)."""
    if _cached_full_prompt is None:
        load_system_prompt()

    return [
        {
            'index': i,
            'short-section-name': s['short-section-name'],
            'description': s['description'],
            'depth': s.get('depth', 0),
            'parent': s.get('parent'),
            'is_container': s.get('is_container', False),
        }
        for i, s in enumerate(_sections)
    ]


def _write_section_file(section_name: str, content: str) -> None:
    """Write content to a section file, including the ## header."""
    file_path = _get_section_file_path(section_name)

    description = ""
    for section in _sections:
        if section['short-section-name'] == section_name:
            description = section['description']
            break

    header = f"## {section_name}: {description}\n"
    full_content = header + content

    try:
        with open(file_path, 'w', encoding='utf-8') as fh:
            fh.write(full_content)
    except Exception as exc:
        log.error(f"Could not write section file {file_path}: {exc}")
        raise


def apply_prompt_operation(
    section_name: str,
    operation: str,
    content: str = "",
    target: str = "",
    confirm_overwrite: bool = False,
) -> Tuple[str, str]:
    """
    Perform a surgical edit on a specific system prompt section.
    Works on any section at any depth in the tree.

    Returns (new_section_text, status_message).
    Raises ValueError for invalid operations or missing arguments.
    """
    global _sections, _cached_full_prompt

    section_idx = -1
    for i, section in enumerate(_sections):
        if section['short-section-name'] == section_name:
            section_idx = i
            break

    if section_idx == -1:
        available = ', '.join(s['short-section-name'] for s in _sections)
        raise ValueError(
            f"Section '{section_name}' not found. "
            f"Available sections: {available}"
        )

    # Prevent edits to container sections (they have no body — only [SECTIONS])
    if _sections[section_idx].get('is_container'):
        raise ValueError(
            f"Section '{section_name}' is a container (has sub-sections) and has no editable body. "
            f"Edit its child sections instead."
        )

    current_body = _sections[section_idx]['body']
    op = operation.strip().lower()

    if op == "append":
        if not content:
            raise ValueError("'append' requires non-empty content.")
        separator = "\n" if current_body.endswith("\n") else "\n\n"
        new_body = current_body + separator + content
        msg = f"Appended {len(content)} chars to section '{section_name}'."

    elif op == "prepend":
        if not content:
            raise ValueError("'prepend' requires non-empty content.")
        separator = "\n\n" if not content.endswith("\n") else ""
        new_body = content + separator + current_body
        msg = f"Prepended {len(content)} chars to section '{section_name}'."

    elif op == "replace":
        if not target:
            raise ValueError("'replace' requires a non-empty target string.")
        if target not in current_body:
            raise ValueError(
                f"'replace' target not found in section '{section_name}'. "
                f"Target (first 80 chars): {target[:80]!r}"
            )
        new_body = current_body.replace(target, content, 1)
        msg = f"Replaced target in section '{section_name}'."

    elif op == "delete":
        if not target:
            raise ValueError("'delete' requires a non-empty target string.")
        lines = current_body.splitlines(keepends=True)
        filtered = [ln for ln in lines if target not in ln]
        removed = len(lines) - len(filtered)
        if removed == 0:
            raise ValueError(
                f"'delete' target not found in section '{section_name}'. "
                f"Target: {target[:80]!r}"
            )
        new_body = "".join(filtered)
        msg = f"Deleted {removed} line(s) from section '{section_name}'."

    elif op == "overwrite":
        if not confirm_overwrite:
            raise ValueError(
                "'overwrite' requires confirm_overwrite=true. "
                "This replaces the ENTIRE section."
            )
        if not content:
            raise ValueError("'overwrite' requires non-empty content.")
        new_body = content
        msg = f"Full overwrite of section '{section_name}' ({len(new_body)} chars)."

    else:
        raise ValueError(
            f"Unknown operation {operation!r}. "
            "Valid operations: append, prepend, replace, delete, overwrite."
        )

    _sections[section_idx]['body'] = new_body
    _write_section_file(section_name, new_body)
    _cached_full_prompt = _assemble_full_prompt()

    log.info(f"apply_prompt_operation({section_name}, {op}): {msg}")
    return new_body, msg
