"""
plan_engine.py — Two-tier plan decomposition and execution engine.

Lifecycle:
  1. Concept steps created by human or reasoning model (step_type='concept')
  2. Decomposer (Haiku 4.5) breaks concept steps into task steps (step_type='task')
  3. Executor picks up approved task steps and runs tool_call specs
  4. Parent concept steps auto-complete when all child tasks are done
  5. Goal auto-completes when all concept steps are done

Supports goal-less plans (goal_id=0) for ad-hoc work.
"""

import asyncio
import json
import logging

log = logging.getLogger("plan_engine")

# ---------------------------------------------------------------------------
# Table helpers — uses same prefix pattern as memory.py
# ---------------------------------------------------------------------------

def _PLANS():
    from memory import _PLANS as _P
    return _P()

def _GOALS():
    from memory import _GOALS as _G
    return _G()

# ---------------------------------------------------------------------------
# Decomposer system prompt
# ---------------------------------------------------------------------------

_DECOMPOSER_SYSTEM = """\
You are a plan decomposition engine. Your job is to take a concept step (a
human-readable description of intent) and break it into discrete, executable
task steps.

You will receive:
- The concept step description
- The list of available tools and their argument schemas
- Optional context (goal title, other concept steps for sequencing awareness)

For EACH task step you produce, output:
- description: short human-readable label for what this step does
- tool_call: JSON object with "tool" and "args" keys, or null if not tool-executable
- target: "model" if tool_call is populated, "human" if it requires human action,
  "investigate" if you cannot determine how to execute it with current tools
- reason: brief explanation of why this target was chosen (only for human/investigate)

Rules:
1. Each task step must be ONE atomic action — one tool call or one human action.
2. Prefer existing tools. Do not invent tool names.
3. If the concept step says "human:" or explicitly names a person, set target="human".
4. If unsure whether the system can do it, set target="investigate" — do NOT guess.
5. Order task steps logically (they will be assigned ascending step_order).
6. Keep descriptions concise but unambiguous.
7. For db_query tool calls, write the actual SQL in args.query.
8. If a concept step is already atomic (one tool call), emit exactly one task step.

Respond with ONLY a JSON array. No markdown, no explanation outside the JSON.

Example output:
[
  {
    "description": "Query current API endpoint list from config table",
    "tool_call": {"tool": "db_query", "args": {"query": "SELECT * FROM api_endpoints WHERE active=1"}},
    "target": "model"
  },
  {
    "description": "Review endpoint list and decide which to monitor",
    "target": "human",
    "reason": "Requires human judgment on monitoring priorities"
  },
  {
    "description": "Research best uptime monitoring approach for our stack",
    "target": "investigate",
    "reason": "Need to evaluate available monitoring tools and integrations"
  }
]
"""

# ---------------------------------------------------------------------------
# Build tool catalog for decomposer context
# ---------------------------------------------------------------------------

def _build_tool_catalog() -> str:
    """
    Build a compact tool catalog string for the decomposer.
    Lists tool names and their parameter schemas.
    """
    try:
        from tools import get_all_openai_tools
        all_tools = get_all_openai_tools()
        lines = []
        for td in all_tools:
            fn = td.get("function", {})
            name = fn.get("name", "?")
            desc = fn.get("description", "")[:120]
            params = fn.get("parameters", {}).get("properties", {})
            param_names = list(params.keys())
            lines.append(f"- {name}({', '.join(param_names)}): {desc}")
        return "\n".join(lines)
    except Exception as e:
        log.warning(f"_build_tool_catalog failed: {e}")
        return "(tool catalog unavailable)"


# ---------------------------------------------------------------------------
# Decompose a concept step into task steps
# ---------------------------------------------------------------------------

async def decompose_concept_step(
    concept_step_id: int,
    model_key: str = "plan-decomposer",
    auto_approve: bool = False,
) -> list[dict]:
    """
    Takes a concept step and decomposes it into task steps via LLM.

    Returns list of created task step dicts (id, description, target, tool_call, etc.)
    """
    from database import fetch_dicts, execute_insert
    from config import LLM_REGISTRY
    from agents import _build_lc_llm, _content_to_str
    from langchain_core.messages import SystemMessage, HumanMessage

    # ---- Load the concept step ----
    rows = await fetch_dicts(
        f"SELECT p.*, g.title as goal_title, g.description as goal_description "
        f"FROM {_PLANS()} p "
        f"LEFT JOIN {_GOALS()} g ON g.id = p.goal_id "
        f"WHERE p.id = {concept_step_id} AND p.step_type = 'concept' "
        f"LIMIT 1"
    )
    if not rows:
        raise ValueError(f"Concept step id={concept_step_id} not found or not a concept step")

    concept = rows[0]
    goal_id = concept["goal_id"]
    goal_title = concept.get("goal_title", "") or ""
    goal_desc = concept.get("goal_description", "") or ""

    # ---- Load sibling concept steps for context ----
    siblings = await fetch_dicts(
        f"SELECT step_order, description, status FROM {_PLANS()} "
        f"WHERE goal_id = {goal_id} AND step_type = 'concept' AND id != {concept_step_id} "
        f"ORDER BY step_order"
    ) or []

    # ---- Build the user prompt ----
    tool_catalog = _build_tool_catalog()

    context_parts = []
    if goal_title:
        context_parts.append(f"Goal: {goal_title}")
    if goal_desc:
        context_parts.append(f"Goal description: {goal_desc}")
    if siblings:
        sib_lines = [
            f"  step {s['step_order']}: {s['description']} [{s['status']}]"
            for s in siblings
        ]
        context_parts.append("Other steps in this plan:\n" + "\n".join(sib_lines))

    context_block = "\n".join(context_parts) if context_parts else "(no additional context)"

    user_prompt = f"""\
CONCEPT STEP TO DECOMPOSE:
{concept["description"]}

CONTEXT:
{context_block}

AVAILABLE TOOLS:
{tool_catalog}

Decompose the concept step into discrete task steps. Output JSON array only."""

    # ---- Call the decomposer LLM ----
    if model_key not in LLM_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}")

    cfg = LLM_REGISTRY[model_key]
    timeout = cfg.get("llm_call_timeout", 90)
    llm = _build_lc_llm(model_key)
    msgs = [SystemMessage(content=_DECOMPOSER_SYSTEM), HumanMessage(content=user_prompt)]

    try:
        response = await asyncio.wait_for(llm.ainvoke(msgs), timeout=timeout)
        raw = _content_to_str(response.content)
    except Exception as e:
        log.error(f"decompose_concept_step: LLM call failed: {e}")
        raise

    # ---- Parse the response ----
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])

    try:
        tasks = json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.error(f"decompose_concept_step: JSON parse failed: {e}\nRaw: {raw[:500]}")
        raise ValueError(f"Decomposer returned invalid JSON: {e}")

    if not isinstance(tasks, list):
        raise ValueError(f"Decomposer returned {type(tasks).__name__}, expected list")

    # ---- Insert task steps into DB ----
    created = []
    approval = "approved" if auto_approve else "proposed"

    for i, task in enumerate(tasks, start=1):
        desc = str(task.get("description", "")).replace("'", "''")
        target = task.get("target", "model")
        if target not in ("model", "human", "investigate"):
            target = "investigate"

        tool_call_raw = task.get("tool_call")
        tool_call_sql = "NULL"
        if tool_call_raw and isinstance(tool_call_raw, dict):
            tc_json = json.dumps(tool_call_raw).replace("'", "''")
            tool_call_sql = f"'{tc_json}'"
        elif target == "model":
            # Model target but no tool_call — downgrade to investigate
            target = "investigate"

        reason = str(task.get("reason", "")).replace("'", "''")
        if reason:
            desc_with_reason = f"{desc} — {reason}" if target in ("human", "investigate") else desc
        else:
            desc_with_reason = desc

        executor = "NULL"
        if target == "model":
            executor = "'plan-executor'"

        sql = (
            f"INSERT INTO {_PLANS()} "
            f"(goal_id, step_order, description, status, step_type, parent_id, "
            f"target, executor, tool_call, approval, source, session_id) "
            f"VALUES ({goal_id}, {i}, '{desc_with_reason}', 'pending', 'task', {concept_step_id}, "
            f"'{target}', {executor}, {tool_call_sql}, '{approval}', 'assistant', NULL)"
        )
        try:
            row_id = await execute_insert(sql)
            created.append({
                "id": row_id,
                "step_order": i,
                "description": task.get("description", ""),
                "target": target,
                "tool_call": tool_call_raw,
                "approval": approval,
            })
        except Exception as e:
            log.error(f"decompose_concept_step: insert failed for task {i}: {e}")

    # ---- Mark concept step as in_progress (decomposition done, awaiting execution) ----
    if created:
        from database import execute_sql
        await execute_sql(
            f"UPDATE {_PLANS()} SET status = 'in_progress' WHERE id = {concept_step_id}"
        )

    log.info(
        f"decompose_concept_step: concept={concept_step_id} → {len(created)} task steps "
        f"(approval={approval})"
    )
    return created


# ---------------------------------------------------------------------------
# Approve / reject proposed task steps
# ---------------------------------------------------------------------------

async def approve_plan(concept_step_id: int, approve: bool = True) -> str:
    """
    Approve or reject all proposed task steps under a concept step.
    """
    from database import execute_sql, fetch_dicts

    new_approval = "approved" if approve else "rejected"
    await execute_sql(
        f"UPDATE {_PLANS()} SET approval = '{new_approval}' "
        f"WHERE parent_id = {concept_step_id} AND approval = 'proposed'"
    )
    rows = await fetch_dicts(
        f"SELECT id FROM {_PLANS()} WHERE parent_id = {concept_step_id}"
    )
    count = len(rows) if rows else 0
    return f"{count} task steps {new_approval} for concept step id={concept_step_id}"


# ---------------------------------------------------------------------------
# Execute a single task step
# ---------------------------------------------------------------------------

async def execute_task_step(task_step_id: int) -> str:
    """
    Execute a single approved task step using three-tier execution:

    1. Direct tool call — call the Python executor function directly (zero LLM cost).
    2. Primary executor — LLM-based execution via model_roles["plan_executor"].
    3. Fallback executor — LLM-based execution via model_roles["plan_executor_fallback"].

    The `executor` column is written with the actual execution method for audit trail:
      "direct"       — tier 1 succeeded
      "<model_key>"  — tier 2 or 3 LLM model that executed it

    Returns the result text.
    """
    from database import fetch_dicts, execute_sql
    from tools import get_tool_executor
    from state import current_client_id

    rows = await fetch_dicts(
        f"SELECT * FROM {_PLANS()} WHERE id = {task_step_id} AND step_type = 'task' LIMIT 1"
    )
    if not rows:
        return f"Task step id={task_step_id} not found"

    step = rows[0]
    if step["approval"] != "approved":
        return f"Task step id={task_step_id} is not approved (approval={step['approval']})"
    if step["target"] != "model":
        return f"Task step id={task_step_id} target={step['target']} — cannot auto-execute"
    if step["status"] in ("done", "skipped"):
        return f"Task step id={task_step_id} already {step['status']}"

    # Mark in_progress
    await execute_sql(
        f"UPDATE {_PLANS()} SET status = 'in_progress' WHERE id = {task_step_id}"
    )

    tool_call_raw = step.get("tool_call")
    if not tool_call_raw:
        await execute_sql(
            f"UPDATE {_PLANS()} SET status = 'pending', "
            f"result = 'ERROR: no tool_call defined' WHERE id = {task_step_id}"
        )
        return f"Task step id={task_step_id} has no tool_call"

    # Parse tool_call
    if isinstance(tool_call_raw, str):
        try:
            tool_call = json.loads(tool_call_raw)
        except json.JSONDecodeError:
            err = f"Invalid tool_call JSON: {tool_call_raw[:200]}"
            await execute_sql(
                f"UPDATE {_PLANS()} SET status = 'pending', "
                f"result = '{err.replace(chr(39), chr(39)*2)}' WHERE id = {task_step_id}"
            )
            return err
    else:
        tool_call = tool_call_raw

    tool_name = tool_call.get("tool", "")
    tool_args = tool_call.get("args", {})

    # ------------------------------------------------------------------
    # TIER 1: Direct tool call (zero LLM cost)
    # ------------------------------------------------------------------
    executor_fn = get_tool_executor(tool_name)
    if not executor_fn:
        err = f"Unknown tool: {tool_name}"
        await _fail_task(task_step_id, err)
        return err

    direct_err = None
    mapped_args = _map_tool_args(executor_fn, tool_args, tool_name)

    try:
        if isinstance(mapped_args, dict):
            result = await executor_fn(**mapped_args)
        else:
            result = await executor_fn()
        result_str = str(result) if result else "(no output)"

        # Direct call succeeded
        await _complete_task(task_step_id, result_str, executor="direct")
        parent_id = step.get("parent_id")
        if parent_id:
            await _check_parent_completion(parent_id)
        log.info(f"execute_task_step: id={task_step_id} tool={tool_name} executor=direct → done")
        return result_str

    except Exception as e:
        direct_err = str(e)
        log.warning(
            f"execute_task_step: id={task_step_id} tool={tool_name} direct failed: {direct_err[:200]}"
        )

    # ------------------------------------------------------------------
    # TIER 2 & 3: LLM executor with failover
    # ------------------------------------------------------------------
    llm_result = await _try_llm_executors(
        task_step_id, step, tool_name, tool_args, direct_err
    )
    if llm_result is not None:
        parent_id = step.get("parent_id")
        if parent_id:
            await _check_parent_completion(parent_id)
        return llm_result

    # All tiers exhausted — fail the task
    err = f"All executors failed. Direct: {direct_err}"
    await _fail_task(task_step_id, err)
    return err


def _map_tool_args(executor_fn, tool_args: dict, tool_name: str) -> dict:
    """Map tool_call args to executor function parameter names."""
    if not isinstance(tool_args, dict):
        return tool_args
    import inspect
    try:
        sig = inspect.signature(executor_fn)
        valid_params = set(sig.parameters.keys())
        mapped = {}
        for k, v in tool_args.items():
            if k in valid_params:
                mapped[k] = v
            else:
                _synonyms = {
                    "query": "sql", "sql": "query",
                    "text": "content", "content": "text",
                    "message": "prompt", "prompt": "message",
                }
                alt = _synonyms.get(k, "")
                if alt and alt in valid_params:
                    mapped[alt] = v
                    log.info(f"_map_tool_args: mapped '{k}' → '{alt}' for {tool_name}")
                else:
                    mapped[k] = v
        return mapped
    except (ValueError, TypeError):
        return tool_args


async def _try_llm_executors(
    task_step_id: int,
    step: dict,
    tool_name: str,
    tool_args: dict,
    direct_err: str | None,
) -> str | None:
    """
    Try primary then fallback LLM executor models.
    Returns result string on success, None if all fail.
    """
    from config import get_model_role

    executor_roles = ["plan_executor", "plan_executor_fallback"]
    last_err = ""

    for role in executor_roles:
        try:
            model_key = get_model_role(role)
        except KeyError:
            log.debug(f"_try_llm_executors: role '{role}' not configured, skipping")
            continue

        log.info(
            f"_try_llm_executors: id={task_step_id} tool={tool_name} "
            f"trying {role}={model_key}"
        )

        result = await _llm_execute_tool(model_key, tool_name, tool_args, step, direct_err)
        if result is not None:
            await _complete_task(task_step_id, result, executor=model_key)
            log.info(
                f"execute_task_step: id={task_step_id} tool={tool_name} "
                f"executor={model_key} ({role}) → done"
            )
            return result
        else:
            last_err = f"{role}={model_key} failed"
            log.warning(f"_try_llm_executors: {last_err}")

    return None


async def _llm_execute_tool(
    model_key: str,
    tool_name: str,
    tool_args: dict,
    step: dict,
    direct_err: str | None,
) -> str | None:
    """
    Execute a tool call via an LLM model using llm_call(mode='tool').
    Returns the result string on success, None on failure.
    """
    from agents import llm_call

    # Build a prompt that tells the executor what to do
    args_json = json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)
    context = step.get("description", "")
    err_note = f"\nNote: direct execution failed with: {direct_err}" if direct_err else ""

    prompt = (
        f"Execute this tool call:\n"
        f"Tool: {tool_name}\n"
        f"Arguments: {args_json}\n"
        f"Context: {context}{err_note}\n\n"
        f"Call the {tool_name} tool with the arguments above and return the result."
    )

    try:
        result = await llm_call(
            model=model_key,
            prompt=prompt,
            mode="tool",
            sys_prompt="target",
            history="none",
            tool=tool_name,
        )
        # llm_call returns "ERROR: ..." on failure
        if result and result.startswith("ERROR:"):
            log.warning(f"_llm_execute_tool: {model_key}/{tool_name} returned: {result[:200]}")
            return None
        return result or None
    except Exception as e:
        log.error(f"_llm_execute_tool: {model_key}/{tool_name} exception: {e}")
        return None


async def _complete_task(task_step_id: int, result_str: str, executor: str = "direct"):
    """Mark a task step as done with result and executor audit trail."""
    from database import execute_sql
    result_escaped = result_str[:4000].replace("'", "''")
    executor_escaped = executor.replace("'", "''")
    await execute_sql(
        f"UPDATE {_PLANS()} SET status = 'done', "
        f"result = '{result_escaped}', "
        f"executor = '{executor_escaped}' "
        f"WHERE id = {task_step_id}"
    )


async def _fail_task(task_step_id: int, error: str):
    """
    Mark a task step as failed and propagate failure up the stack.

    Cascade:
      task → status='pending', result=error
      parent concept → status='blocked' (if any child failed)
      goal → status='blocked' (if any concept is blocked)
    """
    from database import execute_sql, fetch_dicts
    escaped = error[:2000].replace("'", "''")
    await execute_sql(
        f"UPDATE {_PLANS()} SET status = 'pending', "
        f"result = '{escaped}' WHERE id = {task_step_id}"
    )

    # Propagate to parent concept step
    rows = await fetch_dicts(
        f"SELECT parent_id FROM {_PLANS()} WHERE id = {task_step_id} LIMIT 1"
    )
    if rows and rows[0].get("parent_id"):
        parent_id = rows[0]["parent_id"]
        await _check_parent_failure(parent_id, error)

    log.warning(f"_fail_task: task {task_step_id} failed: {error[:100]}")


async def _check_parent_failure(parent_id: int, error: str):
    """
    When a task step fails, mark the parent concept step as blocked
    and propagate up to the goal.
    """
    from database import execute_sql, fetch_dicts

    # Mark concept step as blocked with the error
    escaped = error[:2000].replace("'", "''")
    await execute_sql(
        f"UPDATE {_PLANS()} SET status = 'pending', "
        f"result = CONCAT(COALESCE(result, ''), ' | BLOCKED: {escaped}') "
        f"WHERE id = {parent_id} AND status IN ('pending', 'in_progress')"
    )
    log.info(f"_check_parent_failure: concept {parent_id} blocked by task failure")

    # Propagate to goal
    goal_rows = await fetch_dicts(
        f"SELECT goal_id FROM {_PLANS()} WHERE id = {parent_id} LIMIT 1"
    )
    if goal_rows and goal_rows[0].get("goal_id"):
        goal_id = goal_rows[0]["goal_id"]
        # Only block goal if it was active — don't downgrade already-blocked
        await execute_sql(
            f"UPDATE {_GOALS()} SET status = 'blocked' "
            f"WHERE id = {goal_id} AND status = 'active'"
        )
        log.info(f"_check_parent_failure: goal {goal_id} blocked due to task failure")


# ---------------------------------------------------------------------------
# Auto-completion logic
# ---------------------------------------------------------------------------

async def _check_parent_completion(parent_id: int):
    """
    If all task steps under a concept step are done/skipped,
    mark the concept step as done. Then check if the goal is complete.
    """
    from database import fetch_dicts, execute_sql

    remaining = await fetch_dicts(
        f"SELECT COUNT(*) as cnt FROM {_PLANS()} "
        f"WHERE parent_id = {parent_id} AND status NOT IN ('done', 'skipped')"
    )
    if remaining and remaining[0]["cnt"] == 0:
        await execute_sql(
            f"UPDATE {_PLANS()} SET status = 'done' WHERE id = {parent_id}"
        )
        log.info(f"_check_parent_completion: concept step id={parent_id} auto-completed")

        # Check if goal is now complete
        goal_rows = await fetch_dicts(
            f"SELECT goal_id FROM {_PLANS()} WHERE id = {parent_id} LIMIT 1"
        )
        if goal_rows and goal_rows[0]["goal_id"]:
            await _check_goal_completion(goal_rows[0]["goal_id"])


async def _check_goal_completion(goal_id: int):
    """
    If all concept steps for a goal are done/skipped,
    mark the goal as done.
    """
    from database import fetch_dicts, execute_sql

    remaining = await fetch_dicts(
        f"SELECT COUNT(*) as cnt FROM {_PLANS()} "
        f"WHERE goal_id = {goal_id} AND step_type = 'concept' "
        f"AND status NOT IN ('done', 'skipped')"
    )
    if remaining and remaining[0]["cnt"] == 0:
        await execute_sql(
            f"UPDATE {_GOALS()} SET status = 'done', "
            f"auto_process_status = CASE "
            f"  WHEN auto_process_status IS NOT NULL THEN 'completed' "
            f"  ELSE auto_process_status END "
            f"WHERE id = {goal_id} AND status = 'active'"
        )
        log.info(f"_check_goal_completion: goal id={goal_id} auto-completed")


# ---------------------------------------------------------------------------
# Execute all pending approved task steps for a goal (or all goals)
# ---------------------------------------------------------------------------

async def execute_pending_tasks(
    goal_id: int | None = None,
    max_steps: int = 10,
) -> list[dict]:
    """
    Execute pending approved model-targeted task steps.
    Returns list of {id, tool, result, status} for each executed step.
    """
    from database import fetch_dicts

    where = (
        f"WHERE step_type = 'task' AND status = 'pending' "
        f"AND approval = 'approved' AND target = 'model'"
    )
    if goal_id is not None:
        where += f" AND goal_id = {goal_id}"

    rows = await fetch_dicts(
        f"SELECT id FROM {_PLANS()} {where} "
        f"ORDER BY goal_id, step_order LIMIT {max_steps}"
    )
    if not rows:
        return []

    results = []
    for row in rows:
        step_id = row["id"]
        try:
            result = await execute_task_step(step_id)
            results.append({"id": step_id, "status": "done", "result": result[:500]})
        except Exception as e:
            results.append({"id": step_id, "status": "error", "result": str(e)[:500]})

    return results


# ---------------------------------------------------------------------------
# Decompose all pending concept steps for a goal (or all goals)
# ---------------------------------------------------------------------------

async def decompose_pending_concepts(
    goal_id: int | None = None,
    model_key: str = "plan-decomposer",
    auto_approve: bool = False,
) -> list[dict]:
    """
    Find concept steps that have no child task steps and decompose them.
    Returns summary of decomposition results.
    """
    from database import fetch_dicts

    where = (
        f"WHERE p.step_type = 'concept' AND p.status = 'pending' "
        f"AND NOT EXISTS ("
        f"  SELECT 1 FROM {_PLANS()} c WHERE c.parent_id = p.id"
        f")"
    )
    if goal_id is not None:
        where += f" AND p.goal_id = {goal_id}"

    rows = await fetch_dicts(
        f"SELECT p.id, p.description FROM {_PLANS()} p {where} "
        f"ORDER BY p.goal_id, p.step_order"
    )
    if not rows:
        return []

    results = []
    for row in rows:
        try:
            tasks = await decompose_concept_step(
                row["id"], model_key=model_key, auto_approve=auto_approve
            )
            results.append({
                "concept_id": row["id"],
                "description": row["description"][:100],
                "task_count": len(tasks),
                "tasks": tasks,
            })
        except Exception as e:
            results.append({
                "concept_id": row["id"],
                "description": row["description"][:100],
                "error": str(e),
            })

    return results


# ---------------------------------------------------------------------------
# Full pipeline: decompose → approve → execute (for automated runs)
# ---------------------------------------------------------------------------

async def run_plan_pipeline(
    goal_id: int | None = None,
    model_key: str = "plan-decomposer",
    auto_approve: bool = False,
    max_exec_steps: int = 10,
) -> dict:
    """
    Full pipeline:
    1. Decompose any un-decomposed concept steps
    2. Execute approved task steps

    Returns summary dict.
    """
    decomp_results = await decompose_pending_concepts(
        goal_id=goal_id, model_key=model_key, auto_approve=auto_approve
    )

    exec_results = await execute_pending_tasks(
        goal_id=goal_id, max_steps=max_exec_steps
    )

    return {
        "decomposed": decomp_results,
        "executed": exec_results,
    }


# ---------------------------------------------------------------------------
# Create a concept step (convenience wrapper)
# ---------------------------------------------------------------------------

async def create_concept_step(
    description: str,
    goal_id: int = 0,
    step_order: int = 1,
    source: str = "assistant",
    target: str = "model",
    approval: str = "proposed",
    session_id: str = "",
) -> int:
    """
    Create a concept step. Returns the new row ID.

    source: who created it (user, assistant, directive, session)
    target: default target for decomposition hint (model, human, investigate)
    approval: proposed (needs review) or auto (skip review)
    """
    from database import execute_insert

    desc = description.replace("'", "''")
    source = source if source in ("session", "user", "directive", "assistant") else "assistant"
    target = target if target in ("model", "human", "investigate") else "model"
    approval = approval if approval in ("proposed", "approved", "rejected", "auto") else "proposed"
    sid = session_id.replace("'", "''") if session_id else "NULL"
    sid_sql = f"'{sid}'" if session_id else "NULL"

    sql = (
        f"INSERT INTO {_PLANS()} "
        f"(goal_id, step_order, description, status, step_type, parent_id, "
        f"target, executor, tool_call, approval, source, session_id) "
        f"VALUES ({goal_id}, {step_order}, '{desc}', 'pending', 'concept', NULL, "
        f"'{target}', NULL, NULL, '{approval}', '{source}', {sid_sql})"
    )
    return await execute_insert(sql)


# ---------------------------------------------------------------------------
# View plan (for display / inspection)
# ---------------------------------------------------------------------------

async def view_plan(goal_id: int | None = None, include_done: bool = False) -> str:
    """
    Render a human-readable view of the plan hierarchy.
    """
    from database import fetch_dicts

    status_filter = "" if include_done else "AND p.status NOT IN ('done','skipped')"
    where = f"WHERE p.step_type = 'concept' {status_filter}"
    if goal_id is not None:
        where += f" AND p.goal_id = {goal_id}"

    concepts = await fetch_dicts(
        f"SELECT p.*, g.title as goal_title FROM {_PLANS()} p "
        f"LEFT JOIN {_GOALS()} g ON g.id = p.goal_id "
        f"{where} ORDER BY p.goal_id, p.step_order"
    )
    if not concepts:
        return "(no active plan steps)"

    lines = []
    current_goal = None
    for c in concepts:
        gid = c.get("goal_id", 0)
        if gid != current_goal:
            current_goal = gid
            gtitle = c.get("goal_title") or f"(ad-hoc, goal_id={gid})"
            lines.append(f"\n## {gtitle}")

        status_icon = {"pending": "○", "in_progress": "▶", "done": "✓", "skipped": "—"}.get(
            c["status"], "?"
        )
        approval_tag = f" [{c.get('approval', '?')}]" if c.get("approval") != "approved" else ""
        target_tag = f" →{c.get('target', '?')}" if c.get("target") != "model" else ""
        lines.append(
            f"  {status_icon} [{c['id']}] step {c['step_order']}: "
            f"{c['description']}{approval_tag}{target_tag}"
        )

        # Load child task steps
        task_filter = "" if include_done else "AND status NOT IN ('done','skipped')"
        tasks = await fetch_dicts(
            f"SELECT * FROM {_PLANS()} WHERE parent_id = {c['id']} {task_filter} "
            f"ORDER BY step_order"
        )
        if tasks:
            for t in tasks:
                t_icon = {"pending": "·", "in_progress": "▸", "done": "✓", "skipped": "—"}.get(
                    t["status"], "?"
                )
                t_target = f" →{t['target']}" if t.get("target") != "model" else ""
                t_approval = f" [{t.get('approval', '?')}]" if t.get("approval") != "approved" else ""
                tc = ""
                if t.get("tool_call"):
                    try:
                        tc_data = json.loads(t["tool_call"]) if isinstance(t["tool_call"], str) else t["tool_call"]
                        tc = f" tool:{tc_data.get('tool', '?')}"
                    except Exception:
                        tc = " tool:?"
                lines.append(
                    f"      {t_icon} [{t['id']}] {t['description']}{tc}{t_target}{t_approval}"
                )
                if t.get("result") and t["status"] == "done":
                    result_preview = str(t["result"])[:80].replace("\n", " ")
                    lines.append(f"           → {result_preview}")

    return "\n".join(lines)
