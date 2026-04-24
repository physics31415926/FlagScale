"""Task plan — structured multi-step planning with persistence."""

import os
import time
import uuid

from typing import Dict, List, Optional

import yaml


VALID_STEP_STATUSES = ("pending", "doing", "done", "skipped", "blocked")
VALID_PLAN_STATUSES = ("active", "completed", "abandoned")

STATUS_ICONS = {
    "pending": " ",
    "doing": "→",
    "done": "✓",
    "skipped": "-",
    "blocked": "!",
}


class TaskPlan:
    """Manages structured task plans with YAML persistence."""

    def __init__(self, plan_dir: str):
        self._dir = plan_dir

    def _plan_path(self, plan_id: str) -> str:
        return os.path.join(self._dir, f"{plan_id}.yaml")

    def _active_path(self) -> str:
        return os.path.join(self._dir, "active.yaml")

    def _save(self, plan: dict):
        os.makedirs(self._dir, exist_ok=True)
        plan["updated"] = time.time()
        path = self._plan_path(plan["id"])
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(plan, f, allow_unicode=True, default_flow_style=False)
        if plan["status"] == "active":
            active_path = self._active_path()
            with open(active_path, "w", encoding="utf-8") as f:
                yaml.dump({"active_id": plan["id"]}, f)

    def _clear_active(self):
        active_path = self._active_path()
        if os.path.isfile(active_path):
            os.remove(active_path)

    def create(self, title: str, steps: List[str], session_id: str = "") -> dict:
        old = self.get_active()
        if old:
            old["status"] = "abandoned"
            old["updated"] = time.time()
            self._save(old)
            self._clear_active()

        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        step_list = []
        for i, desc in enumerate(steps, 1):
            step_list.append({
                "id": i,
                "title": desc,
                "status": "pending",
                "notes": "",
                "depends_on": [i - 1] if i > 1 else [],
            })

        plan = {
            "id": plan_id,
            "title": title,
            "status": "active",
            "created": time.time(),
            "updated": time.time(),
            "session_id": session_id,
            "steps": step_list,
        }
        self._save(plan)
        return plan

    def get_active(self) -> Optional[dict]:
        active_path = self._active_path()
        if not os.path.isfile(active_path):
            return None
        try:
            with open(active_path, "r", encoding="utf-8") as f:
                ref = yaml.safe_load(f)
            active_id = ref.get("active_id")
            if not active_id:
                return None
            return self._load(active_id)
        except Exception:
            return None

    def _load(self, plan_id: str) -> Optional[dict]:
        path = self._plan_path(plan_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception:
            return None

    def update_step(self, step_id: int, status: str, notes: str = "") -> dict:
        plan = self.get_active()
        if not plan:
            raise ValueError("No active plan")
        if status not in VALID_STEP_STATUSES:
            raise ValueError(f"Invalid status: {status}")

        step = self._find_step(plan, step_id)
        step["status"] = status
        if notes:
            step["notes"] = notes

        if status in ("done", "skipped"):
            for s in plan["steps"]:
                if s["status"] == "pending":
                    deps = s.get("depends_on", [])
                    if not deps or all(
                        self._find_step(plan, d)["status"] in ("done", "skipped")
                        for d in deps
                    ):
                        s["status"] = "doing"
                        break

        self._save(plan)
        return plan

    def add_steps(self, steps: List[str], after_step_id: Optional[int] = None) -> dict:
        plan = self.get_active()
        if not plan:
            raise ValueError("No active plan")

        existing_ids = [s["id"] for s in plan["steps"]]
        next_id = max(existing_ids) + 1 if existing_ids else 1

        new_steps = []
        for i, desc in enumerate(steps):
            sid = next_id + i
            new_steps.append({
                "id": sid,
                "title": desc,
                "status": "pending",
                "notes": "",
                "depends_on": [],
            })

        if after_step_id is not None:
            idx = next(
                (i for i, s in enumerate(plan["steps"]) if s["id"] == after_step_id),
                None,
            )
            if idx is None:
                raise ValueError(f"Step {after_step_id} not found")
            for ns in new_steps:
                ns["depends_on"] = [after_step_id]
            insert_pos = idx + 1
            plan["steps"] = plan["steps"][:insert_pos] + new_steps + plan["steps"][insert_pos:]
        else:
            if plan["steps"]:
                last_id = plan["steps"][-1]["id"]
                for ns in new_steps:
                    ns["depends_on"] = [last_id]
                    last_id = ns["id"]
            plan["steps"].extend(new_steps)

        self._save(plan)
        return plan

    def skip_step(self, step_id: int, reason: str = "") -> dict:
        return self.update_step(step_id, "skipped", notes=reason or "skipped")

    def complete(self) -> dict:
        plan = self.get_active()
        if not plan:
            raise ValueError("No active plan")
        plan["status"] = "completed"
        self._save(plan)
        self._clear_active()
        return plan

    def abandon(self, reason: str = "") -> dict:
        plan = self.get_active()
        if not plan:
            raise ValueError("No active plan")
        plan["status"] = "abandoned"
        if reason:
            plan["abandon_reason"] = reason
        self._save(plan)
        self._clear_active()
        return plan

    def summary(self) -> str:
        plan = self.get_active()
        if not plan:
            return "No active plan."
        return self._format_plan(plan)

    def context_for_prompt(self) -> str:
        plan = self.get_active()
        if not plan:
            return ""
        lines = []
        for s in plan["steps"]:
            icon = STATUS_ICONS.get(s["status"], " ")
            line = f"{s['id']}. [{icon}] {s['title']}"
            if s.get("notes"):
                line += f" — {s['notes']}"
            lines.append(line)
        return (
            f'<active-plan title="{plan["title"]}">\n'
            + "\n".join(lines)
            + "\n</active-plan>"
        )

    def _format_plan(self, plan: dict) -> str:
        lines = [f"Plan: {plan['title']} [{plan['status']}]"]
        for s in plan["steps"]:
            icon = STATUS_ICONS.get(s["status"], " ")
            line = f"  {s['id']}. [{icon}] {s['title']}"
            if s.get("notes"):
                line += f" — {s['notes']}"
            lines.append(line)
        done = sum(1 for s in plan["steps"] if s["status"] in ("done", "skipped"))
        lines.append(f"Progress: {done}/{len(plan['steps'])}")
        return "\n".join(lines)

    @staticmethod
    def _find_step(plan: dict, step_id: int) -> dict:
        for s in plan["steps"]:
            if s["id"] == step_id:
                return s
        raise ValueError(f"Step {step_id} not found")

    def list_plans(self) -> List[dict]:
        if not os.path.isdir(self._dir):
            return []
        plans = []
        for fname in sorted(os.listdir(self._dir)):
            if not fname.startswith("plan_") or not fname.endswith(".yaml"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    plan = yaml.safe_load(f)
                steps = plan.get("steps", [])
                plans.append({
                    "id": plan.get("id", fname),
                    "title": plan.get("title", ""),
                    "status": plan.get("status", "?"),
                    "done": sum(1 for s in steps if s.get("status") in ("done", "skipped")),
                    "total": len(steps),
                    "created": plan.get("created", 0),
                })
            except Exception:
                continue
        return plans

    def clear_completed(self) -> int:
        if not os.path.isdir(self._dir):
            return 0
        count = 0
        for fname in os.listdir(self._dir):
            if not fname.startswith("plan_") or not fname.endswith(".yaml"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    plan = yaml.safe_load(f)
                if plan.get("status") in ("completed", "abandoned"):
                    os.remove(path)
                    count += 1
            except Exception:
                continue
        return count
