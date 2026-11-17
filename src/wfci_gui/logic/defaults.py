"""Read a script's parameters and their defaults by parsing its source with ``ast``.

No script is executed to read its defaults. Three kinds of
parameter are found:

  * ``flag``   -- an argparse ``add_argument("--name", ...)``; its default is resolved
                  through ``defaults["key"]`` / ``RUN_CONFIG["key"]`` when written so;
  * ``env``    -- ``os.environ.get("NAME", default)`` / ``os.getenv(...)``;
  * ``assign`` -- a module-level ``name = <literal>`` (flat scripts),
                  including assignments inside module-level ``if``/``else``/``try``.

A value that is computed rather than written as a literal is reported with
``literal=False``: for ``assign`` parameters it cannot be overridden (the GUI shows it
read-only); a flag can still be passed even if its default is computed.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# Names that hold the editor-run configuration dictionary.
RUN_CONFIG_PREFIX = "RUN_CONFIG"


@dataclass
class Parameter:
    name: str                      # override key: argparse dest, env var name or variable name
    kind: str                      # "flag" | "env" | "assign"
    default: Any = None            # the literal default (None when not literal)
    literal: bool = True           # False: default is computed (source text in `source`)
    source: str = ""               # source text of the default expression
    flags: dict[str, Any] = field(default_factory=dict)  # option string -> value it stores
    action: str = "store"          # store | store_true | store_false | boolean_optional | const
    choices: tuple | None = None
    nargs: Any = None
    type_name: str | None = None   # int | float | str | Path | custom
    help: str = ""
    conditional: bool = False      # assigned inside a module-level if/else (varies per branch)
    lineno: int = 0

    @property
    def overridable(self) -> bool:
        """Flags and env vars can always be passed; an assignment only if every one is literal."""
        return self.kind != "assign" or self.literal

    def describe_default(self) -> str:
        return repr(self.default) if self.literal else f"(computed: {self.source})"


@dataclass
class ScriptParameters:
    path: Path
    flags: list[Parameter]
    env: list[Parameter]
    assigns: list[Parameter]
    run_config: dict[str, Parameter]
    file_hash: str

    @property
    def uses_argparse(self) -> bool:
        return bool(self.flags)

    def all(self) -> list[Parameter]:
        return [*self.flags, *self.env, *self.assigns]

    def get(self, name: str) -> Parameter:
        for parameter in self.all():
            if parameter.name == name:
                return parameter
        raise KeyError(f"{self.path.name} has no parameter {name!r}")


# ----------------------------------------------------------------------------- helpers
def file_hash(path: str | Path) -> str:
    """sha256 of the script file, stored with presets (a different hash is warned about)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def literal_value(node: ast.AST) -> tuple[bool, Any]:
    """(True, value) when ``node`` is a Python literal, else (False, None)."""
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False, None


def module_assignments(tree: ast.Module) -> Iterator[tuple[ast.stmt, str, ast.AST, bool]]:
    """Every module-level ``name = value`` as (statement, name, value node, conditional).

    Recurses into module-level if/else, try and with bodies (still module scope) but
    never into functions or classes. Tuple targets (``a, b = ...``) are not parameters.
    """
    def walk(statements: list[ast.stmt], conditional: bool):
        for statement in statements:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        yield statement, target.id, statement.value, conditional
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                if isinstance(statement.target, ast.Name):
                    yield statement, statement.target.id, statement.value, conditional
            elif isinstance(statement, ast.If):
                yield from walk(statement.body, True)
                yield from walk(statement.orelse, True)
            elif isinstance(statement, ast.Try):
                for block in (statement.body, statement.orelse, statement.finalbody):
                    yield from walk(block, conditional)
                for handler in statement.handlers:
                    yield from walk(handler.body, True)
            elif isinstance(statement, ast.With):
                yield from walk(statement.body, conditional)
    yield from walk(tree.body, False)


def _source(text: str, node: ast.AST) -> str:
    return (ast.get_source_segment(text, node) or ast.dump(node)).strip()


# ----------------------------------------------------------------------------- reader
def read_script_parameters(path: str | Path) -> ScriptParameters:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))

    # Module-level literal constants (for choices=SOME_TUPLE) and RUN_CONFIG entries.
    constants: dict[str, Any] = {}
    run_config: dict[str, Parameter] = {}
    by_name: dict[str, list[tuple[ast.stmt, ast.AST, bool]]] = {}
    for statement, name, value, conditional in module_assignments(tree):
        if name.startswith(RUN_CONFIG_PREFIX) and isinstance(value, ast.Dict):
            for key_node, value_node in zip(value.keys, value.values):
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                    is_literal, literal = literal_value(value_node)
                    run_config.setdefault(key_node.value, Parameter(
                        name=key_node.value, kind="run_config", default=literal,
                        literal=is_literal, source=_source(text, value_node),
                        lineno=value_node.lineno))
            continue
        by_name.setdefault(name, []).append((statement, value, conditional))
        is_literal, literal = literal_value(value)
        if is_literal and not conditional:
            constants[name] = literal

    flags = _argparse_parameters(tree, text, run_config, constants)
    env = _env_parameters(tree, text)
    assigns = []
    for name, occurrences in by_name.items():
        if name.startswith("_") or name in {"__all__", "__doc__"}:
            continue
        literals = [literal_value(value) for _, value, _ in occurrences]
        all_literal = all(ok for ok, _ in literals)
        conditional = any(cond for _, _, cond in occurrences)
        # Default shown: the first unconditional literal, else the first literal.
        default, source = None, _source(text, occurrences[0][1])
        for (statement, value, cond), (ok, literal) in zip(occurrences, literals):
            if ok and not cond:
                default, source = literal, _source(text, value)
                break
        else:
            for (statement, value, cond), (ok, literal) in zip(occurrences, literals):
                if ok:
                    default, source = literal, _source(text, value)
                    break
        assigns.append(Parameter(name=name, kind="assign", default=default, literal=all_literal,
                                 source=source, conditional=conditional,
                                 lineno=occurrences[0][0].lineno))
    return ScriptParameters(path=path, flags=flags, env=env, assigns=assigns,
                            run_config=run_config, file_hash=file_hash(path))


def _argparse_parameters(tree: ast.Module, text: str, run_config: dict[str, Parameter],
                         constants: dict[str, Any]) -> list[Parameter]:
    """One Parameter per argparse destination (several option strings may share one)."""
    by_dest: dict[str, Parameter] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        options = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        long_options = [o for o in options if o.startswith("--")]
        if not long_options:
            continue  # positional or short-only arguments are not offered in the form
        keywords = {k.arg: k.value for k in node.keywords if k.arg}
        dest = (keywords["dest"].value if isinstance(keywords.get("dest"), ast.Constant)
                else long_options[0][2:].replace("-", "_"))
        action = "store"
        if "action" in keywords:
            action_node = keywords["action"]
            if isinstance(action_node, ast.Constant):
                action = str(action_node.value)
            elif isinstance(action_node, ast.Attribute) and action_node.attr == "BooleanOptionalAction":
                action = "boolean_optional"
            else:
                action = "custom"
        default, is_literal, source = None, True, ""
        if "default" in keywords:
            default, is_literal, source = _resolve_default(keywords["default"], text, run_config)
        elif action == "store_true":
            default = False
        elif action == "store_false":
            default = True
        choices = None
        if "choices" in keywords:
            ok, value = literal_value(keywords["choices"])
            if not ok and isinstance(keywords["choices"], ast.Name):
                value = constants.get(keywords["choices"].id)
                ok = value is not None
            if ok and isinstance(value, (list, tuple, set, frozenset)):
                choices = tuple(sorted(value) if isinstance(value, (set, frozenset)) else value)
        nargs = literal_value(keywords["nargs"])[1] if "nargs" in keywords else None
        type_name = None
        if "type" in keywords:
            type_node = keywords["type"]
            type_name = type_node.id if isinstance(type_node, ast.Name) else "custom"
        help_text = literal_value(keywords["help"])[1] if "help" in keywords else ""
        stores = {"store_true": True, "store_false": False}.get(action)
        if dest in by_dest:
            # A second option string for the same destination (--debug / --full).
            existing = by_dest[dest]
            existing.flags[long_options[0]] = stores
            if existing.action in ("store_true", "store_false") and stores is not None:
                existing.action = "bool_pair"
            continue
        by_dest[dest] = Parameter(
            name=dest, kind="flag", default=default, literal=is_literal, source=source,
            flags={long_options[0]: stores}, action=action, choices=choices, nargs=nargs,
            type_name=type_name, help=str(help_text or ""), lineno=node.lineno)
    return list(by_dest.values())


def _resolve_default(node: ast.AST, text: str, run_config: dict[str, Parameter]) -> tuple[Any, bool, str]:
    """Default of an argparse argument: a literal, or a RUN_CONFIG lookup, or computed."""
    # defaults["key"] / RUN_CONFIG["key"] / config["key"]
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        key = node.slice.value
        if isinstance(key, str) and key in run_config:
            entry = run_config[key]
            return entry.default, entry.literal, entry.source
    ok, value = literal_value(node)
    if ok:
        return value, True, _source(text, node)
    # list(RUN_CONFIG["x"]) / tuple(...) around a lookup
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("list", "tuple") and len(node.args) == 1):
        value, is_literal, source = _resolve_default(node.args[0], text, run_config)
        if is_literal and isinstance(value, (list, tuple)):
            return (list(value) if node.func.id == "list" else tuple(value)), True, source
    return None, False, _source(text, node)


def _env_parameters(tree: ast.Module, text: str) -> list[Parameter]:
    """``os.environ.get("NAME", default)``, ``os.getenv("NAME", default)`` and ``os.environ["NAME"]``."""
    found: dict[str, Parameter] = {}
    for node in ast.walk(tree):
        name_node, default_node = None, None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func = node.func
            is_environ_get = (func.attr == "get" and isinstance(func.value, ast.Attribute)
                              and func.value.attr == "environ")
            is_getenv = func.attr == "getenv"
            if (is_environ_get or is_getenv) and node.args:
                name_node = node.args[0]
                default_node = node.args[1] if len(node.args) > 1 else None
        if name_node is None or not (isinstance(name_node, ast.Constant) and isinstance(name_node.value, str)):
            continue
        name = name_node.value
        if name in found:
            continue
        default, is_literal, source = None, True, ""
        if default_node is not None:
            is_literal, default = literal_value(default_node)
            source = _source(text, default_node)
        found[name] = Parameter(name=name, kind="env", default=default, literal=is_literal,
                                source=source, lineno=node.lineno)
    return list(found.values())


# ----------------------------------------------------------------------------- overrides
class OverrideError(ValueError):
    pass


def apply_assignment_overrides(text: str, overrides: dict[str, Any], filename: str = "<script>") -> ast.Module:
    """Parse ``text`` and replace the literal value of each overridden module-level name.

    Every assignment of the name is replaced (one per if/else branch, if any), so the
    override wins whichever branch runs. Refused (OverrideError) when the name is never
    assigned at module level, or when any of its assignments is computed rather than a
    literal -- replacing a computation would change what the script means.
    """
    tree = ast.parse(text, filename=filename)
    occurrences: dict[str, list[ast.stmt]] = {}
    for statement, name, value, _ in module_assignments(tree):
        occurrences.setdefault(name, []).append(statement)
    for name, value in overrides.items():
        statements = occurrences.get(name)
        if not statements:
            raise OverrideError(f"{filename}: no module-level assignment of {name!r}")
        for statement in statements:
            if not literal_value(statement.value)[0]:
                raise OverrideError(
                    f"{filename}:{statement.lineno}: {name} = {_source(text, statement.value)} is "
                    "computed, not a literal, so it cannot be overridden (it needs an approved edit)")
        if not literal_value(ast.parse(repr(value), mode="eval").body)[0]:
            raise OverrideError(f"override {name}={value!r} is not a Python literal")
        for statement in statements:
            # A fresh node per statement: one node shared by two places confuses compile().
            replacement = ast.parse(repr(value), mode="eval").body
            statement.value = ast.copy_location(replacement, statement.value)
    return ast.fix_missing_locations(tree)


def format_literal(value: Any) -> str:
    """Compact Python literal for ``--set name=value`` (no spaces inside lists)."""
    if isinstance(value, (list, tuple)):
        inner = ",".join(format_literal(v) for v in value)
        return f"[{inner}]" if isinstance(value, list) else f"({inner}{',' if len(value) == 1 else ''})"
    return repr(value)
