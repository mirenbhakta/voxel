#!/usr/bin/env python3
"""
Structured matcher for the auto-permissions hook.

Reads a Bash command from stdin, parses it into a typed pipeline AST, and
matches it against rules expressed as Python pattern matches. Prints one of:

    ALLOW <reason>
    NOLOG <reason>
    (nothing)            -- fall through to manual permission flow

The hook treats ALLOW as auto-approval, NOLOG as "skip the fallthrough log
but still ask the user", and empty as fall through.

Why a parser instead of more regexes? Bash commands have a small grammar
(pipelines of commands, each with subcommands, flags, positionals, strings,
and a couple of whitelisted redirects). Parsing once and matching against
the structured form is dramatically more readable than trying to express
the same thing as nested character classes.

The grammar accepted by this matcher is intentionally narrow:

    pipeline   := command ('|' command)*
    command    := word token*
    token      := positional | short_flag | long_flag | string | redirect | dashdash
    short_flag := '-' [A-Za-z] [A-Za-z0-9-]* ('=' value)?
    long_flag  := '--' name ('=' value)?
    positional := safe_word
    string     := double-quoted or single-quoted literal
    redirect   := '2>&1' | '2>/dev/null'   (whitelist; nothing else)
    dashdash   := '--'

Anything outside this grammar -- '&&', ';', '||', backticks, '$()', '<',
'>' (other than the whitelisted redirects), or any token containing shell
metacharacters -- causes parsing to fail and the matcher exits with no
output (fall through to manual review).
"""

from __future__ import annotations

import re
import shlex
import sys
from dataclasses import dataclass, field
from typing import Iterable, Optional


# --- Token classification ---------------------------------------------------

# Safe characters allowed in positional words and flag values. Includes
# everything needed for git revs ({}@~^), file paths (/_.-), version
# specifiers (=:), package names (,), URLs (#?+), and shell-safe
# punctuation. Excludes: whitespace, quotes, backslashes, dollars,
# backticks, semicolons, ampersands, pipes, angle brackets, parentheses,
# braces (except {} which we allow for git revs), brackets.
SAFE_WORD_RE = re.compile(r"^[A-Za-z0-9_./,=:@#+?~^!{}@-]+$")

# Whitelisted redirect tokens. Anything else with > or < fails to parse.
WHITELIST_REDIRECTS = {"2>&1", "2>/dev/null", "1>/dev/null", "&>/dev/null"}


@dataclass
class Token:
    """A classified token from a single command."""

    kind: str  # "positional" | "short_flag" | "long_flag" | "string" | "redirect" | "dashdash"
    value: str
    flag_value: Optional[str] = None  # for short_flag/long_flag with =value


@dataclass
class Cmd:
    """A single command in a pipeline."""

    name: str  # the executable, e.g. "cargo", "git", "/tmp/xwin-check.sh"
    sub: Optional[str] = None  # first positional after name, if it looks like a verb
    args: list[Token] = field(default_factory=list)  # remaining tokens after sub

    def positionals(self) -> list[str]:
        """Positional arg values (not including subcommand)."""
        return [t.value for t in self.args if t.kind == "positional"]

    def flags(self) -> list[Token]:
        """Short and long flag tokens."""
        return [t for t in self.args if t.kind in ("short_flag", "long_flag")]

    def has_flag(self, *names: str) -> bool:
        """Check if any of the named long flags are present (e.g. has_flag('check'))."""
        for t in self.args:
            if t.kind == "long_flag" and t.value in names:
                return True
        return False


@dataclass
class Pipeline:
    """A pipeline of one or more commands joined by '|'."""

    commands: list[Cmd]


# --- Parser -----------------------------------------------------------------


class ParseError(Exception):
    pass


# Subcommand-style verb pattern. The first positional argument is treated
# as a subcommand if it matches this shape. Allows hyphens for verbs like
# "ls-files", "rev-parse", "cat-file".
VERB_RE = re.compile(r"^[a-z][a-z0-9-]*$")


def parse(cmd: str) -> Pipeline:
    """
    Parse a bash command string into a typed Pipeline.

    Raises ParseError if the command contains constructs outside the
    accepted grammar.
    """

    # Reject command substitution anywhere, even inside double quotes,
    # since it executes shell. `$(...)`, `${...}`, and backticks all run
    # commands and are unconditionally rejected.
    if "$(" in cmd or "`" in cmd or "${" in cmd:
        raise ParseError("command substitution not allowed")

    # Reject other shell control constructs (&&, ||, ;, here-docs,
    # redirects other than the whitelist) when they appear outside
    # quoted regions.
    _reject_unquoted_metas(cmd)

    try:
        raw_tokens = shlex.split(cmd, posix=True)
    except ValueError as e:
        raise ParseError(f"shlex error: {e}") from e

    if not raw_tokens:
        raise ParseError("empty command")

    # Split on '|' into segments.
    segments: list[list[str]] = [[]]
    for tok in raw_tokens:
        if tok == "|":
            segments.append([])
        else:
            segments[-1].append(tok)

    if any(not seg for seg in segments):
        raise ParseError("empty pipeline segment")

    commands = [_parse_segment(seg) for seg in segments]
    return Pipeline(commands=commands)


# Patterns of unquoted shell metacharacters that the matcher refuses to
# touch. We strip quoted regions before scanning so that quoted content can
# legitimately contain these characters.
_DISALLOWED_UNQUOTED = re.compile(r"(&&|\|\||;|`|\$\(|\$\{|>>|<<|<|>(?!&[12]?$|/dev/null))")


def _reject_unquoted_metas(cmd: str) -> None:
    """
    Strip quoted regions and reject the result if it contains shell
    metacharacters that aren't part of our accepted grammar.

    The redirects '2>&1' and '2>/dev/null' are handled specially: they're
    matched as token-shaped strings later, so the '>' check has to allow
    them. Bare '|' (pipeline separator) is allowed.
    """

    # Strip double-quoted and single-quoted regions.
    stripped = re.sub(r'"(?:\\.|[^"\\])*"', '""', cmd)
    stripped = re.sub(r"'[^']*'", "''", stripped)

    # Allow '2>&1', '2>/dev/null', '&>/dev/null', '1>/dev/null'.
    stripped = re.sub(r"\b[12]?>(?:&[12]|/dev/null)\b", "", stripped)
    stripped = re.sub(r"&>/dev/null\b", "", stripped)

    if _DISALLOWED_UNQUOTED.search(stripped):
        raise ParseError("disallowed shell metacharacter outside quotes")


def _parse_segment(tokens: list[str]) -> Cmd:
    """Parse a single pipeline segment into a Cmd."""

    if not tokens:
        raise ParseError("empty segment")

    name = tokens[0]
    # The command name itself must be a safe word or absolute path.
    if not (SAFE_WORD_RE.match(name) or _is_safe_path(name)):
        raise ParseError(f"unsafe command name: {name!r}")

    rest = tokens[1:]
    sub: Optional[str] = None
    args: list[Token] = []

    # First non-flag token after the command name may be a subcommand.
    sub_consumed = False
    for raw in rest:
        if not sub_consumed and not raw.startswith("-") and VERB_RE.match(raw):
            sub = raw
            sub_consumed = True
            continue
        args.append(_classify_token(raw))
        sub_consumed = True  # only the first eligible token becomes the sub

    return Cmd(name=name, sub=sub, args=args)


def _is_safe_path(s: str) -> bool:
    """Check if a string is a safe absolute or relative path used as exec."""
    # Allow / and . for paths. No spaces, no metacharacters. shlex already
    # split on spaces, so this is just a character check.
    return bool(re.match(r"^[A-Za-z0-9_./-]+$", s))


def _classify_token(raw: str) -> Token:
    """Classify a raw shlex-split token by shape."""

    if raw == "--":
        return Token(kind="dashdash", value="--")

    if raw in WHITELIST_REDIRECTS:
        return Token(kind="redirect", value=raw)

    # 2>file forms beyond the whitelist are rejected.
    if ">" in raw or "<" in raw:
        raise ParseError(f"unrecognized redirect token: {raw!r}")

    if raw.startswith("--"):
        body = raw[2:]
        name, _, value = body.partition("=")
        if not re.match(r"^[A-Za-z][A-Za-z0-9-]*$", name):
            raise ParseError(f"malformed long flag: {raw!r}")
        if value and not SAFE_WORD_RE.match(value):
            raise ParseError(f"unsafe long flag value: {raw!r}")
        return Token(kind="long_flag", value=name, flag_value=value or None)

    if raw.startswith("-") and len(raw) > 1 and raw[1].isalpha():
        body = raw[1:]
        name, _, value = body.partition("=")
        if not re.match(r"^[A-Za-z][A-Za-z0-9-]*$", name):
            raise ParseError(f"malformed short flag: {raw!r}")
        if value and not SAFE_WORD_RE.match(value):
            raise ParseError(f"unsafe short flag value: {raw!r}")
        return Token(kind="short_flag", value=name, flag_value=value or None)

    # Plain positional or quoted string literal. After shlex, any token
    # that contains shell metacharacters (|, ;, &, <, >) must have come
    # from inside quotes -- if those characters had been unquoted, shlex
    # would have split them into separate tokens. Command substitution
    # markers ($(, `, ${) were already rejected for the whole input.
    #
    # NUL and newlines are still rejected as a defense against weird
    # encoded payloads.
    if "\0" in raw or "\n" in raw:
        raise ParseError(f"control char in token: {raw!r}")

    if SAFE_WORD_RE.match(raw):
        return Token(kind="positional", value=raw)

    # Token contains characters outside the safe-word set. Treat it as a
    # string literal -- it must have been quoted in the original command.
    return Token(kind="string", value=raw)


# --- Decisions --------------------------------------------------------------


@dataclass
class Allow:
    reason: str


@dataclass
class NoLog:
    reason: str = "always-ask command"


Decision = Optional[Allow | NoLog]


# --- Rule helpers -----------------------------------------------------------


# Pipe-tail filters: commands that may appear after the main command in a
# pipeline. They consume stdin and produce stdout, never touch the filesystem
# beyond the file paths their args explicitly name.
PIPE_TAIL_FILTERS = {
    "head", "tail", "grep", "wc", "sort", "uniq", "cut", "cat", "less",
    "jq", "awk", "sed", "tr", "rev",
}


def pipe_tail_safe(tail: list[Cmd]) -> bool:
    """Check that all post-pipe commands are in the safe filter set."""
    for cmd in tail:
        if cmd.name not in PIPE_TAIL_FILTERS:
            return False
        # Filter args must already be safe by construction (parsed tokens),
        # but we additionally reject filters that take subcommands we don't
        # recognize. None of the filters in our set use subcommands.
    return True


def all_args_safe(args: list[Token]) -> bool:
    """All args must already have parsed cleanly. This is a no-op signal."""
    return True  # parsing already enforced safety


# --- Rule sets --------------------------------------------------------------

CARGO_READ = {
    "check", "test", "build", "tree", "metadata", "clippy", "doc", "bench",
    "rustc", "rustdoc", "locate-project", "read-manifest", "verify-project",
    "pkgid", "search", "version", "help",
}

GIT_READ = {
    "status", "log", "diff", "show", "blame", "ls-files", "ls-tree",
    "shortlog", "cat-file", "describe", "rev-parse", "grep", "rev-list",
    "whatchanged", "name-rev", "count-objects", "fsck", "verify-pack",
    "verify-commit", "verify-tag", "symbolic-ref", "for-each-ref",
    "merge-base", "cherry", "range-diff", "branch", "tag", "remote",
    "stash", "reflog", "config", "fetch",
}

# git verbs that mutate state. Matching here means: do NOT auto-approve,
# but also do NOT log to the fallthrough corpus -- the user always wants
# to confirm these manually.
GIT_NOLOG = {
    "commit", "push", "reset", "rebase", "merge", "cherry-pick", "am",
    "revert", "gc", "prune", "filter-branch", "filter-repo", "update-ref",
    "update-index", "apply", "format-patch", "send-email", "request-pull",
    "bundle", "svn", "p4",
}

GH_OBJECTS = {
    "issue", "pr", "release", "run", "workflow", "gist", "label", "cache",
    "variable", "secret", "ruleset", "repo", "ssh-key", "gpg-key", "alias",
}

GH_READ_VERBS = {
    "view", "list", "status", "diff", "checks", "view-log", "browse",
}

GH_NOLOG_VERBS = {
    "create", "edit", "close", "reopen", "comment", "merge", "review",
    "delete", "lock", "unlock", "update-branch", "ready", "develop",
    "checkout", "set", "run", "enable", "disable", "cancel", "rerun",
    "delete-key", "clone", "fork", "sync", "rename", "archive",
    "unarchive", "edit-protection",
}

RUSTUP_READ_VERBS = {
    "show", "which", "version",
}
RUSTUP_READ_SUBVERBS = {
    "list", "show", "installed", "active",
}

SAFE_SYSTEM_INFO = {
    "uname", "lscpu", "free", "df", "nproc", "date", "env", "pwd", "id",
    "whoami", "hostname", "which", "type", "command",
}


def is_gh_read(c: Cmd) -> bool:
    """gh <object> <verb>: read-only verb."""
    if c.name != "gh" or c.sub not in GH_OBJECTS:
        return False
    pos = c.positionals()
    return bool(pos) and pos[0] in GH_READ_VERBS


def is_gh_nolog(c: Cmd) -> bool:
    """gh <object> <verb>: state-changing verb."""
    if c.name != "gh" or c.sub not in GH_OBJECTS:
        return False
    pos = c.positionals()
    return bool(pos) and pos[0] in GH_NOLOG_VERBS


def is_gh_api_write(c: Cmd) -> bool:
    """gh api with -X POST/PATCH/PUT/DELETE or -f/-F field args."""
    if c.name != "gh" or c.sub != "api":
        return False
    for i, t in enumerate(c.args):
        # -X / --method <verb>
        if t.kind in ("short_flag", "long_flag") and t.value in ("X", "method"):
            if t.flag_value and t.flag_value.upper() in ("POST", "PATCH", "PUT", "DELETE"):
                return True
            # Value in next token.
            if i + 1 < len(c.args):
                nxt = c.args[i + 1]
                if nxt.kind == "positional" and nxt.value.upper() in ("POST", "PATCH", "PUT", "DELETE"):
                    return True
        # -f / -F / --field / --raw-field imply a write.
        if t.kind == "short_flag" and t.value in ("f", "F"):
            return True
        if t.kind == "long_flag" and t.value in ("field", "raw-field"):
            return True
    return False


# --- Main classifier --------------------------------------------------------


def classify(p: Pipeline) -> Decision:
    """Match the parsed pipeline against the rule set."""

    head, *tail = p.commands

    # Pipe-tail safety: any rule that allows a pipeline must check this.
    tail_safe = pipe_tail_safe(tail)

    # --- Cargo read-only ---
    if head.name == "cargo" and head.sub in CARGO_READ and tail_safe:
        return Allow(f"cargo {head.sub}")

    # --- Bare git read verbs ---
    if head.name == "git" and head.sub in GIT_READ and tail_safe:
        return Allow(f"git {head.sub}")

    # --- git mutating verbs (no-log) ---
    if head.name == "git" and head.sub in GIT_NOLOG:
        return NoLog(f"git {head.sub}")

    # --- gh read-only ---
    if is_gh_read(head) and tail_safe:
        return Allow(f"gh {head.sub} read")

    # --- gh state-changing (no-log) ---
    if is_gh_nolog(head) or is_gh_api_write(head):
        return NoLog(f"gh {head.sub} write")

    # --- rustup queries ---
    if head.name == "rustup" and tail_safe:
        if head.sub in ("target", "component", "toolchain"):
            pos = head.positionals()
            if pos and pos[0] in RUSTUP_READ_SUBVERBS:
                return Allow(f"rustup {head.sub} {pos[0]}")
        if head.sub in RUSTUP_READ_VERBS:
            return Allow(f"rustup {head.sub}")
        if head.has_flag("version"):
            return Allow("rustup --version")

    # --- Safe system info commands ---
    if head.name in SAFE_SYSTEM_INFO and tail_safe:
        return Allow(f"system info: {head.name}")

    # --- Allowed exec wrappers (loaded from local rules) ---
    for rule in EXEC_RULES:
        if rule.matches(head) and tail_safe:
            return Allow(f"exec wrapper: {head.name}")

    return None


# --- Local exec wrapper rules ----------------------------------------------


@dataclass
class ExecRule:
    """A rule that allows running a specific exec path with safe args."""

    path: str  # exact path to match (e.g. "/tmp/xwin-check.sh")
    sub_in: Optional[set[str]] = None  # if set, sub must be in this set

    def matches(self, c: Cmd) -> bool:
        if c.name != self.path:
            return False
        if self.sub_in is not None and c.sub not in self.sub_in:
            return False
        return True


# Default rules. Local rules can extend this list by importing
# permission_rules_local and appending.
EXEC_RULES: list[ExecRule] = []


def _load_local_rules() -> None:
    """Load .claude/hooks/permission_rules.local.py if present."""
    import importlib.util
    from pathlib import Path

    here = Path(__file__).parent
    local = here / "permission_rules.local.py"
    if not local.exists():
        return

    spec = importlib.util.spec_from_file_location("permission_rules_local", local)
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # Local rules are best-effort. Don't crash the hook on a syntax error
        # in the user's personal file.
        return

    extra = getattr(mod, "EXEC_RULES", None)
    if extra:
        EXEC_RULES.extend(extra)


# --- Entry point ------------------------------------------------------------


def main() -> int:
    cmd = sys.stdin.read().strip()
    if not cmd:
        return 0

    _load_local_rules()

    try:
        pipeline = parse(cmd)
    except ParseError:
        # Fall through to manual review.
        return 0

    decision = classify(pipeline)
    if isinstance(decision, Allow):
        print(f"ALLOW {decision.reason}")
    elif isinstance(decision, NoLog):
        print(f"NOLOG {decision.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
