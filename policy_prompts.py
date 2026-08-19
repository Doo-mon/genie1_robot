"""EE、joint 与 bread 策略入口共享的 prompt 目录及交互确认流程。

目录保留当前任务、历史任务和 Shell 命名 prompt 的原始文本，包括有意保留的重复项。
``PROMPT_GROUPS`` 定义分组，``PROFILE_GROUPS`` 决定各入口菜单的主分组与可见范围。
调用方可提供 literal prompt、裸编号、``group:key`` 或 ``named:key``，也可现场输入自定义文本。
所有候选都必须再次确认；仅 ``y/yes`` 放行，``q`` 退出，拒绝则返回完整选择菜单。
模块只依赖 Python 标准库，不导入 NumPy、机器人、网络、CLI 或录制组件。
同步、RTC、rollout 入口在 reset、启动 runner 或 recorder 之前调用这里完成安全确认。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


JOINT_DESK_PROMPTS = {
    "1": "The left arm picks up the pen on the desk and places the pen into the pen holder at the top of the storage box",
    "2": "The right arm picks up the eraser on the desk and places the eraser on the third shelf from the bottom of the storage box",
    "3": "The left arm picks up the correction fluid on the desk and places the correction fluid on the third shelf from the bottom of the storage box",
    "4": "The right arm picks up the charger on the desk and places the charger on the third shelf from the bottom of the storage box",
    "5": "The left arm picks up the pencil case on the desk and places the pencil case on the second shelf from the bottom of the storage box",
    "6": "The right arm picks up the stapler on the desk and places the stapler on the second shelf from the bottom of the storage box",
    "7": "TThe right arm pushes the books on the desk to the edge of the desk and the right arm picks up the books on the desk and places the books on the bookshelf attached to the side of the storage box",
    "8": "The right arm picks up the trash on the desk and places the trash into the trash can",
    "9": "In the children's room, tidy the desk by placing every object in its designated location. Use the left arm to put the pen in the top pen holder, the correction fluid on the third shelf from the bottom, and the pencil case on the second shelf from the bottom. Use the right arm to put the eraser and charger on the third shelf, the stapler on the second shelf, push the books to the desk edge and place them onthe side bookshelf, and putthe trash inthe trash can.",
}


JOINT_LEGACY_SPLIT_PROMPTS = {
    "1": (
        "The left arm picks up the pen on the desk, and the left arm places the pen "
        "into the pen holder at the top of the storage box"
    ),
    "2": (
        "The right arm picks up the eraser on the desk, and the right arm places the "
        "eraser on the third layer of the storage box from the bottom"
    ),
    "3": (
        "The left arm picks up the correction fluid on the desk, and the left arm places "
        "the correction fluid on the third layer of the storage box from the bottom"
    ),
    "4": (
        "The right arm picks up the charger on the desk, and the right arm places the "
        "charger on the third layer of the storage box from the bottom"
    ),
    "5": (
        "The left arm picks up the pencil case on the desk, and the left arm places the "
        "pencil case on the second layer of the storage box from the bottom"
    ),
    "6": (
        "The right arm picks up the stapler on the desk, and the right arm places the "
        "stapler on the second layer of the storage box from the bottom"
    ),
    "7": (
        "The right arm pushes the books on the desk to the edge of the desk, and the "
        "right arm picks up the books on the desk, and the right arm places the books "
        "on the bookshelf beside the storage box"
    ),
    "8": (
        "The right arm picks up the trash on the desk, and the right arm places the "
        "trash into the trash can"
    ),
    "9": "The left arm picks up the pen on the desk",
    "10": "The left arm places the pen into the pen holder at the top of the storage box",
    "11": "The right arm picks up the eraser on the desk",
    "12": "The right arm places the eraser on the third layer of the storage box from the bottom",
    "13": "The left arm picks up the correction fluid on the desk",
    "14": "The left arm places the correction fluid on the third layer of the storage box from the bottom",
    "15": "The right arm picks up the charger on the desk",
    "16": "The right arm places the charger on the third layer of the storage box from the bottom",
    "17": "The left arm picks up the pencil case on the desk",
    "18": "The left arm places the pencil case on the second layer of the storage box from the bottom",
    "19": "The right arm picks up the stapler on the desk",
    "20": "The right arm places the stapler on the second layer of the storage box from the bottom",
    "21": "The right arm pushes the books on the desk to the edge of the desk",
    "22": "The right arm picks up the books on the desk",
    "23": "The right arm places the books on the bookshelf beside the storage box",
    "24": "The right arm picks up the trash on the desk",
    "25": "The right arm places the trash into the trash can",
}


JOINT_DRESSING_TABLE_PROMPTS = {
    "1": "The left arm picks up the BB cream from the desk and places the BB cream into the middle compartment at the front of the top layer of the storage box",
    "2": "The left arm picks up the concealer from the desk and places the concealer into the transparent drawer on the top layer of the storage box",
    "3": "The left arm picks up the eye cream from the desk and places the eye cream into the middle compartment at the front of the top layer of the storage box",
    "4": "The left arm picks up the lipstick from the desk and places the lipstick into the top open compartment on the left side of the storage box",
    "5": "The left arm picks up the liquid foundation from the desk and places the liquid foundation into the top open compartment on the left side of the storage box",
    "6": "The right arm picks up the eyebrow pencil from the desk and places the eyebrow pencil into the middle open compartment on the left side of the storage box",
    "7": "The right arm picks up the lotion from the desk and places the lotion into the long strip-shaped compartment at the back of the top layer of the storage box",
    "8": "The right arm picks up the makeup sponge from the desk and places the makeup sponge into the right compartment at the front of the top layer of the storage box",
    "9": "The right arm picks up the powder puff from the desk and places the powder puff into the right compartment at the front of the top layer of the storage box",
    "10": "The right arm picks up the serum from the desk and places the serum into the long strip-shaped compartment at the back of the top layer of the storage box",
    "11": "The right arm picks up the skin toner from the desk and places the skin toner into the long strip-shaped compartment at the back of the top layer of the storage box",
    "12": "The right arm pulls open the transparent drawer on the top layer of the storage box",
    "13": "The right arm pushes the transparent drawer on the top layer back into the storage box",
    "14": "The left arm picks up the concealer from the desk and places the concealer into the transparent drawer on the top layer of the storage box",
    "15": "Tidy up the dressing table",
    "16": "The right arm pushes the books on the desk to the edge of the desk, and the right arm picks up the books on the desk, and the right arm places the books on the bookshelf beside the storage box",
}


EE_5659_PROMPTS = {
    "1": "The left arm picks up the tissue on the table and places the tissue into the leftmost white square storage box",
    "2": "The left arm picks up the toy car on the table and places the toy car into the green rectangular storage box",
    "3": "The right arm picks up the remote control on the table and places the remote control into the white storage box with an orange compartment",
    "4": "The right arm picks up the snack on the table and places the snack into the rightmost yellow storage box",
    "5": "The left arm picks up the cup on the table and places the cup onto the black round coaster",
}


EE_LEGACY_DESK_PROMPTS = {
    "1": (
        "The right arm picks up the stapler on the desk, and the right arm "
        "places the stapler on the second layer of the storage box from the bottom"
    ),
    "2": (
        "The left arm picks up the pen on the desk, and the left arm places "
        "the pen into the pen holder at the top of the storage box"
    ),
    "3": (
        "The right arm pushes the books on the desk to the edge of the desk, "
        "and the right arm picks up the books on the desk, and the right arm "
        "places the books on the bookshelf beside the storage box"
    ),
    "4": (
        "The right arm picks up the trash on the desk, and the right arm "
        "places the trash into the trash can"
    ),
    "5": (
        "The right arm picks up the eraser on the desk, and the right arm "
        "places the eraser on the third layer of the storage box from the bottom"
    ),
    "6": (
        "The left arm picks up the correction fluid on the desk, and the left "
        "arm places the correction fluid on the third layer of the storage box "
        "from the bottom"
    ),
    "7": (
        "The right arm picks up the charger on the desk, and the right arm "
        "places the charger on the third layer of the storage box from the bottom"
    ),
    "8": (
        "The left arm picks up the pencil case on the desk, and the left arm "
        "places the pencil case on the second layer of the storage box from the bottom"
    ),
}


BREAD_PROMPTS = {
    "1": "Pick up the Sliced bread from the white table with the right hand and place it on/in the white plate",
    "2": "Pick up the Long loaf from the white table with the right hand and place it on/in the white plate",
    "3": "Pick up the bread ring from the white table with the right hand and place it on/in the white plate",
    "4": "Pick up the dinner roll from the white table with the right hand and place it on/in the white plate",
}


NAMED_PROMPTS = {
    "bread-all": (
        "Pick up the bread ring from the white table with the right hand and place it on/in the white plate. "
        "Pick up the long loaf from the white table with the right hand and place it on/in the white plate. "
        "Pick up the dinner roll from the white table with the right hand and place it on/in the white plate. "
        "Pick up the sliced bread from the white table with the right hand and place it on/in the white plate."
    ),
    "5659-all": (
        "The left arm picks up the tissue on the table and places the tissue into the leftmost white square storage box. "
        "The left arm picks up the toy car on the table and places the toy car into the green rectangular storage box. "
        "The right arm picks up the remote control on the table and places the remote control into the white storage box with an orange compartment. "
        "The right arm picks up the snack on the table and places the snack into the rightmost yellow storage box. "
        "The left arm picks up the cup on the table and places the cup onto the black round coaster."
    ),
    "tidy-desk": "Tidy up the desk",
}


PROMPT_GROUPS = {
    "joint-desk": JOINT_DESK_PROMPTS,
    "joint-legacy-split": JOINT_LEGACY_SPLIT_PROMPTS,
    "joint-dressing-table": JOINT_DRESSING_TABLE_PROMPTS,
    "ee-5659": EE_5659_PROMPTS,
    "ee-legacy-desk": EE_LEGACY_DESK_PROMPTS,
    "bread": BREAD_PROMPTS,
}


PROFILE_GROUPS = {
    "joint": (
        "joint-desk",
        "joint-legacy-split",
        "joint-dressing-table",
    ),
    "ee": (
        "ee-5659",
        "ee-legacy-desk",
        "bread",
    ),
    "5659": ("ee-5659",),
    "bread": ("bread",),
}


_NAMED_PROMPT_ALIASES = {
    "tidy-up-the-desk": "tidy-desk",
}


def _normalize_name(value: str) -> str:
    normalized = str(value).strip().casefold().replace("_", "-")
    return "-".join(normalized.split())


def _groups_for_profile(profile: str) -> tuple[str, ...]:
    normalized = _normalize_name(profile)
    if normalized in PROFILE_GROUPS:
        return PROFILE_GROUPS[normalized]
    if normalized in PROMPT_GROUPS:
        return (normalized,)
    available = ", ".join(PROFILE_GROUPS)
    raise KeyError(f"Unknown prompt profile {profile!r}; available profiles: {available}")


def resolve_prompt_key(profile: str, prompt_key: str) -> str:
    """Resolve a profile-relative, group-qualified, or named prompt key.

    A bare key selects from the profile's first (primary) group.  Other groups
    use ``group:key`` and global named prompts use ``named:key``.  Group and
    named identifiers are case-insensitive and accept underscores in place of
    hyphens; numeric prompt keys remain exact.
    """

    groups = _groups_for_profile(profile)
    selector = str(prompt_key).strip()
    if not selector:
        raise KeyError("Prompt key cannot be empty")

    if ":" not in selector:
        group_name = groups[0]
        item_key = selector
    else:
        namespace, item_key = selector.split(":", 1)
        namespace = _normalize_name(namespace)
        item_key = item_key.strip()
        if namespace == "named":
            named_key = _normalize_name(item_key)
            named_key = _NAMED_PROMPT_ALIASES.get(named_key, named_key)
            try:
                return NAMED_PROMPTS[named_key]
            except KeyError as error:
                available = ", ".join(NAMED_PROMPTS)
                raise KeyError(
                    f"Unknown named prompt {item_key!r}; available: {available}"
                ) from error
        group_name = namespace
        if group_name not in groups:
            available = ", ".join(groups)
            raise KeyError(
                f"Prompt group {namespace!r} is not in profile {profile!r}; "
                f"available: {available}"
            )

    if not item_key:
        raise KeyError("Prompt item key cannot be empty")
    try:
        return PROMPT_GROUPS[group_name][item_key]
    except KeyError as error:
        available = ", ".join(PROMPT_GROUPS[group_name])
        raise KeyError(
            f"Unknown prompt key {item_key!r} in group {group_name!r}; "
            f"available: {available}"
        ) from error


def _show_prompt_menu(
    profile: str,
    groups: Sequence[str],
    output_fn: Callable[[str], object],
) -> None:
    output_fn(f"\nPrompt profile: {_normalize_name(profile)}")
    for index, group_name in enumerate(groups):
        primary = " (primary; bare numbers select from this group)" if index == 0 else ""
        output_fn(f"[{group_name}]{primary}")
        for key, prompt in PROMPT_GROUPS[group_name].items():
            output_fn(f"  {key}: {prompt}")
    output_fn("[named]")
    for key, prompt in NAMED_PROMPTS.items():
        output_fn(f"  {key}: {prompt}")
    output_fn(
        "Enter a primary-group number, group:key, named:key, a custom prompt, or q."
    )


def _read_input(input_fn: Callable[[str], str], prompt: str) -> str | None:
    try:
        return input_fn(prompt)
    except (EOFError, KeyboardInterrupt):
        return None


def select_and_confirm_prompt(
    profile: str,
    supplied_prompt: str | None = None,
    prompt_key: str | None = None,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], object] = print,
) -> str:
    """Select a prompt and require explicit confirmation before execution.

    ``supplied_prompt`` is a literal and takes precedence over ``prompt_key``.
    Every candidate, including a supplied literal or resolved key, is shown and
    confirmed.  Only ``y``/``yes`` accepts it; ``q`` returns ``"q"``; every
    other response returns to the profile menu.
    """

    groups = _groups_for_profile(profile)
    reader = input if input_fn is None else input_fn
    candidate: str | None = None

    if supplied_prompt is not None:
        candidate = str(supplied_prompt)
    elif prompt_key is not None and str(prompt_key).strip():
        try:
            candidate = resolve_prompt_key(profile, prompt_key)
        except KeyError as error:
            output_fn(str(error))

    while True:
        if candidate is None:
            _show_prompt_menu(profile, groups, output_fn)
            selection = _read_input(reader, "Select prompt: ")
            if selection is None:
                return "q"
            selection = selection.strip()
            if selection.casefold() == "q":
                return "q"
            if not selection:
                continue

            try:
                candidate = resolve_prompt_key(profile, selection)
            except KeyError as error:
                # Key-shaped input is almost certainly a mistyped selector.  All
                # other text is accepted as a literal custom prompt.
                namespace = _normalize_name(selection.split(":", 1)[0])
                if (
                    selection.isdigit()
                    or namespace == "named"
                    or namespace in PROMPT_GROUPS
                ):
                    output_fn(str(error))
                    continue
                candidate = selection

        output_fn(f"\nPrompt: {candidate}")
        output_fn("Confirm with y/yes; q exits; any other answer returns to the menu.")
        confirmation = _read_input(reader, "Use this prompt and start execution? ")
        if confirmation is None:
            return "q"
        normalized = confirmation.strip().casefold()
        if normalized in {"y", "yes"}:
            return candidate
        if normalized == "q":
            return "q"
        candidate = None


__all__ = [
    "JOINT_DESK_PROMPTS",
    "JOINT_LEGACY_SPLIT_PROMPTS",
    "JOINT_DRESSING_TABLE_PROMPTS",
    "EE_5659_PROMPTS",
    "EE_LEGACY_DESK_PROMPTS",
    "BREAD_PROMPTS",
    "PROMPT_GROUPS",
    "PROFILE_GROUPS",
    "NAMED_PROMPTS",
    "resolve_prompt_key",
    "select_and_confirm_prompt",
]
