"""Minimal coloured-print helpers replacing the xerrors.cprint dependency."""

RESET = "\033[0m"


class _CP:
    def green(self, text: object) -> str:
        return f"\033[32m{text}{RESET}"

    def blue(self, text: object) -> str:
        return f"\033[34m{text}{RESET}"

    def yellow(self, text: object) -> str:
        return f"\033[33m{text}{RESET}"

    def warning(self, text: object) -> str:
        return f"\033[33m[WARNING] {text}{RESET}"

    def blue_background(self, text: object) -> str:
        return f"\033[44m{text}{RESET}"

    def red(self, text: object) -> str:
        return f"\033[31m{text}{RESET}"


cp = _CP()
