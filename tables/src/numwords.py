"""Number words for prose macros (user rule 2026-10-07): every integer count from zero to ten is spelled out in prose,
counts above ten stay digits, and a ratio k/n becomes "k of n" ("all n" when k == n).
Numbers with units or decimals, metrics, settings, seed indices and table cells keep digits; the generators call these
helpers for macros that are read in prose only."""

WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']


def num_word(n):
    """0-10 -> word, larger integers -> digits."""
    n = int(n)
    return WORDS[n] if 0 <= n <= 10 else str(n)


def k_of_n(k, n):
    """Count k out of n in prose: 'all four' if k == n, 'none of the three'-style is left to the caller, else 'k of n'."""
    k, n = int(k), int(n)
    if k == n:
        return f'all {num_word(n)}'
    return f'{num_word(k)} of {num_word(n)}'
