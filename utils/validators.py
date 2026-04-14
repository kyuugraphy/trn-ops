import re


def validate_iban(iban: str) -> tuple[bool, str]:
    """Validate an IBAN using the mod-97 algorithm.

    Returns (is_valid, message).
    """
    if not iban or not iban.strip():
        return False, "IBAN is required"

    cleaned = iban.replace(" ", "").replace("-", "").upper()

    if len(cleaned) < 15 or len(cleaned) > 34:
        return False, f"IBAN length must be 15-34 characters (got {len(cleaned)})"

    if not re.match(r"^[A-Z]{2}\d{2}", cleaned):
        return False, "IBAN must start with 2-letter country code + 2 check digits"

    if not re.match(r"^[A-Z0-9]+$", cleaned):
        return False, "IBAN contains invalid characters"

    rearranged = cleaned[4:] + cleaned[:4]
    numeric = ""
    for ch in rearranged:
        if ch.isdigit():
            numeric += ch
        else:
            numeric += str(ord(ch) - ord("A") + 10)

    if int(numeric) % 97 != 1:
        return False, "IBAN check digit verification failed"

    return True, "Valid IBAN"


def validate_ico(ico: str) -> tuple[bool, str]:
    """Validate Czech ICO (8-digit company identifier). Empty is allowed."""
    if not ico or not ico.strip():
        return True, ""
    cleaned = ico.strip()
    if not re.match(r"^\d{8}$", cleaned):
        return False, "ICO must be exactly 8 digits"
    return True, "Valid ICO"


def validate_rc(rc: str) -> tuple[bool, str]:
    """Validate Czech RC (9-10 digit birth number). Empty is allowed."""
    if not rc or not rc.strip():
        return True, ""
    cleaned = rc.strip().replace("/", "")
    if not re.match(r"^\d{9,10}$", cleaned):
        return False, "RC must be 9-10 digits"
    return True, "Valid RC"
