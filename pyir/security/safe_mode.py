safe_mode: bool = False

def set_safe_mode(enabled: bool = True):
    """
    Enable or disable safe mode globally.
    """
    global safe_mode
    safe_mode = enabled

__all__ = ['safe_mode', 'set_safe_mode']