from datetime import datetime
import com.beyoncloud.config.settings.env_config as config

def get_current_date_string() -> str:
    currDate = datetime.now().strftime(config.COMMON_CONFIG.dateformat)
    return currDate

def current_timestamp_trim() -> str:
    """
    Returns current date-time as a compact string: DDMMYYYYHHMMSS
    Example: 10/09/2025 13:09:25 -> 10092025130925
    """
    now = datetime.now()
    # Format as DDMMYYYYHHMMSS
    compact = now.strftime("%d%m%Y%H%M%S")
    return compact

def current_date_trim() -> str:
    """
    Returns current date as a compact string: DDMMYYYY
    Example: 21/09/2025 13:09:25 -> 21092025
    """
    now = datetime.now()
    # Format as DDMMYYYY
    compact = now.strftime("%d%m%Y")
    return compact