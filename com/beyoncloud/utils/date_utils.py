from datetime import datetime
import com.beyoncloud.config.settings.env_config as config

def get_current_date_string() -> str:
    current_date = datetime.now().strftime(config.COMMON_CONFIG["dateformat"])
    return current_date

def get_current_timestamp_string() -> str:
    current_timestamp = datetime.now().strftime(config.COMMON_CONFIG["timestamp_format"])
    return current_timestamp

def current_timestamp_trim() -> str:
    """
    Returns current date-time as a compact string: DDMMYYYYHHMMSS
    Example: 10/09/2025 13:09:25 -> 10092025130925
    """
    now = datetime.now()
    # Format as DDMMYYYYHHMMSS
    compact = now.strftime(config.COMMON_CONFIG["timestamp_trim"])
    return compact

def current_date_trim() -> str:
    """
    Returns current date as a compact string: DDMMYYYY
    Example: 21/09/2025 13:09:25 -> 21092025
    """
    now = datetime.now()
    # Format as DDMMYYYY
    compact = now.strftime(config.COMMON_CONFIG["date_trim"])
    return compact