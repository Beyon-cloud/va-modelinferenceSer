from datetime import datetime
import com.beyoncloud.config.settings.env_config as config

def getCurrentDateString() -> str:
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