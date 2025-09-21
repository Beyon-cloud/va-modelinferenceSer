from pydantic import BaseModel, Field, StrictInt, StrictStr
from typing import Self
from enum import Enum
from com.beyoncloud.common.constants import Numerical, CommonPatterns

class CommonResponse(BaseModel):
    """
    Represents the common response structure for any operations.

    Attributes:
        status (str): The status of the task (e.g., STARTED, SUCCESS, FAILED).
        message (str): It containing additional details about the response.

    Author: Jenson J (10-09-2025)
    """

    status: str = Field(..., description="The status of the task.")
    message: str = Field(..., description="Additional details about the task response.")

class CommonResponseBuilder:
    def __init__(self):
        self.common_response = CommonResponse(
            status = CommonPatterns.EMPTY_SPACE,
            message = CommonPatterns.EMPTY_SPACE
        )

    def with_status(self, status: str) -> Self:
        self.common_response.status = status
        return self

    def with_message(self, message: str) -> Self:
        self.common_response.message = message
        return self

    def build(self) -> CommonResponse:
        return self.common_response