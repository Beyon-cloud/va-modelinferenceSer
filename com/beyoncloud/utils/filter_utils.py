from com.beyoncloud.common.constants import RepositoryKeys, RepositoryOps


class FilterUtils:
    """
    Utility class for building standardized filter dictionaries
    for database queries or repository operations.

    Author:
        Balaji G.R (24-07-2025)

    Description:
        Provides static helper methods to generate common filter
        dictionaries (eq, ne, like, in) that can be passed to
        repository layers for dynamic query building.

    Each method returns a dictionary with the keys:
    - FIELD → Name of the model field
    - OP → The operation to perform (e.g., =, !=, LIKE, IN)
    - VALUE → The value(s) to filter against

    Example:
    --------
    ```python
    FilterUtils.eq("status", "ACTIVE")
    # ➝ {"FIELD": "status", "OP": "=", "VALUE": "ACTIVE"}
    ```
    """

    @staticmethod
    def eq(model_field: str, value):
        """Create an equality filter (field = value)."""
        return {
            RepositoryKeys.FIELD: model_field,
            RepositoryKeys.OPERATION: RepositoryOps.EQUALS,
            RepositoryKeys.VALUE: value,
        }

    @staticmethod
    def ne(model_field: str, value):
        """Create a not-equal filter (field != value)."""
        return {
            RepositoryKeys.FIELD: model_field,
            RepositoryKeys.OPERATION: RepositoryOps.NOT_EQUALS,
            RepositoryKeys.VALUE: value,
        }

    @staticmethod
    def like(model_field: str, value):
        """Create a LIKE filter (field LIKE value)."""
        return {
            RepositoryKeys.FIELD: model_field,
            RepositoryKeys.OPERATION: RepositoryOps.LIKE,
            RepositoryKeys.VALUE: value,
        }

    @staticmethod
    def in_list(model_field: str, values: list):
        """Create an IN filter (field IN values)."""
        return {
            RepositoryKeys.FIELD: model_field,
            RepositoryKeys.OPERATION: RepositoryOps.IN,
            RepositoryKeys.VALUE: values,
        }
