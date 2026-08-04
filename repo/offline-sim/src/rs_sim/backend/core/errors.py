"""Fail-closed backend backend errors.

These are backend-owned exceptions. They intentionally do not duplicate the
shared schema or Transport submit result enums.
"""


class BackendError(RuntimeError):
    """Base class for deterministic backend contract failures."""


class BackendContractError(BackendError):
    """The caller violated an immutable shared schema/backend contract."""


class DuplicateRegistrationError(BackendContractError):
    """An immutable object was registered twice with different content."""


class UnknownObjectError(BackendContractError):
    """An event referenced an object not known to the backend."""


class IllegalTransitionError(BackendContractError):
    """A state transition was attempted out of causal order."""


class CapacityConfigurationError(BackendContractError):
    """The fixed staging capacity cannot hold one canonical task."""
