"""Scheduling contract errors."""


class SchedulingContractError(RuntimeError):
    """Base class for fail-closed scheduling contract violations."""


class SharedSchemaError(SchedulingContractError):
    """The injected shared-schema schema does not satisfy the frozen field contract."""


class TaskizationError(SchedulingContractError):
    """Canonical task ranges or registrations are invalid."""


class AuthorityError(SchedulingContractError):
    """Plan authority or task-state transition is invalid."""


class BindingError(SchedulingContractError):
    """Prepared-to-real binding is not total and one-to-one."""


class CompilationError(SchedulingContractError):
    """A batch cannot be compiled under the current contract."""


class CatalogueSealError(SchedulingContractError):
    """A GLOBAL phase catalogue was sealed too early or mutated after seal."""


class FormalRuntimeError(SchedulingContractError):
    """The formal scheduling service-line runtime violated a causal contract."""
