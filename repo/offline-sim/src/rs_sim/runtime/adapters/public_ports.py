from __future__ import annotations

"""Read-only adapters over backend/scheduler public state for formal transport Wave A.

The adapters prevent transport from importing Scheduler/Backend private containers.
They do not build a full runtime; Wave B Integration Owner owns wiring.
"""

from dataclasses import dataclass
from typing import Any

from rs_sim import (
    AuthorityStamp,
    CanonicalTransferTask,
    NetworkTopology,
    PhaseKey,
    ReceivePermit,
    TaskResourceFootprint,
    make_task_resource_footprint,
)


@dataclass(frozen=True, slots=True)
class CatalogueTaskLookup:
    catalogue: Any

    def task(self, task_id: str) -> CanonicalTransferTask:
        task = self.catalogue.get(str(task_id))
        if not isinstance(task, CanonicalTransferTask):
            raise TypeError("formal TaskLookupPort requires CanonicalTransferTask")
        return task


@dataclass(frozen=True, slots=True)
class ReceiverPermitLookup:
    receiver: Any

    def permit(self, task_id: str) -> ReceivePermit | None:
        permit = self.receiver.receive_permit(str(task_id))
        if permit is not None and not isinstance(permit, ReceivePermit):
            raise TypeError("formal PermitLookupPort requires ReceivePermit")
        return permit


@dataclass(frozen=True, slots=True)
class PhaseAuthorityValidation:
    authority: Any

    def authority_is_current(
        self, *, phase_key: PhaseKey, authority_stamp: AuthorityStamp
    ) -> bool:
        return bool(self.authority.stamp_is_current(phase_key, authority_stamp))


@dataclass(frozen=True, slots=True)
class SharedTopologyTaskResolver:
    topology: NetworkTopology

    def footprint(self, task: CanonicalTransferTask) -> TaskResourceFootprint:
        if not isinstance(task, CanonicalTransferTask):
            raise TypeError("formal resource resolver requires CanonicalTransferTask")
        return make_task_resource_footprint(
            task_id=task.task_id,
            src_rank=task.src_rank,
            dst_rank=task.dst_rank,
            topology=self.topology,
        )


__all__ = [
    "CatalogueTaskLookup",
    "PhaseAuthorityValidation",
    "ReceiverPermitLookup",
    "SharedTopologyTaskResolver",
]
