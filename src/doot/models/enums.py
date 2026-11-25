"""Domain enumerations and enforced job state machine."""

from enum import StrEnum


class ATSProvider(StrEnum):
    """Supported Applicant Tracking System platforms."""

    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"
    CUSTOM = "custom"


class CompanySource(StrEnum):
    """Origin of a company record in the watchlist."""

    MANUAL = "manual"
    DORK = "dork"
    SEED = "seed"


class RunType(StrEnum):
    """Tracked pipeline execution types."""

    DISCOVER = "discover"
    DORK = "dork"
    SCORE = "score"
    DIGEST = "digest"
    APPLY = "apply"


class RunStatus(StrEnum):
    """Execution status of a pipeline run."""

    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class JobStatus(StrEnum):
    """Lifecycle states of a discovered job posting."""

    DISCOVERED = "DISCOVERED"
    FILTERED_OUT = "FILTERED_OUT"
    SCORED = "SCORED"
    SKIPPED = "SKIPPED"
    NOTIFIED = "NOTIFIED"
    PREPARED = "PREPARED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPLIED = "APPLIED"
    FAILED = "FAILED"
    NEEDS_MANUAL_ACTION = "NEEDS_MANUAL_ACTION"


ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.DISCOVERED: frozenset(
        {JobStatus.FILTERED_OUT, JobStatus.SCORED, JobStatus.FAILED}
    ),
    JobStatus.FILTERED_OUT: frozenset(),
    JobStatus.SCORED: frozenset({JobStatus.SKIPPED, JobStatus.NOTIFIED, JobStatus.FAILED}),
    JobStatus.SKIPPED: frozenset(),
    JobStatus.NOTIFIED: frozenset(
        {
            JobStatus.PREPARED,
            JobStatus.SKIPPED,
            JobStatus.APPLIED,
            JobStatus.NEEDS_MANUAL_ACTION,
        }
    ),
    JobStatus.PREPARED: frozenset(
        {
            JobStatus.AWAITING_APPROVAL,
            JobStatus.NEEDS_MANUAL_ACTION,
            JobStatus.FAILED,
            JobStatus.SKIPPED,
        }
    ),
    JobStatus.AWAITING_APPROVAL: frozenset(
        {
            JobStatus.APPLIED,
            JobStatus.PREPARED,
            JobStatus.FAILED,
            JobStatus.NEEDS_MANUAL_ACTION,
            JobStatus.SKIPPED,
        }
    ),
    JobStatus.FAILED: frozenset(
        {
            JobStatus.PREPARED,
            JobStatus.SCORED,
            JobStatus.FILTERED_OUT,
            JobStatus.NEEDS_MANUAL_ACTION,
            JobStatus.SKIPPED,
        }
    ),
    JobStatus.NEEDS_MANUAL_ACTION: frozenset(
        {
            JobStatus.APPLIED,
            JobStatus.SKIPPED,
            JobStatus.PREPARED,
        }
    ),
    JobStatus.APPLIED: frozenset(),
}
