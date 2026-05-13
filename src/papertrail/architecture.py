"""Abstract base class for every PaperTrail architecture.

Each architecture (arch_00 through arch_06) inherits from ``Architecture`` and
implements ``async run(topic, ...) -> BenchmarkResult``. The benchmark harness
is generic over this base class: feed it any ``Architecture`` instance, get a
comparable result. This is the foundation of the D1 (Agentic Architecture)
side of the project — every orchestration pattern plugs into the same socket.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from papertrail.benchmark import BenchmarkResult, Modes


class Architecture(ABC):
    """Base class every PaperTrail architecture must subclass.

    Concrete subclasses MUST declare three class-level attributes:

    - ``name``         stable identifier, e.g. ``"arch_01_sequential"``
    - ``version``      semver of this architecture's wiring (bumps often)
    - ``description``  short human-readable explanation of the pattern

    They MUST also implement ``async run(topic, ...)``.

    The ABC pattern gives us three things at once:

    1. Python refuses to instantiate the class unless ``run()`` is implemented
       (``TypeError`` at construction time).
    2. ``mypy --strict`` flags any subclass whose ``run()`` signature drifts.
    3. ``__init_subclass__`` (below) checks ClassVars at class-creation time
       so a missing ``name`` fails on ``import``, not on first use.
    """

    # ClassVar so mypy treats these as class-level, not instance-level.
    # Subclasses re-declare them with concrete string values.
    name: ClassVar[str]
    version: ClassVar[str]
    description: ClassVar[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Verify concrete subclasses declared the required ClassVars.

        Runs at class-creation time (when Python first imports the subclass).
        We skip the check on intermediate abstract subclasses — those legally
        defer attribute declaration to their concrete descendants.

        Why this exists: ``mypy`` catches missing ClassVars at typecheck time,
        but ``mypy`` is opt-in. This check is unconditional and lives in the
        runtime, so even a script that bypasses static analysis still fails
        fast and with a clear message.
        """
        super().__init_subclass__(**kwargs)
        # Abstract intermediates (still have unimplemented abstract methods)
        # are exempt — they're not meant to be instantiated, so they're not
        # required to declare the trio yet.
        if getattr(cls, "__abstractmethods__", None):
            return
        for attr in ("name", "version", "description"):
            value = getattr(cls, attr, None)
            if not isinstance(value, str) or not value:
                raise TypeError(
                    f"{cls.__name__} must declare ClassVar[str] '{attr}' with a non-empty value"
                )

    @abstractmethod
    async def run(
        self,
        topic: str,
        *,
        prompt_versions: dict[str, str] | None = None,
        modes: Modes | None = None,
    ) -> BenchmarkResult:
        """Execute the architecture against ``topic`` and return a result.

        Args:
            topic: The research topic to investigate.
            prompt_versions: Per-agent prompt version selection. ``None`` means
                "use the latest version of each prompt". The subclass decides
                which prompt names are valid; unknown names should raise.
            modes: Architecture-internal knobs (tools on/off, target paper
                count). ``None`` means ``Modes()`` defaults.

        Returns:
            A fully populated ``BenchmarkResult``. The architecture is
            responsible for filling ``deliverable``, ``telemetry``, and
            ``provenance`` — the harness does not stitch them together.

        Notes:
            ``async`` is non-negotiable: arch_03 (parallel fan-out) and
            arch_06 (hierarchical) run agents concurrently, and the project
            stack (httpx, asyncio) is async throughout. A sync architecture
            would be a footgun, not a simplification.
        """
        raise NotImplementedError  # pragma: no cover
