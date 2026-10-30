"""KnobParameters: hold line variables as GTPSA parameters, and restore them.

Assigning a parameter to a line variable makes xdeps re-evaluate the dependent
expressions in the TPSA algebra, so the driven element strengths hold Taylor series in
the knob and a map tracked on the same descriptor carries d(coord)/d(knob).
"""

from __future__ import annotations

from typing import Any

from madng_tpsa import Descriptor, Tpsa


def _scalar_value(value: Any) -> float:
    if isinstance(value, Tpsa):
        return float(value.const_part)
    return float(value)


class KnobParameters:
    """Line variables ``names`` held as parameters ``1..len(names)`` of ``descriptor``.

    ``apply()`` puts the parameters in, ``refresh(values)`` moves them to new knob
    values, ``teardown()`` restores the knob values to plain doubles and disables
    only element owners promoted by this parameter context. Owners that were
    already in TPSA mode remain enabled, preserving unrelated parameter series.

    While the parameters are in, the knobs must be moved with ``refresh``: assigning a
    float to ``line.vars`` re-propagates constants and drops the knob dependence.
    Keep the dependency topology fixed until teardown; refresh changes only the
    knob base points and does not acquire newly added expression owners.
    """

    def __init__(self, line: Any, names: list[str], descriptor: Descriptor) -> None:
        self.line = line
        self.names = list(names)
        self.descriptor = descriptor
        for name in self.names:
            if name not in line.vars:
                raise KeyError(f"knob {name!r} is not a line variable")
        if descriptor.num_params != len(self.names):
            raise ValueError(
                f"descriptor has {descriptor.num_params} parameters, "
                f"expected {len(self.names)} (one per knob)"
            )
        self._applied = False
        self._reached = None      # element names the knob expressions reach
        self._initially_enabled = None

    def __len__(self) -> int:
        return len(self.names)

    @property
    def applied(self) -> bool:
        """Are the parameters currently in the line variables?"""
        return self._applied

    def apply(self, values: list[float] | None = None) -> None:
        """Assign parameter ``k+1`` to knob ``k``, seeded at its current value."""
        if values is None:
            values = [_scalar_value(self.line[name]) for name in self.names]
        values = [_scalar_value(value) for value in values]
        if len(values) != len(self.names):
            raise ValueError("one value is required for each knob")
        if not self._applied:
            # A shared environment may already contain another active map.
            # Own only promotions caused by this parameter context; a shared
            # variable may legitimately reach an owner outside this line.
            self._initially_enabled = {
                id(element) for element in self.line._element_dict.values()
                if getattr(getattr(element, '_xobject', None), '_tpsa_enabled', 0)
            }
            self._reached = None
        for index, (name, value) in enumerate(zip(self.names, values), start=1):
            self.line.vars[name] = self.descriptor.param(index, _scalar_value(value))
        self._applied = True
        if self._reached is None:
            # The expression topology is fixed, so the reached elements are too.
            self._reached = sorted({
                name for name, element in self.line._element_dict.items()
                if id(element) not in self._initially_enabled
                and getattr(getattr(element, '_xobject', None), '_tpsa_enabled', 0)
            })

    def refresh(self, values: list[float]) -> None:
        """Move active parameters to new base points, preserving their seeds."""
        if not self._applied:
            raise RuntimeError("apply the knob parameters before refreshing them")
        self.apply(values)

    def teardown(self) -> None:
        """Restore doubles and release only this context's new TPSA owners."""
        if not self._applied:
            return
        self.apply_doubles([_scalar_value(self.line[name]) for name in self.names])

    def apply_doubles(self, values: list[float]) -> None:
        """Set knobs to floats and release this context's newly promoted owners."""
        values = [_scalar_value(value) for value in values]
        if len(values) != len(self.names):
            raise ValueError("one value is required for each knob")
        for name, value in zip(self.names, values):
            self.line.vars[name] = _scalar_value(value)
        # xdeps re-propagates the new floats, but an enabled element stores them as
        # constant series, so the switch back to doubles is explicit by disabling tpsa
        # for the reached elements.
        elements = self.line.element_dict
        for name in (self._reached or ()):
            element = elements[name]
            if getattr(getattr(element, "_xobject", None), "_tpsa_enabled", 0):
                element.disable_tpsa()
        self._applied = False
        self._reached = None
        self._initially_enabled = None

    def _used_owners(self):
        """Canonical used elements and their slice/replica parent closure."""
        pending = list(self.line.element_names)
        seen = set()
        while pending:
            name = pending.pop()
            element = self.line._element_dict[name]
            if id(element) in seen:
                continue
            seen.add(id(element))
            yield name, element
            parent = getattr(element, 'parent_name', None)
            if parent is not None:
                pending.append(parent)

    def driven_elements(self) -> list[tuple[str, str]]:
        """``(element, attribute)`` of every field currently holding a series."""
        driven = []
        for name, element in self._used_owners():
            if not getattr(getattr(element, "_xobject", None), "_tpsa_enabled", 0):
                continue
            for attr in getattr(element, "_float_or_tpsa_fields", ()):
                driven.append((name, attr))
        return sorted(driven)

    def strength_jacobian(self) -> dict[tuple[str, str], list[float]]:
        """``d strength / d knob`` per driven field.

        Distinct from ``ParticlesTpsa.param_jacobian`` (``d coord / d knob``): this is the
        upstream element-strength sensitivity that seeds the map.
        """
        jacobian = {}
        for element_name, attr in self.driven_elements():
            element = self.line._element_dict[element_name]
            value = getattr(element, element._rename.get(attr, attr))
            if isinstance(value, Tpsa):
                jacobian[(element_name, attr)] = value.param_grad()
        return jacobian

    def __repr__(self) -> str:
        return f"KnobParameters(names={self.names}, applied={self._applied})"
