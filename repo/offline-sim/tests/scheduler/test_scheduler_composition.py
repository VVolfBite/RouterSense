import pytest
from rs_sim.scheduler.decorators.composition import parse_algorithm_expression
from rs_sim.scheduler.planning.planner import PlannerScope
from rs_sim.scheduler.decorators.planning_gate import PlanningMode


def test_parse_composed_algorithm():
    item=parse_algorithm_expression('safe(joint(global_(rscf())))')
    assert item.core_id=='rscf'
    assert item.scope is PlannerScope.WINDOW_JOINT
    assert item.planning is PlanningMode.GLOBAL
    assert item.safe is True
    assert item.expression=='safe(joint(global_(rscf())))'


def test_local_event_composition():
    item=parse_algorithm_expression('local(event(fifo()))')
    assert item.core_id=='fifo'
    assert item.expression=='local(event(fifo()))'


@pytest.mark.parametrize('value',[
    'rscf()',
    'joint(rscf())',
    'global_(rscf())',
    'safe(local(global_(rscf())))',
    'joint(global_(rscf_v15()))',
])
def test_reject_incomplete_or_historical_composition(value):
    with pytest.raises(ValueError):
        parse_algorithm_expression(value)
