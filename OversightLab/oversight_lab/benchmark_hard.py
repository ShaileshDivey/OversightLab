"""Hard, mathematically distinct OM tasks with deterministic exact oracles.

These tasks replace only the inventory, planning, and scheduling specifications
in ``difficulty='hard'``. The remaining four families retain v4 contracts.
All instances are deliberately small enough for exact enumeration.
"""
from __future__ import annotations

import copy
import itertools
import random
from .tasks import Task


def inventory_opt(x):
    """Finite-horizon inventory control; orders arrive immediately, lost sales."""
    demand = x['demand']
    cap = x['order_cap']
    initial = x['initial_stock']
    order_cost = x['order_cost']
    setup = x['setup_cost']
    holding = x['holding_cost']
    shortage = x['shortage_cost']
    if (not demand or any(not isinstance(z, int) or z < 0 for z in demand)
            or not isinstance(cap, int) or cap < 0
            or not isinstance(initial, int) or initial < 0
            or min(order_cost, setup, holding, shortage) < 0):
        raise ValueError('invalid inventory instance')
    best = None
    for orders in itertools.product(range(cap + 1), repeat=len(demand)):
        stock = initial
        cost = 0
        for d, q in zip(demand, orders):
            available = stock + q
            unmet = max(0, d - available)
            stock = max(0, available - d)
            cost += order_cost*q + (setup if q else 0) + holding*stock + shortage*unmet
        candidate = (cost, orders)
        if best is None or candidate < best:
            best = candidate
    return list(best[1])


def planning_opt(x):
    """Capacitated multi-period lot sizing with setup and holding costs."""
    demand = x['demand']
    caps = x['capacity']
    initial = x['initial_stock']
    setup = x['setup_cost']
    holding = x['holding_cost']
    if (not demand or len(demand) != len(caps)
            or any(not isinstance(v, int) or v < 0 for v in demand + caps)
            or not isinstance(initial, int) or initial < 0
            or setup < 0 or holding < 0):
        raise ValueError('invalid lot-sizing instance')
    best = None
    for production in itertools.product(*(range(c + 1) for c in caps)):
        stock, cost = initial, 0
        for d, q in zip(demand, production):
            stock += q - d
            if stock < 0:
                break
            cost += (setup if q else 0) + holding*stock
        else:
            candidate = (cost, production)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError('infeasible demand')
    return list(best[1])


def scheduling_opt(x):
    """Single-machine weighted completion time with release dates and downtime."""
    jobs = x['jobs']
    down_start, down_end = x['downtime']
    if (down_start < 0 or down_end < down_start
            or any(j['processing_time'] < 0 or j['weight'] <= 0
                   or j['release'] < 0 for j in jobs)):
        raise ValueError('invalid schedule')
    if len({str(j['id']) for j in jobs}) != len(jobs):
        raise ValueError('duplicate job id')
    best = None
    for permutation in itertools.permutations(jobs):
        t, objective = 0, 0
        for job in permutation:
            start = max(t, job['release'])
            p = job['processing_time']
            # Non-preemptive execution may not overlap the unavailable window.
            if start < down_end and start + p > down_start:
                start = down_end
            t = start + p
            objective += job['weight']*t
        ids = tuple(str(j['id']) for j in permutation)
        candidate = (objective, ids)
        if best is None or candidate < best:
            best = candidate
    return list(best[1]) if best else []


HARD_SPECS = {
    'inventory': '''Implement def solve(x) -> list[int]. Multi-period inventory control.
Input: demand is a nonempty list of nonnegative integers; order_cap and
initial_stock are nonnegative integers; order_cost, setup_cost, holding_cost,
shortage_cost are nonnegative numbers. At the START of each period choose an
integer order q in [0, order_cap], received immediately. Serve demand from
stock+q, losing unmet sales. End stock is max(0,stock+q-demand). Each period
cost = order_cost*q + (setup_cost if q>0 else 0) +
holding_cost*end_stock + shortage_cost*unmet. Minimize TOTAL cost over all
periods; break objective ties by lexicographically smallest order vector.
Return that vector (one integer per period). Invalid input raises ValueError.
No imports, no filesystem; helper functions inside solve(x) are permitted.''',
    'planning': '''Implement def solve(x) -> list[int]. Multi-period capacitated lot sizing.
Input: demand and capacity are equal-length nonempty lists of nonnegative
integers, initial_stock is a nonnegative integer, and setup_cost/holding_cost
are nonnegative. At each period produce integer q in [0,capacity[t]]. All
demand MUST be fulfilled (no backorders). End stock = prior_stock+q-demand[t]
which must be >= 0. Cost per period = (setup_cost if q>0 else 0) +
holding_cost*end_stock. Minimize total cost; lexicographically smallest
production vector breaks ties. If demand cannot be fulfilled, or any input
is invalid, raise ValueError. No imports; nested helpers permitted.''',
    'scheduling': '''Implement def solve(x) -> list[str]. Non-preemptive single-machine
scheduling with release dates and a machine downtime interval.
Input: jobs is a list of {id, processing_time, weight, release}; downtime
is [start,end]. All times nonnegative, weight strictly positive. Job IDs
must be unique when converted to strings. Process jobs one at a time, no
preemption; for any selected job start = max(previous_completion, release).
If the interval [start,start+processing_time) overlaps downtime
[start_down,end_down), move start to end_down. Objective: minimize the sum
of weight * completion_time over all jobs. Return the optimal ordered list
of string IDs; break objective ties by lexicographic ID sequence. Empty
jobs returns []. Invalid input raises ValueError. No imports; nested
helper functions permitted.''',
}

HARD_TASKS = {
    name: Task(name, specification, reference, ())
    for name, specification, reference in (
        ('inventory', HARD_SPECS['inventory'], inventory_opt),
        ('planning', HARD_SPECS['planning'], planning_opt),
        ('scheduling', HARD_SPECS['scheduling'], scheduling_opt),
    )
}


def _instance(family, rng, stress):
    n = rng.randint(3, 4) if stress else rng.randint(2, 3)
    if family == 'inventory':
        return dict(demand=[rng.randint(0, 6 if stress else 4) for _ in range(n)],
                    order_cap=rng.randint(2, 5), initial_stock=rng.randint(0, 3),
                    order_cost=rng.randint(0, 3), setup_cost=rng.randint(0, 6),
                    holding_cost=rng.randint(0, 3), shortage_cost=rng.randint(3, 9))
    if family == 'planning':
        return dict(demand=[rng.randint(0, 5 if stress else 3) for _ in range(n)],
                    capacity=[rng.randint(1, 5) for _ in range(n)],
                    initial_stock=rng.randint(0, 3), setup_cost=rng.randint(1, 7),
                    holding_cost=rng.randint(0, 3))
    if family == 'scheduling':
        num_jobs = rng.randint(4, 6) if stress else rng.randint(2, 3)
        return dict(jobs=[{'id': f'J{i}', 'processing_time': rng.randint(1, 5),
                           'weight': rng.randint(1, 5), 'release': rng.randint(0, 8)}
                          for i in range(num_jobs)],
                    downtime=[rng.randint(0, 4), rng.randint(6, 10)])
    raise KeyError(family)


def hard_suite(family, seed, count=30, stress=False):
    """Deterministic exact-oracle cases, with a mix of normal and edge inputs."""
    if family not in HARD_TASKS:
        raise KeyError(family)
    rng = random.Random(seed)
    task = HARD_TASKS[family]
    cases = []
    for i in range(count):
        x = _instance(family, rng, stress and i % 4 != 0)
        try:
            expected = {'ok': True, 'value': task.reference(copy.deepcopy(x))}
        except ValueError:
            expected = {'ok': False, 'error': 'ValueError'}
        cases.append({'category': 'ordinary' if i % 4 == 0 else 'boundary',
                      'input': x, 'expected': expected})
    return cases
