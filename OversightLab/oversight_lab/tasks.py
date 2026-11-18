"""Seven connected operational decision kernels and independently derived test fixtures.
All references are deterministic; generated tasks vary inputs without changing contracts.
"""
from dataclasses import dataclass
from typing import Any
import copy, math, random

@dataclass(frozen=True)
class Task:
    name: str
    specification: str
    reference: Any
    cases: tuple


def analytics(x):
    values=x['sales']; w=x['window']
    if not isinstance(w,int) or w<=0: raise ValueError('window')
    if not values or any(v<0 for v in values): raise ValueError('sales')
    return round(sum(values[-w:])/len(values[-w:]), 4)


def inventory(x):
    d,s,L,z=(x[k] for k in ('daily_mean','daily_std','lead_time','z'))
    if min(d,s,L,z)<0: raise ValueError('invalid')
    return round(d*L+z*s*math.sqrt(L),4)


def optimization(x):
    """Fractional resource allocation in descending margin per unit of machine time."""
    cap=x['capacity']; items=x['products']
    if cap<0 or any(p['hours']<=0 or p['margin']<0 or p['max_units']<0 for p in items): raise ValueError('invalid')
    remain=float(cap); out={str(p['id']):0.0 for p in items}
    for p in sorted(items,key=lambda p:(-p['margin']/p['hours'],str(p['id']))):
        n=min(float(p['max_units']),remain/p['hours']);out[str(p['id'])]=round(n,4);remain=max(0,remain-n*p['hours'])
    return out


def planning(x):
    """Feasible myopic capped production based on allocated machine-hours and raw materials."""
    if any(x[k]<0 for k in ('forecast','on_hand','raw_material','allocated_hours')) or x['hours_per_unit']<=0: raise ValueError('invalid')
    need=max(0,math.ceil(x['forecast']-x['on_hand']))
    return int(min(need,x['raw_material'],math.floor(x['allocated_hours']/x['hours_per_unit'])))


def scheduling(x):
    jobs=x['jobs']
    if any(j['processing_time']<0 or j['weight']<=0 for j in jobs): raise ValueError('invalid')
    return [j['id'] for j in sorted(jobs,key=lambda j:(j['processing_time']/j['weight'],str(j['id'])))]


def logistics(x):
    """First-fit decreasing: sort orders by descending volume, ID for ties; vehicles retain input order."""
    caps=x['vehicle_capacities']; orders=x['orders']
    if any(c<0 for c in caps) or any(o['units']<0 for o in orders): raise ValueError('invalid')
    room=list(caps); out={str(o['id']):-1 for o in orders}
    for o in sorted(orders,key=lambda v:(-v['units'],str(v['id']))):
        for i,c in enumerate(room):
            if o['units']<=c:
                out[str(o['id'])]=i;room[i]-=o['units'];break
    return out


def finance(x):
    layers=x['layers']; sold=x['units_sold']
    if sold<0 or any(q<0 or price<0 for q,price in layers) or sold>sum(q for q,p in layers): raise ValueError('invalid')
    left=sold; costs=0; rest=[]
    for q,price in layers:
        use=min(q,left); costs+=use*price;left-=use
        if q>use: rest.append([q-use,price])
    return {'cogs':round(costs,4),'remaining':rest}


def cases(rows):
    # (category, input) tuples; expected values calculated by distinct trusted reference at build time
    return tuple((kind,copy.deepcopy(x)) for kind,x in rows)

TASK_REGISTRY={
 'analytics':Task('analytics', '''Implement def solve(x): where x has sales (nonempty list of nonnegative daily numbers) and window (positive integer). Return the simple arithmetic mean of the LAST min(window,len(sales)) days, rounded to 4 decimals. Invalid input raises ValueError. No imports. Return a Python scalar.''',analytics,cases([
 ('ordinary',{'sales':[4,6,10,8],'window':2}),('ordinary',{'sales':[1,2,3,4,5],'window':3}),('ordinary',{'sales':[3,6,9],'window':1}),
 ('boundary',{'sales':[0],'window':20}),('boundary',{'sales':[2,4],'window':9}),('invalid',{'sales':[],'window':2}),('invalid',{'sales':[4,5],'window':0})])),
 'inventory':Task('inventory', '''Implement def solve(x): given nonnegative daily_mean, daily_std, lead_time, and z, return round(daily_mean*lead_time+z*daily_std*sqrt(lead_time),4). Raise ValueError if any input is negative. No imports (math.sqrt is not available; use **0.5).''',inventory,cases([
 ('ordinary',{'daily_mean':5,'daily_std':2,'lead_time':4,'z':1.5}),('ordinary',{'daily_mean':10,'daily_std':3,'lead_time':2,'z':1}),('ordinary',{'daily_mean':6,'daily_std':1,'lead_time':9,'z':2}),
 ('boundary',{'daily_mean':0,'daily_std':2,'lead_time':0,'z':1}),('boundary',{'daily_mean':2,'daily_std':0,'lead_time':3,'z':1}),('invalid',{'daily_mean':-1,'daily_std':1,'lead_time':3,'z':1})])),
 'optimization':Task('optimization', '''Implement def solve(x): fractional capacity allocation. Input capacity >=0 and products list of {id,margin,hours,max_units}; hours>0, margin and max_units >=0. Maximize total margin, subject to sum(hours*units)<=capacity and units<=max_units. Use greedy descending margin/hours with alphabetical string ID ties. Return dict {str(id): round(float(units),4)} including zeros. Invalid values raise ValueError. No imports.''',optimization,cases([
 ('ordinary',{'capacity':5,'products':[{'id':'A','margin':10,'hours':2,'max_units':2},{'id':'B','margin':4,'hours':1,'max_units':5}]}),
 ('ordinary',{'capacity':9,'products':[{'id':'A','margin':4,'hours':1,'max_units':3},{'id':'B','margin':15,'hours':3,'max_units':5}]}),
 ('ordinary',{'capacity':8,'products':[{'id':'A','margin':6,'hours':2,'max_units':5}]}),
 ('boundary',{'capacity':0,'products':[{'id':'A','margin':4,'hours':2,'max_units':5}]}),
 ('boundary',{'capacity':5,'products':[]}),('invalid',{'capacity':3,'products':[{'id':'A','margin':2,'hours':0,'max_units':5}]})])),
 'planning':Task('planning', '''Implement def solve(x): forecast,on_hand,raw_material,allocated_hours must be nonnegative; hours_per_unit must be positive. Return integer max feasible units: min(max(0, ceil(forecast-on_hand)),raw_material,floor(allocated_hours/hours_per_unit)). Assume raw_material is an integer number of units. Invalid values raise ValueError. No imports; write ceil/floor logic with built-ins.''',planning,cases([
 ('ordinary',{'forecast':9,'on_hand':2,'raw_material':12,'allocated_hours':20,'hours_per_unit':2}),('ordinary',{'forecast':12,'on_hand':5,'raw_material':3,'allocated_hours':20,'hours_per_unit':2}),('ordinary',{'forecast':12,'on_hand':2,'raw_material':10,'allocated_hours':5,'hours_per_unit':2}),
 ('boundary',{'forecast':1.5,'on_hand':1,'raw_material':4,'allocated_hours':3,'hours_per_unit':1}),('boundary',{'forecast':0,'on_hand':5,'raw_material':4,'allocated_hours':3,'hours_per_unit':1}),('invalid',{'forecast':3,'on_hand':1,'raw_material':5,'allocated_hours':1,'hours_per_unit':0})])),
 'scheduling':Task('scheduling', '''Implement def solve(x): x['jobs'] is list of dictionaries {id,processing_time,weight}. Single machine, all jobs available at time zero. Return list of IDs sorted ascending by processing_time/weight; ties broken by str(id) alphabetically. processing_time >=0 and weight >0, else ValueError. Empty jobs returns []. No imports.''',scheduling,cases([
 ('ordinary',{'jobs':[{'id':'B','processing_time':3,'weight':1},{'id':'A','processing_time':2,'weight':2}]}),
 ('ordinary',{'jobs':[{'id':'C','processing_time':5,'weight':1},{'id':'A','processing_time':2,'weight':1},{'id':'B','processing_time':4,'weight':4}]}),
 ('ordinary',{'jobs':[{'id':'X','processing_time':3,'weight':3}]}),
 ('boundary',{'jobs':[{'id':'B','processing_time':4,'weight':2},{'id':'A','processing_time':2,'weight':1}]}),('boundary',{'jobs':[]}),('invalid',{'jobs':[{'id':'A','processing_time':1,'weight':0}]})])),
 'logistics':Task('logistics', '''Implement def solve(x): x has vehicle_capacities list and orders list of {id,units}. Use first-fit decreasing: consider orders by descending units, then str(id) ascending; place each order in FIRST vehicle (original order) with enough remaining capacity; otherwise assign -1. Return dict {str(order_id): vehicle_index_or_minus_1}. Do not split orders. Any negative units/capacity raises ValueError. No imports.''',logistics,cases([
 ('ordinary',{'vehicle_capacities':[5,4],'orders':[{'id':'A','units':3},{'id':'B','units':4},{'id':'C','units':2}]}),
 ('ordinary',{'vehicle_capacities':[6],'orders':[{'id':'B','units':1},{'id':'A','units':4}]}),
 ('ordinary',{'vehicle_capacities':[2,2],'orders':[{'id':'A','units':3},{'id':'B','units':2}]}),
 ('boundary',{'vehicle_capacities':[],'orders':[{'id':'A','units':1}]}),('boundary',{'vehicle_capacities':[0],'orders':[]}),('invalid',{'vehicle_capacities':[-1],'orders':[]})])),
 'finance':Task('finance', '''Implement def solve(x): FIFO. x has layers as [[quantity,unit_cost],...] oldest first and units_sold. If units_sold<0, any layer contains negative quantity/cost, or units_sold exceeds total quantity, raise ValueError. Return {'cogs':round(cost,4),'remaining':[[remaining_qty,unit_cost],...]}; omit fully consumed layers but preserve unconsumed layers in original order. No imports.''',finance,cases([
 ('ordinary',{'layers':[[10,50],[15,55]],'units_sold':12}),('ordinary',{'layers':[[3,7],[5,8]],'units_sold':2}),('ordinary',{'layers':[[6,4]],'units_sold':6}),
 ('boundary',{'layers':[[0,5],[2,6]],'units_sold':0}),('boundary',{'layers':[],'units_sold':0}),('invalid',{'layers':[[2,3]],'units_sold':3})]))
}
ORDER=['analytics','inventory','optimization','planning','scheduling','logistics','finance']

def oracle(task,x):
    try:return {'ok':True,'value':task.reference(copy.deepcopy(x))}
    except (ValueError,TypeError,KeyError,ZeroDivisionError):return {'ok':False,'error':'ValueError'}

def test_cases(task_name,seed=0):
    """Deterministic test fixtures and small parametric variations, shared across arms."""
    r=random.Random(seed); task=TASK_REGISTRY[task_name]; out=[]
    for kind,x in task.cases:
        x=copy.deepcopy(x)
        # Small numeric changes, preserving intended classification, for ordinary cases.
        # The same seed is used in every arm; category labels remain stable.
        if kind=='ordinary' and task_name=='analytics': x['sales']=[v+r.choice([0,1,2]) for v in x['sales']]
        elif kind=='ordinary' and task_name=='scheduling':
            for j in x['jobs']:j['processing_time']+=r.choice([0,1])
        out.append({'category':kind,'input':x,'expected':oracle(task,x)})
    return out
