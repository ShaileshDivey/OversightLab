"""One deterministic miniature supply-chain application of the seven code components.
Scenario performance is supplementary; component evaluation is the primary estimand.
"""
import random
from .tasks import TASK_REGISTRY
from .evaluator import execute

def _call(family,source,x,reference=False):
    if reference:return TASK_REGISTRY[family].reference(x)
    out,error=execute(source[family],[x])
    if error or not out[0].get('ok'):raise ValueError(f'{family}: {error or out[0].get("error")}')
    return out[0]['value']

def evaluate_system(codes,seed=42,reference=False):
    r=random.Random(seed)
    products=[
     {'id':'A','price':100,'cost':55,'hours':2,'on_hand':3,'stock':[[3,55]],'margin':45},
     {'id':'B','price':80,'cost':45,'hours':1,'on_hand':4,'stock':[[4,45]],'margin':35},
     {'id':'C','price':120,'cost':70,'hours':3,'on_hand':2,'stock':[[2,70]],'margin':50},
    ]
    try:
        for p in products:
            hist=[max(0,int(r.gauss(7,1.5))) for _ in range(5)]
            p['forecast']=float(_call('analytics',codes,{'sales':hist,'window':3},reference))
            p['reorder_point']=float(_call('inventory',codes,{
                'daily_mean':p['forecast'],'daily_std':2,'lead_time':2,'z':1},reference))
        allocation=_call('optimization',codes,{'capacity':28,'products':[
            {'id':p['id'],'margin':p['margin'],'hours':p['hours'],'max_units':max(0,p['forecast']-p['on_hand'])}
            for p in products]},reference)
        for p in products:
            p['quantity']=int(_call('planning',codes,{
                'forecast':p['forecast'],'on_hand':p['on_hand'],'raw_material':15,
                'allocated_hours':float(allocation[p['id']])*p['hours'],'hours_per_unit':p['hours']},reference))
        jobs=[{'id':p['id'],'processing_time':p['hours']*p['quantity'],'weight':p['margin']} for p in products if p['quantity']>0]
        sequence=_call('scheduling',codes,{'jobs':jobs},reference)
        orders=[{'id':p['id'],'units':p['quantity']+p['on_hand']} for p in products]
        vehicle_assignments=_call('logistics',codes,{'vehicle_capacities':[12,12,12],'orders':orders},reference)
        sales=0; cogs=0;filled=0
        for p in products:
            available=p['quantity']+p['on_hand']
            shipped=min(available,round(p['forecast'])) if vehicle_assignments.get(p['id'],-1)>=0 else 0
            val=_call('finance',codes,{'layers':p['stock']+([[p['quantity'],p['cost']]] if p['quantity'] else []),'units_sold':shipped},reference)
            cogs+=val['cogs'];sales+=shipped*p['price'];filled+=shipped
        production_cost=sum(p['quantity']*p['cost'] for p in products)
        # Costs of delivered goods + simple production/transport overhead; not a calibrated economic model.
        # COGS reported separately to avoid double counting new output in the cash-contribution metric.
        return {'feasible':True,'cash_contribution':round(sales-production_cost-15*len(set(v for v in vehicle_assignments.values() if v>=0)),2),
                'sales_revenue':round(sales,2),'cogs':round(cogs,2),'fulfilled_units':filled,
                'produced_units':sum(p['quantity'] for p in products),'sequence':sequence}
    except (Exception) as ex:
        return {'feasible':False,'cash_contribution':None,'sales_revenue':None,'cogs':None,
            'fulfilled_units':0,'produced_units':0,'error':str(ex)[:160]}
