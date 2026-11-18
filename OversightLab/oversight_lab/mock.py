"""Deterministic no-cost mock for testing the entire orchestration without API keys.
It is NOT an LLM, and mock results are not scientific experimental results.
"""
MOCK_SOLUTIONS={
'analytics':'''def solve(x):
    a=x['sales'];w=x['window']
    if not a or not isinstance(w,int) or w<=0 or any(v<0 for v in a): raise ValueError('invalid')
    a=a[-w:]
    return round(sum(a)/len(a),4)
''',
'inventory':'''def solve(x):
    d=x['daily_mean'];s=x['daily_std'];L=x['lead_time'];z=x['z']
    if min(d,s,L,z)<0:raise ValueError('invalid')
    return round(d*L+z*s*(L**0.5),4)
''',
'optimization':'''def solve(x):
    cap=x['capacity']; ps=x['products']
    if cap<0 or any(p['hours']<=0 or p['margin']<0 or p['max_units']<0 for p in ps):raise ValueError('invalid')
    out={str(p['id']):0.0 for p in ps}
    for p in sorted(ps,key=lambda p:(-p['margin']/p['hours'],str(p['id']))):
        qty=min(float(p['max_units']),cap/p['hours'])
        out[str(p['id'])]=round(qty,4)
        cap=max(0,cap-qty*p['hours'])
    return out
''',
'planning':'''def solve(x):
    f=x['forecast'];h=x['on_hand'];r=x['raw_material'];a=x['allocated_hours'];t=x['hours_per_unit']
    if min(f,h,r,a)<0 or t<=0:raise ValueError('invalid')
    need=max(0,-(-(f-h)//1))
    return int(min(need,r,a//t))
''',
'scheduling':'''def solve(x):
    jobs=x['jobs']
    if any(j['processing_time']<0 or j['weight']<=0 for j in jobs):raise ValueError('invalid')
    return [j['id'] for j in sorted(jobs,key=lambda j:(j['processing_time']/j['weight'],str(j['id'])))]
''',
'logistics':'''def solve(x):
    room=list(x['vehicle_capacities']);orders=x['orders']
    if any(c<0 for c in room) or any(o['units']<0 for o in orders):raise ValueError('invalid')
    out={str(o['id']):-1 for o in orders}
    for o in sorted(orders,key=lambda o:(-o['units'],str(o['id']))):
        for i in range(len(room)):
            if room[i]>=o['units']:
                room[i]-=o['units'];out[str(o['id'])]=i;break
    return out
''',
'finance':'''def solve(x):
    layers=x['layers'];sold=x['units_sold']
    if sold<0 or sold>sum(z[0] for z in layers) or any(q<0 or p<0 for q,p in layers):raise ValueError('invalid')
    rest=[];cost=0
    for q,p in layers:
        take=min(sold,q);cost+=take*p;sold-=take
        if q>take:rest=rest+[[q-take,p]]
    return {'cogs':round(cost,4),'remaining':rest}
'''
}
