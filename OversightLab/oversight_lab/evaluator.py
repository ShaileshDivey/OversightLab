"""Restricted single-function runner. IMPORTANT: not a hardened security boundary.
Untrusted submissions should be evaluated in external hardened isolation, not Colab.
"""
import ast, json, os, subprocess, sys, textwrap
from .tasks import oracle, TASK_REGISTRY

ALLOWED_CALLS={'abs','all','any','bool','dict','enumerate','float','int','isinstance','len','list','max','min','range','reversed','round','sorted','str','sum','tuple','zip','ValueError','TypeError','OverflowError','ZeroDivisionError','Exception','map','filter','set','divmod','pow'}
ALLOWED_NODES=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.Assign,ast.AnnAssign,ast.AugAssign,
 ast.Name,ast.Load,ast.Store,ast.Constant,ast.BinOp,ast.UnaryOp,ast.BoolOp,ast.Compare,ast.If,ast.IfExp,
 ast.For,ast.While,ast.Break,ast.Continue,ast.Call,ast.keyword,ast.Subscript,ast.Slice,ast.List,ast.Tuple,
 ast.Dict,ast.Set,ast.ListComp,ast.DictComp,ast.SetComp,ast.GeneratorExp,ast.comprehension,ast.Lambda,
 ast.Raise,ast.Try,ast.ExceptHandler,ast.Pass,ast.Expr,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod,ast.Pow,ast.USub,
 ast.UAdd,ast.And,ast.Or,ast.Not,ast.Eq,ast.NotEq,ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.Is,ast.IsNot,
 ast.In,ast.NotIn,ast.JoinedStr,ast.FormattedValue,ast.Attribute)

def validate(code):
    if len(code)>14000:raise ValueError('code too long')
    tree=ast.parse(code)
    if len(list(ast.walk(tree)))>1500:raise ValueError('AST too large')
    # Permit named top-level helper functions, but no executable top-level statements.
    if not tree.body or any(not isinstance(n, ast.FunctionDef) for n in tree.body):
        raise ValueError('Top-level code must contain only function definitions')
    functions = [n for n in tree.body if n.name == 'solve']
    if len(functions) != 1:
        raise ValueError('Exactly one def solve(x) is required')
    if len(functions[0].args.args)!=1 or functions[0].args.args[0].arg!='x':
        raise ValueError('Signature must be def solve(x)')
    locally_defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for n in ast.walk(tree):
        if not isinstance(n,ALLOWED_NODES):raise ValueError(f'Disallowed Python syntax: {type(n).__name__}')
        if isinstance(n,ast.Name) and n.id.startswith('__'):raise ValueError('Dunder names prohibited')
        if isinstance(n,ast.Attribute) and n.attr not in {'append','extend','insert','pop','remove','get','setdefault','update','items','keys','values','copy','sort','reverse','count','index','add','discard'}:
            raise ValueError('Only basic list/dict methods permitted')
        if isinstance(n,ast.Call) and not ((isinstance(n.func,ast.Name) and n.func.id in ALLOWED_CALLS | locally_defined) or
                 (isinstance(n.func,ast.Attribute) and n.func.attr in {'append','extend','insert','pop','remove','get','setdefault','update','items','keys','values','copy','sort','reverse','count','index','add','discard'})):
            raise ValueError('Only allowlisted builtin calls and basic methods permitted')
        # A named nested helper is ordinary Python; the worker remains restricted.
        if isinstance(n,ast.ExceptHandler) and n.type is not None:
            exceptions = n.type.elts if isinstance(n.type,ast.Tuple) else [n.type]
            if not all(isinstance(e,ast.Name) and e.id in {'ValueError','TypeError','OverflowError','ZeroDivisionError','Exception'} for e in exceptions):
                raise ValueError('Exception handler type not permitted')
    return code

# Worker runs in fresh Python process with no site imports and stripped builtin namespace.
WORKER=r'''
import json, sys, ast
try:
 import resource
 resource.setrlimit(resource.RLIMIT_CPU,(2,2))
 resource.setrlimit(resource.RLIMIT_AS,(256*1024*1024,256*1024*1024))
 resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
except ImportError: pass
req=json.loads(sys.stdin.read())
permit=('abs','all','any','bool','dict','enumerate','float','int','isinstance','len','list','max','min','range','reversed','round','sorted','str','sum','tuple','zip','ValueError','TypeError','OverflowError','ZeroDivisionError','Exception','map','filter','set','divmod','pow')
b=__builtins__ if isinstance(__builtins__,dict) else vars(__builtins__)
restricted={k:b[k] for k in permit}
ns={'__builtins__':restricted}
exec(compile(req['code'],'<agent>','exec'),ns,ns)
out=[]
for case in req['cases']:
 try:
  v=ns['solve'](case)
  json.dumps(v,allow_nan=False)
  out.append({'ok':True,'value':v})
 except ValueError:out.append({'ok':False,'error':'ValueError'})
 except BaseException as e:out.append({'ok':False,'error':type(e).__name__})
print(json.dumps(out,allow_nan=False))
'''

def execute(code, inputs, timeout=5):
    try:validate(code)
    except Exception as e:return [{'ok':False,'error':'SyntaxOrPolicyError'} for _ in inputs],str(e)
    payload=json.dumps({'code':code,'cases':inputs})
    try:
        proc=subprocess.run([sys.executable,'-I','-S','-c',WORKER],input=payload,text=True,capture_output=True,
          timeout=timeout,env={'PYTHONIOENCODING':'utf-8'},cwd='/tmp')
        if proc.returncode!=0:raise RuntimeError(proc.stderr[-300:])
        value=json.loads(proc.stdout)
        if len(value)!=len(inputs):raise ValueError('test count mismatch')
        return value,None
    except Exception as e:return [{'ok':False,'error':'ExecutionError'} for _ in inputs],str(e)

def score(task_name,code,cases):
    outputs,error=execute(code,[c['input'] for c in cases]);correct=[]
    for o,c in zip(outputs,cases):
        exp=c['expected'];good=(o==exp)
        # For optimization and other floating outcomes, tiny numerical discrepancies tolerated.
        if o.get('ok') and exp.get('ok') and not good:
            good=_nearly(o.get('value'),exp.get('value'))
        correct.append(bool(good))
    execution_ok = [bool(o.get('ok')) or (o.get('error') == 'ValueError' and
                    c['expected'].get('error') == 'ValueError')
                    for o,c in zip(outputs,cases)]
    executable = error is None
    successes = sum(execution_ok)
    return {'execution_valid':executable,
      'execution_success_rate':round(successes/len(cases),4) if cases else 0.0,
      'quality_given_execution':round(sum(correct)/successes,4) if successes else None,
      'failure_type': ('policy_or_parse' if error and 'Syntax' in str(error) else
                       'worker_failure' if error else
                       'runtime_or_logic' if successes < len(cases) else 'none'),
      'score':round(sum(correct)/len(correct),4) if cases else 0.0,
      'passed':sum(correct),'total':len(correct),'by_category':{cat:sum(int(b) for b,c in zip(correct,cases) if c['category']==cat) for cat in sorted({c['category'] for c in cases})},
      'correct':correct,'error':error}

def _nearly(x,y):
    if isinstance(x,(int,float)) and isinstance(y,(int,float)):return abs(x-y)<0.00011
    if isinstance(x,dict) and isinstance(y,dict) and x.keys()==y.keys():return all(_nearly(x[k],y[k]) for k in x)
    if isinstance(x,list) and isinstance(y,list) and len(x)==len(y):return all(_nearly(a,b) for a,b in zip(x,y))
    return x==y
