"""Deterministic environment profiles and resolved Scenario instances."""
from pathlib import Path
from typing import Any, Mapping
import hashlib, json
import numpy as np

def _set(dst, path, value):
    cur=dst; parts=path.split('.')
    for part in parts[:-1]: cur=cur.setdefault(part,{})
    cur[parts[-1]]=value

def _get(src, path):
    cur = src
    for part in path.split('.'):
        if not isinstance(cur, Mapping) or part not in cur:
            raise ValueError(f"scenario constraint references missing path: {path}")
        cur = cur[part]
    return cur

def _profile_rule(profile, path):
    """Find a sampling rule by its resolved (group-relative) path."""
    for group in ("fixed", "geometry", "sampled", "physics", "appearance", "runtime"):
        cur = profile.get(group, {}) or {}
        if path in cur:
            return cur[path]
        try:
            for part in path.split('.'):
                cur = cur[part]
        except (KeyError, TypeError):
            continue
        return cur
    return None

def _flatten(value, prefix=''):
    for key,item in value.items():
        path=f'{prefix}.{key}' if prefix else str(key)
        if isinstance(item,Mapping) and 'distribution' not in item: yield from _flatten(item,path)
        else: yield path,item

def _sample(raw,rng):
    if not isinstance(raw,Mapping) or raw.get('distribution','fixed')=='fixed': return raw.get('value') if isinstance(raw,Mapping) and 'value' in raw else raw
    if raw['distribution']=='uniform': return np.asarray(rng.uniform(raw['low'],raw['high'])).item()
    if raw['distribution']=='exponential':
        low=float(raw.get('low',0.0)); scale=float(raw.get('scale',1.0)); high=raw.get('high')
        if not np.isfinite(low) or not np.isfinite(scale) or scale<=0: raise ValueError('exponential distribution requires finite low and scale > 0')
        if high is None: return float(low+rng.exponential(scale))
        high=float(high)
        if not np.isfinite(high) or high<=low: raise ValueError('bounded exponential distribution requires finite high > low')
        # Inverse CDF of an exponential conditioned on [low, high]. Unlike
        # clipping an unbounded sample, this creates no probability spike at
        # the upper limit and remains exactly reproducible from the seed.
        span=high-low; normalizer=-np.expm1(-span/scale); u=float(rng.random())
        return float(low-scale*np.log1p(-u*normalizer))
    if raw['distribution']=='choice':
        values = list(raw['values'])
        if not values: raise ValueError('choice distribution requires at least one value')
        return values[int(rng.integers(len(values)))]
    raise ValueError(f"unsupported distribution: {raw['distribution']}")

def _apply_constraints(profile, parameters, seed):
    """Apply declarative cross-field constraints after independent sampling."""
    constraints = profile.get("constraints", ()) or ()
    rng = np.random.default_rng(int(seed) + 4000012)
    for rule in constraints:
        if not isinstance(rule, Mapping) or rule.get("type") != "not_equal":
            raise ValueError(f"unsupported scenario constraint: {rule!r}")
        paths = list(rule.get("paths", ()))
        if len(paths) != 2:
            raise ValueError("not_equal constraint requires exactly two paths")
        left_path, right_path = map(str, paths)
        left = _get(parameters, left_path)
        if _get(parameters, right_path) != left:
            continue
        raw = _profile_rule(profile, right_path)
        values = list(raw.get("values", ())) if (
            isinstance(raw, Mapping) and raw.get("distribution") == "choice"
        ) else []
        candidates = [value for value in values if value != left]
        if not candidates:
            raise ValueError(
                f"cannot satisfy not_equal constraint for {left_path} and {right_path}"
            )
        _set(parameters, right_path, candidates[int(rng.integers(len(candidates)))])

class Scenario:
    def __init__(self,schema_version,task_id,scenario_id,seed,parameters,assets=None,metadata=None): self.schema_version,self.task_id,self.scenario_id,self.seed=schema_version,task_id,scenario_id,int(seed); self.parameters,self.assets,self.metadata=parameters,assets or {},metadata or {}
    def to_dict(self, digest=True):
        out={'schema_version':self.schema_version,'task_id':self.task_id,'scenario_id':self.scenario_id,'seed':self.seed,'parameters':self.parameters,'assets':self.assets,'metadata':self.metadata}
        if digest: out['digest']=self.digest
        return out
    @property
    def digest(self): return hashlib.sha256(json.dumps(self.to_dict(False),sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]
    def save(self,path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    @classmethod
    def load(cls,path):
        r=json.loads(Path(path).read_text(encoding='utf-8')); s=cls(r['schema_version'],r['task_id'],r['scenario_id'],r['seed'],r['parameters'],r.get('assets'),r.get('metadata'))
        if r.get('digest') and r['digest']!=s.digest: raise ValueError('scenario digest mismatch')
        return s

def sample_scenario(profile:Mapping[str,Any], *, seed=0, scenario_id=None):
    out={}
    for path,value in _flatten(profile.get('fixed',{})): _set(out,path,value)
    for i,(group,values) in enumerate((('geometry',profile.get('geometry',profile.get('sampled',{}))),('physics',profile.get('physics',{})),('appearance',profile.get('appearance',{})),('runtime',profile.get('runtime',{})))):
        rng=np.random.default_rng(int(seed)+i*1000003)
        for path,raw in _flatten(values): _set(out,path,_sample(raw,rng))
    _apply_constraints(profile, out, seed)
    return Scenario(str(profile.get('schema_version','1.0')),str(profile['task_id']),scenario_id or f"{profile['task_id']}_{int(seed):06d}",seed,out,profile.get('assets',{}),{'profile':profile.get('profile_id','default')})
