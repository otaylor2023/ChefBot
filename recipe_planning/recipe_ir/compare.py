"""Compare two Recipes of the same source (prediction vs gold, or run i vs run j).

IDs differ between extractions, so elements are aligned by the source text they cite (evidence span overlap).
Inferred elements without a span fall back to matching by verb (steps) or name (ingredients).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from .schema import Recipe


def _overlap(a, b) -> int:
    if a is None or b is None:
        return 0
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _align(pred: list, gold: list, fallback_key) -> dict[str, str]:
    """Greedy one-to-one alignment: largest span overlap first, then fallback key equality."""
    pairs = sorted(((_overlap(p.evidence.span, g.evidence.span), p.id, g.id)
                    for p in pred for g in gold), reverse=True)
    m, used_p, used_g = {}, set(), set()
    for ov, pid, gid in pairs:
        if ov <= 0:
            break
        if pid not in used_p and gid not in used_g:
            m[pid] = gid
            used_p.add(pid)
            used_g.add(gid)
    for p in pred:
        if p.id in used_p:
            continue
        for g in gold:
            if g.id not in used_g and fallback_key(p) == fallback_key(g):
                m[p.id] = g.id
                used_p.add(p.id)
                used_g.add(g.id)
                break
    return m


def _prf(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else (1.0 if not n_gold else 0.0)
    r = tp / n_gold if n_gold else 1.0
    return round(p, 3), round(r, 3), round(2 * p * r / (p + r), 3) if p + r else 0.0


@dataclass
class Comparison:
    step_p: float
    step_r: float
    step_f1: float
    strict_step_f1: float   # aligned AND same verb: two actions citing the same sentence don't both count
    verb_acc: float
    edge_p: float
    edge_r: float
    edge_f1: float
    ingredient_f1: float
    needs_cooking_acc: float
    missed_steps: list[str]
    extra_steps: list[str]
    verb_errors: list[str]

    def to_dict(self):
        return asdict(self)


def compare(pred: Recipe, gold: Recipe, pred_edges: set, gold_edges: set) -> Comparison:
    sm = _align(pred.steps, gold.steps, lambda s: (s.verb, s.evidence.prov))
    gsteps = {s.id: s for s in gold.steps}
    tp = len(sm)
    sp, sr, sf = _prf(tp, len(pred.steps), len(gold.steps))
    verb_ok = [p for p in pred.steps if p.id in sm and p.verb == gsteps[sm[p.id]].verb]
    verb_errors = [f"{p.id}:{p.verb} vs {sm[p.id]}:{gsteps[sm[p.id]].verb}"
                   for p in pred.steps if p.id in sm and p.verb != gsteps[sm[p.id]].verb]
    *_, strict_f1 = _prf(len(verb_ok), len(pred.steps), len(gold.steps))
    mapped = {(sm[a], sm[b]) for a, b in pred_edges if a in sm and b in sm}
    etp = len(mapped & set(gold_edges))
    ep, er, ef = _prf(etp, len(pred_edges), len(gold_edges))

    p_ing = pred.ingredients
    g_ing = gold.ingredients
    norm = lambda i: str(i.name).lower().rstrip("s")
    im = _align(p_ing, g_ing, norm)
    *_, inf = _prf(len(im), len(p_ing), len(g_ing))
    gi = {i.id: i for i in g_ing}
    nc = [p.needs_cooking == gi[im[p.id]].needs_cooking for p in p_ing if p.id in im]
    return Comparison(
        sp, sr, sf, strict_f1, round(len(verb_ok) / tp, 3) if tp else 0.0, ep, er, ef, inf,
        round(sum(nc) / len(nc), 3) if nc else 0.0,
        missed_steps=[f"{g.id}:{g.verb}" for g in gold.steps if g.id not in set(sm.values())],
        extra_steps=[f"{p.id}:{p.verb}" for p in pred.steps if p.id not in sm],
        verb_errors=verb_errors,
    )
