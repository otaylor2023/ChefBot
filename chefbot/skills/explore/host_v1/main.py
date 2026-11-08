"""Drive the kitchen's sweep and map what is on the counters (scripts/real_kitchen.py RealKitchen.explore)."""


def run(kitchen, route="south"):
    r = kitchen.explore(route)
    # what the agent needs: what is where (the full record is the run's explore.json)
    objs = kitchen.objects.objects() if kitchen.objects is not None else []
    return {"ok": r["ok"], "route": route, "seconds": r["seconds"],
            "objects": [f"{o['name']} on {o.get('fixture') or o['counter']}" for o in objs
                        if o["status"] != "rejected" and o["xyz"] is not None],
            "stops": [{k: s.get(k) for k in ("stop", "ok", "reason")} for s in r["stops"]]}
