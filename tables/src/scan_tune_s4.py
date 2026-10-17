import json, os, subprocess, sys, re
roots = ['<RUN_ROOT_B>/P04_mestune/runs_mestune', '<RUN_ROOT_B>/P04_xen0/runs_xen0', '<DATA_ROOT>/MANU/runs']
out = []
for root in roots:
    for dp, dn, fn in os.walk(root, followlinks=True):
        if 'result.json' in fn:
            dn[:] = []
            try: j = json.load(open(os.path.join(dp, 'result.json')))
            except Exception as e: continue
            if j.get('package') != 'xres0': continue
            tb = j.get('trainer_best_valid') or {}
            bt = j.get('best_test') or {}
            bv = j.get('best_valid_primed') or {}
            out.append(dict(root=root, run=dp, cfg=j['config'], ds=str(j['dataset']).split('@')[0], seed=j['seed'], status=j.get('status'),
                            mode=j.get('mode'), te=j.get('test_eval'), tb_r20=tb.get('recall@20'), tb_keys=sorted(tb.keys())[:6],
                            bv_r20=bv.get('recall@20'), bt_r20=bt.get('recall@20'), bt_n20=bt.get('ndcg@20'), best_epoch=j.get('best_epoch'), bvs=j.get('best_valid_score')))
json.dump(out, open('<RUN_ROOT>/study/tune_scan.json', 'w'))
print(len(out))
