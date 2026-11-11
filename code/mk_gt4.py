"""Derives gen_tables4.py (final naming: LACUNA = terminal-head reader `lach`; outcome hypotheses = analysis row) from gen_tables3.py."""
s = open('gen_tables3.py', encoding='utf8').read()
HYP = "'LACUNA, outcome hypotheses'"
R = [
    # main / hop / beam / cross-dataset / answerability
    ("('-', ''), ('lact', LAC + ' (ours)')]", "('-', ''), ('lach', LAC + ' (ours)')]"),
    ("('beam', 'Beam Retrieval$^\\\\dagger$'), ('lach', 'LACUNA, terminal head'), ('lact', LAC)]", "('beam', 'Beam Retrieval$^\\\\dagger$'), ('lact', " + HYP + "), ('lach', LAC)]"),
    ("('lacg', 'LACUNA, terminal head'), ('lact', LAC), ('-', ''), ('beam', 'Beam Retrieval (official)')]", "('lact', " + HYP + "), ('lach', LAC), ('-', ''), ('beam', 'Beam Retrieval (official)')]"),
    ("('jbce', 'Joint reader (same recipe)'), ('lacg', 'LACUNA, terminal head'), ('lact', LAC)]", "('jbce', 'Joint reader (same recipe)'), ('lacg', LAC)]"),
    ("('anscls', 'Answerability classifier (same backbone and pools)'), ('lach', 'LACUNA, terminal head'), ('lact', LAC)]", "('anscls', 'Answerability classifier (same backbone and pools)'), ('lact', " + HYP + "), ('lach', LAC)]"),
    ("for n, lab in [('lach', 'LACUNA, terminal head'), ('lact', LAC)]:", "for n, lab in [('lact', " + HYP + "), ('lach', LAC)]:"),
    # self-termination
    ("          ('lach', 'LACUNA, terminal head (no tuning)', 'lach_dec'),\n          ('lact', LAC + ' (no tuning)', 'lact_dec'),\n          ('lact_cal', LAC + ', $\\\\top$ offset tuned on selection split', 'lact_cal_dec'),",
     "          ('lact', 'LACUNA, outcome hypotheses (no tuning)', 'lact_dec'),\n          ('lach', LAC + ' (no tuning)', 'lach_dec'),\n          ('lach_cal', LAC + ', $\\\\top$ offset tuned on selection split', 'lach_cal_dec'),"),
    # first error
    ("fe_rows = [('lach', 'terminal head'), ('lact', LAC + ' (outcome hypotheses)'), ('abl_t_noroll', 'w/o learner-visited states'), ('abl_t_ce1', 'single-positive CE (loss only)')]",
     "fe_rows = [('lach', LAC + ' (no tuning)'), ('lach_cal', LAC + ' ($\\\\top$ offset tuned)'), ('lact', 'outcome hypotheses (no tuning)'), ('abl_noroll', 'w/o learner-visited states'), ('abl_ce1', 'single-positive CE (loss only)')]"),
    # loops
    ("('lach', 'adaptive', 'LACUNA, terminal head'), ('lac', 'fixed', LAC), ('lac', 'adaptive', LAC + ', verdict-driven')]",
     "('lac', 'adaptive', 'LACUNA, outcome hypotheses'), ('lach', 'fixed', LAC), ('lach', 'adaptive', LAC + ', verdict-driven')]"),
    ("('jbce3', 'Joint reader +1', 'jbce3'), ('lact', LAC, 'lact')]", "('jbce3', 'Joint reader +1', 'jbce3'), ('lacg', LAC, 'lacg')]"),
    ("[('lac_fixed', 'static_fixed', 'LACUNA vs joint reader +1, both fixed'), ('lac_fixed', 'chain_fixed', 'LACUNA vs Chain-CE, both fixed'),\n                  ('lac_fixed', 'dense_fixed', 'LACUNA vs dense PRF, both fixed'),\n                  ('lac_adaptive', 'static_adaptive', 'LACUNA vs joint reader +1 + classifier, adaptive'),\n                  ('lac_adaptive', 'lach_adaptive', 'outcome hypotheses vs terminal head, adaptive')]",
     "[('lach_fixed', 'static_fixed', 'LACUNA vs joint reader +1, both fixed'), ('lach_fixed', 'chain_fixed', 'LACUNA vs Chain-CE, both fixed'),\n                  ('lach_fixed', 'dense_fixed', 'LACUNA vs dense PRF, both fixed'),\n                  ('lach_adaptive', 'static_adaptive', 'LACUNA vs joint reader +1 + classifier, adaptive'),\n                  ('lach_adaptive', 'static_fixed', 'LACUNA adaptive vs joint reader +1 fixed'),\n                  ('lach_adaptive', 'lac_adaptive', 'terminal head vs outcome hypotheses, adaptive')]"),
    # QA
    ("('lacg', 'LACUNA, terminal head'), ('lact', LAC), ('-', ''), ('oracle', 'gold supporting passages$^\\\\ast$')]", "('lacg', LAC), ('-', ''), ('oracle', 'gold supporting passages$^\\\\ast$')]"),
    ("('beam', 'Beam Retrieval (uses $|G|$)'), ('lact', LAC), ('-', ''), ('oracle', 'gold passages$^\\\\ast$')]", "('beam', 'Beam Retrieval (uses $|G|$)'), ('lacg', LAC), ('-', ''), ('oracle', 'gold passages$^\\\\ast$')]"),
    ("('lac_fixed', LAC + ' (fixed)'), ('lac_adaptive', LAC + ', verdict-driven')]", "('lach_fixed', LAC + ' (fixed)'), ('lach_adaptive', LAC + ', verdict-driven')]"),
    ("('q3rr', 'Qwen3-Reranker-4B'), ('lacg', 'LACUNA, terminal head'), ('lact', LAC), ('-', '')", "('q3rr', 'Qwen3-Reranker-4B'), ('lacg', LAC), ('-', '')"),
    # ablations (terminal-head family; the outcome-hypothesis row replaces the head)
    ("          ('lach', 'separate terminal head on $\\\\langle s\\\\rangle$'), ('abl_t_mark', 'state marked in place (no evidence-as-query)'),",
     "          ('lact', 'outcomes as verbalized hypotheses (no terminal head)'), ('abl_mark', 'state marked in place (no evidence-as-query)'),"),
    ("('abl_t_ce1', 'single-positive CE (loss only)'), ('abl_t_prefix', 'chain-prefix states only'),", "('abl_ce1', 'single-positive CE (loss only)'), ('abl_prefix', 'chain-prefix states only'),"),
    ("          ('abl_t_noroll', 'no learner-visited states'), ('abl_t_noworld', 'no world-deleted pools'), ('lact', LAC + ' (full)')]",
     "          ('abl_noroll', 'no learner-visited states'), ('abl_noworld', 'no world-deleted pools'), ('lach', LAC + ' (full)')]"),
    ("def halt_cer(n):\n    return (H.get(n) or {}).get('CER')", "def halt_cer(n):\n    return (H.get(n + '_cal') or {}).get('CER')"),
    ("(halt_cer(n) if n not in ('jbce3', 'abl_rbce') else (H.get('jbce3tuned') or {}).get('CER') if n == 'jbce3' else 0.0)",
     "(halt_cer(n) if n not in ('jbce3', 'abl_rbce', 'lacp') else (H.get('jbce3tuned') or {}).get('CER') if n == 'jbce3' else (H.get('lacp') or {}).get('CER') if n == 'lacp' else 0.0)"),
    ("Stop: CER of the self-terminated set (held-out; static reader: tuned threshold; BCE variant has no stop decision, shown as 0.0).",
     "Stop: CER of the self-terminated set on held-out questions, every variant with its own $\\\\top$ offset tuned on the selection split (static reader: tuned threshold; factorised: its own size rule; the BCE variant has no stop decision, shown as 0.0)."),
    # seeds
    ("sd = [(LAC + ' (stage 1: seed 42)', ['lact', 'lact_s43', 'lact_s44']),", "sd = [(LAC + ' (stage 1: seed 42)', ['lacg', 'mq_lacr_s43', 'mq_lacr_s44']),"),
    ("    hc = [halt_cer(n) for n in names] if names[0] == 'lact' else [None]\n    cells.append('MISSING' if None in hc else (f'{np.mean(hc):.1f}$\\\\pm${np.std(hc):.1f}' if names[0] == 'lact' else fmt((H.get('jbce3tuned') or {}).get('CER'))))",
     "    hk = ['lach_cal', 'lach_s43_cal', 'lach_s44_cal'] if names[0] == 'lacg' else ['jbce3tuned', 'jbce3_s43tuned', 'jbce3_s44tuned']\n    hc = [(H.get(k) or {}).get('CER') for k in hk]\n    cells.append('MISSING' if None in hc else f'{np.mean(hc):.1f}$\\\\pm${np.std(hc):.1f}')"),
    ("for lab, n in [(LAC + ' (stage 1: seed 43)', 'lact_f43'), ('Joint reader +1 (stage 1: seed 43)', 'jbce3_f43')]:", "for lab, n in [(LAC + ' (stage 1: seed 43)', 'lacr_f43'), ('Joint reader +1 (stage 1: seed 43)', 'jbce3_f43')]:"),
    ("                 (fmt(halt_cer(n)) if n.startswith('lact') else 'n/a') + ' \\\\\\\\')", "                 fmt((H.get('lacr_f43_cal' if n.startswith('lacr') else 'jbce3_f43tuned') or {}).get('CER')) + ' \\\\\\\\')"),
    (".replace('n/a', fmt((H.get('jbce3tuned') or {}).get('CER')))", ""),
    ("Stop: held-out CER of the self-terminated set (static: tuned threshold).", "Stop: held-out CER of the self-terminated set with the $\\\\top$ offset (LACUNA) or threshold (static) tuned on the selection split."),
    # significance
    ("[('jbce3', 'Joint reader +1'), ('jbce4', 'Joint reader +2 (time-matched)'), ('chain', 'Chain-CE'), ('lach', 'LACUNA, terminal head')]:\n    s = J(f'{M}/signif_{D}_lact_vs_{b}.json') or {}",
     "[('jbce3', 'Joint reader +1'), ('jbce4', 'Joint reader +2 (time-matched)'), ('chain', 'Chain-CE'), ('abl_ce1', 'single-positive CE (loss only)')]:\n    s = J(f'{M}/signif_{D}_lach_vs_{b}.json') or {}"),
    # latency
    ("    lg = f'logs/mq_lact_T{t}.log'", "    lg = f'logs/mq_lacr_T{t}.log'"),
    ("    n = f'lact_T{t}'", "    n = f'lach_T{t}'"),
    ("for line in open('logs/sched/tr_lact.log') if os.path.exists('logs/sched/tr_lact.log') else []:", "for line in open('logs/mq_lacr_infer.log') if os.path.exists('logs/mq_lacr_infer.log') else []:"),
    ("lat.append('6 (full) & ' + ' & '.join(fmt(cer(D, 'lact', k))", "lat.append('6 (full) & ' + ' & '.join(fmt(cer(D, 'lach', k))"),
    # robustness
    ("for n, lab in [('dense', 'Dense order'), ('chain', 'Chain-CE'), ('jbce3', 'Joint reader +1'), ('lact', LAC)]:", "for n, lab in [('dense', 'Dense order'), ('chain', 'Chain-CE'), ('jbce3', 'Joint reader +1'), ('lach', LAC)]:"),
    ("for n, lab in [('jbce3', 'Joint reader +1'), ('lact', LAC)]:", "for n, lab in [('jbce3', 'Joint reader +1'), ('lach', LAC)]:"),
    ("rows_out([('jbce3', 'Joint reader +1'), ('lact', LAC)], lr)", "rows_out([('jbce3', 'Joint reader +1'), ('lach', LAC)], lr)"),
    ("for lab, names in [('lact', ['lact', 'lact_s43', 'lact_s44']), ('jbce3', ['jbce3', 'jbce3_s43', 'jbce3_s44'])]:", "for lab, names in [('lac', ['lacg', 'mq_lacr_s43', 'mq_lacr_s44']), ('jbce3', ['jbce3', 'jbce3_s43', 'jbce3_s44'])]:"),
]
miss = []
for a, b in R:
    if a not in s:
        miss.append(a[:90])
    s = s.replace(a, b)
s = s.replace('% generated by code/gen_tables3.py', '% generated by code/gen_tables4.py')
open('gen_tables4.py', 'w', encoding='utf8').write(s)
print('not found:', miss)
