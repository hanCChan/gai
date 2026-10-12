#!/usr/bin/env python3
"""Prepare P3 history from user paper and already cached headers, no HDF5 access."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    paper=Path('/home/keyan/code-y/oob-lunwen1/OOB_IOTJ.pdf');text=Path('/tmp/keyan_oob_paper.txt').read_text()
    base=json.loads((ROOT/'configs/shen_evidence/p3_protocol.candidate.json').read_text())
    expected=['b200_2','b200_mini_2','b210_2','n210_2','n210_3','pluto_2'];assert base['final_locked_receivers']==expected
    # PDF layout has underscores converted to spaces. Pin this textual evidence explicitly.
    assert 'Six files form Rblind' in text and 'evaluated the six blind receivers' in text
    headers=json.loads(Path('/home/keyan/code-y/inventory/shen_label_audit_20261011.json').read_text())
    development=set(base['train_receivers']+base['development_receivers']);table=[]
    for role,key in [('train','train_receivers'),('development','development_receivers'),('locked','final_locked_receivers')]:
        for rx in base[key]:
            family=rx.rsplit('_',1)[0]
            cached=[h for h in headers if re.search(r'/'+re.escape(rx)+r'_(?:train|test)\.h5$',h['path'])] if rx in development else []
            if rx in development:
                assert len(cached)==2
                assert all(h['labels']==list(range(31,41)) and h['shape'][1]==16384 for h in cached)
            table.append({'receiver':rx,'P3_role':role,'hardware_family_inferred_from_name':family,
                          'hardware_family_seen_in_training':family in {r.rsplit('_',1)[0] for r in base['train_receivers']},
                          'training_unseen':role!='train','method_development_seen':True,
                          'P1_P2_individual_signal_seen':rx in ('n210_1','rtl_6'),
                          'old_OOB_role':'development' if rx in development else 'formerly_blind_confirmatory_evaluated',
                          'cached_development_headers':cached,'new_signal_opened':False,
                          'independent_research_holdout':False})
    base.update(status='revised_candidate_history_audited_not_frozen',historical_oob_overlap='all 20 receivers used in previous OOB study; six formerly-blind outcomes already reported in supplied manuscript',
                independent_research_holdout_available_in_this_corpus=False,old_paper_sha256=hashlib.sha256(paper.read_bytes()).hexdigest(),
                history_evidence='OOB_IOTJ.pdf Section V-A, Figure4, RQ6/TableVI; manuscript publication/submission status unverified',
                new_signal_access_enabled=False,automatic_start=False,hardware_type_extrapolation_development_receivers=['n210_1'],
                development_model_selection_reuses_historical_domains=True,
                locked_receivers_interpretation='training-unseen and locked for this protocol; previously evaluated and outcomes seen, not independent of research history',
                receiver_history=table,local_package_dates_sessions_cfo_units='unverified',
                final_protocol_prerequisites=['Freeze preprocessing, configuration search budget, seeds, epoch/checkpoint selection and all 20 per-receiver roles before any new access',
                  'Use 10 receivers for fitting; 4 developer receivers for selection, report same-family and unseen-family separately',
                  'Do not claim six previously evaluated domains are never used in research decisions; genuinely fresh research confirmation requires new data under separately authorized access',
                  'Explicit authorization needed to execute P3; keep Day3/4 and six locked receivers inaccessible now'])
    out=ROOT/'configs/shen_p2a/p3_history_review.candidate.json';out.write_text(json.dumps(base,ensure_ascii=False,indent=2)+'\n');print(out)
