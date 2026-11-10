"""Wait for committed campaigns, then review CPU metadata once. Linux watcher."""
import argparse
import json
import os
import time
from pathlib import Path
from review import read, digest, review

def campaigns_ready(project):
    for version, directory in [('V6', 'v6-resolution-final-01'), ('V7', 'v7-controls-448-01')]:
        root = project/'runs'/directory
        marker = root/(version+'_COMPLETE.json')
        if not marker.exists():
            return False
        record = read(marker)
        if record.get('runs') != 10 or record.get('epochs') != 300:
            raise ValueError('Unexpected campaign completion: '+str(marker))
        if digest(root/'RESULTS.json') != record.get('results_sha256'):
            raise ValueError('Campaign results hash mismatch: '+str(root))
    return True

def save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)

def main():
    import fcntl
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out/'watcher.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Watcher already active', flush=True)
            return
        def status(message):
            line = time.strftime('%Y-%m-%d %H:%M:%S%z')+' '+message
            print(line, flush=True)
            (a.out/'STATUS.txt').write_text(line+'\n', encoding='utf-8')
        if (a.out/'ANALYSIS_COMPLETE.json').exists():
            marker = read(a.out/'ANALYSIS_COMPLETE.json')
            if digest(a.out/'REVIEW.json') != marker['review_sha256']:
                raise ValueError('Existing analysis modified')
            status('ANALYSIS_ALREADY_COMPLETE; provenance review still required')
            return
        try:
            status('WAITING_FOR_V7_COMPLETE; CPU only; no training changes')
            while not campaigns_ready(a.project):
                time.sleep(30)
            status('CHECKING_15_RUNS_AND_PAIRED_PREDICTIONS')
            result = review(a.project)
            if result['status'] != 'READY_FOR_PROVENANCE_REVIEW':
                raise ValueError('Campaign markers present but runs missing: '+str(result['missing_runs']))
            save(a.out/'REVIEW.json', result)
            summary = ['# Post-V7 evidence review', '',
                       'All fifteen 448 runs have matching prediction identities and completion metadata.',
                       'Checkpoint binaries were not rehashed. Provenance and patient uncertainty remain unresolved.',
                       'Intervals describe seed variability only. These are exploratory comparisons.', '']
            for c in result['contrasts']:
                summary.append(f"- {c['cohort']} | {c['contrast']} | {c['metric']}: delta={c['mean_delta']:.5f}; positive seeds={c['positive_seeds']}/5")
            (a.out/'SUMMARY.md').write_text('\n'.join(summary)+'\n', encoding='utf-8')
            save(a.out/'ANALYSIS_COMPLETE.json', dict(review_sha256=digest(a.out/'REVIEW.json'),
                 summary_sha256=digest(a.out/'SUMMARY.md'), provenance_verified=False,
                 patient_cluster_inference_performed=False, training_launched=False))
            status('ANALYSIS_COMPLETE; review SUMMARY.md; provenance audit still required')
        except Exception as e:
            status('FAILED '+str(e))
            raise

if __name__ == '__main__':
    main()
