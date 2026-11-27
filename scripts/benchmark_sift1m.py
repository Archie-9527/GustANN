#!/usr/bin/env python3
"""Benchmark SIFT1M on the host or on one preconfigured SwarmIO instance per run."""

import argparse
import json
import re
import shlex
import statistics
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BATCH = 10_000
TOPK = 10
RECALL_TARGETS = (0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
DEFAULT_SCENARIOS = ('diskann', 'libaio', 'liburing')
SIMULATED = {'bam-4k': 4096, 'bam-512': 512}
SWARMIO_IO_UNIT = Path('/proc/swarmio/io_unit_shift')
PHYSICAL = 'bam-physical-4k'
DEFAULT_SEARCH_LIST = (10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
INDEX_SUFFIXES = ('_disk.index', '_pq_pivots.bin', '_pq_compressed.bin')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sift-dir', type=Path, required=True, help='Directory containing sift_{base,query}.fvecs and sift_groundtruth.ivecs')
    parser.add_argument('--out-dir', type=Path, required=True, help='New directory for logs and results')
    parser.add_argument('--index-prefix', type=Path, help='Use or build a float L2 DiskANN index at this prefix')
    parser.add_argument('--gustann-bin-dir', type=Path, default=REPO / 'build/bin')
    parser.add_argument('--diskann-bin-dir', type=Path, default=REPO / 'deps/DiskANN/build/apps')
    parser.add_argument('--scenarios', nargs='+', choices=(*DEFAULT_SCENARIOS, *SIMULATED, PHYSICAL),
                        default=DEFAULT_SCENARIOS, help='Defaults to host scenarios. Select at most one SwarmIO size.')
    parser.add_argument('--search-list', nargs='+', type=int, default=DEFAULT_SEARCH_LIST)
    parser.add_argument('--trials', type=int, default=1, help='Fresh process and full batch per trial')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--beamwidth', type=int, default=4)
    parser.add_argument('--mini-batch', type=int, default=128, help='GustANN hybrid internal minibatch')
    parser.add_argument('--contexts', type=int, default=2, help='GustANN hybrid contexts per worker')
    parser.add_argument('--cache-mib', type=int, default=64, help='Same GPU cache data capacity in each BaM mode')
    parser.add_argument('--bam-queues', type=int, default=135)
    parser.add_argument('--bam-launcher', default='sudo', help='Command prefix for BaM, or an empty string')
    parser.add_argument('--physical-ctrls', type=int, default=1, help='Only used by the deferred physical BaM scenario')
    parser.add_argument('--load-bam', action='store_true', help='Load the index at LBA 0 using the selected BaM I/O size')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without building, writing media or running searches')
    args = parser.parse_args()
    if any(value < TOPK for value in args.search_list):
        parser.error('search list values must be at least 10')
    if min(args.trials, args.threads, args.beamwidth, args.mini_batch, args.contexts,
           args.cache_mib, args.bam_queues, args.physical_ctrls) <= 0:
        parser.error('counts must be positive')
    args.scenarios = tuple(dict.fromkeys(args.scenarios))
    if len(set(args.scenarios) & SIMULATED.keys()) > 1:
        parser.error('select one SwarmIO instance per run: bam-4k or bam-512')
    if PHYSICAL in args.scenarios and len(args.scenarios) != 1:
        parser.error('run the physical BaM scenario separately, after switching the controller mapping')
    if args.load_bam and not any(scenario.startswith('bam-') for scenario in args.scenarios):
        parser.error('--load-bam requires an explicit BaM scenario')
    args.search_list = sorted(set(args.search_list))
    for name in ('sift_dir', 'out_dir', 'gustann_bin_dir', 'diskann_bin_dir', 'index_prefix'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.index_prefix is None:
        args.index_prefix = args.out_dir / 'index' / 'sift1m'
    return args


def run(command, log, dry_run):
    argv = [str(arg) for arg in command]
    display = shlex.join(argv)
    print(display, flush=True)
    if dry_run:
        return ''
    with log.open('w', encoding='utf-8') as transcript:
        transcript.write(f'$ {display}\n')
        transcript.flush()
        result = subprocess.run(argv, stdout=transcript, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'exit {result.returncode}; see {log}')
    output = log.read_text(encoding='utf-8', errors='replace')
    if 'Timed out waiting for BaM controller to stop' in output:
        raise RuntimeError(f'BaM controller did not stop; see {log}')
    return output


def convert_vecs(source, destination, expected_rows, expected_dim=None, min_dim=1):
    """Strip each row's little-endian width and write DiskANN's two-integer header."""
    with source.open('rb') as encoded:
        first = encoded.read(4)
        if len(first) != 4:
            raise ValueError(f'empty or truncated vector file: {source}')
        width = struct.unpack('<I', first)[0]
        if width < min_dim or (expected_dim is not None and expected_dim != width):
            raise ValueError(f'unexpected vector width {width} in {source}')
        row_bytes = 4 * (width + 1)
        if source.stat().st_size != expected_rows * row_bytes:
            raise ValueError(f'expected {expected_rows} records in {source}')
        encoded.seek(0)
        with destination.open('wb') as decoded:
            decoded.write(struct.pack('<II', expected_rows, width))
            for _ in range(expected_rows):
                row = encoded.read(row_bytes)
                if len(row) != row_bytes or row[:4] != first:
                    raise ValueError(f'inconsistent vector record in {source}')
                decoded.write(row[4:])


def prepare_index(args):
    products = [Path(str(args.index_prefix) + suffix) for suffix in INDEX_SUFFIXES]
    if all(path.is_file() for path in products):
        return
    if any(path.exists() for path in products):
        raise ValueError(f'incomplete index at {args.index_prefix}; use a complete or a new index prefix')
    base_bin = args.out_dir / 'sift_base.fbin'
    if not args.dry_run:
        args.index_prefix.parent.mkdir(parents=True, exist_ok=True)
        convert_vecs(args.sift_dir / 'sift_base.fvecs', base_bin, 1_000_000, expected_dim=128)
    run([args.diskann_bin_dir / 'build_disk_index', '--data_type', 'float', '--dist_fn', 'l2',
         '--data_path', base_bin, '--index_path_prefix', args.index_prefix, '-R', 64, '-L', 200,
         '-B', 0.03, '-M', 4, '-T', args.threads], args.out_dir / 'build_index.log', args.dry_run)
    if not args.dry_run and not all(path.is_file() for path in products):
        raise RuntimeError(f'build did not produce the three expected files at {args.index_prefix}')


def check_swarmio_size(scenario):
    expected = SIMULATED[scenario].bit_length() - 1
    try:
        reported = SWARMIO_IO_UNIT.read_text(encoding='ascii').strip()
    except (FileNotFoundError, PermissionError):
        print(f'{scenario}: cannot read {SWARMIO_IO_UNIT}; using the operator-selected instance size',
              file=sys.stderr)
        return None
    if reported != str(expected):
        raise ValueError(f'{scenario} requires SwarmIO io_unit_shift={expected}, but {SWARMIO_IO_UNIT} '
                         f'reports {reported!r}. Create the matching instance and retry.')
    print(f'{scenario}: verified SwarmIO io_unit_shift={reported}', flush=True)
    return expected


def gustann_command(args, scenario, search=None):
    bam = scenario.startswith('bam-')
    binary = 'search_disk' if bam else 'search_disk_hybrid'
    command = shlex.split(args.bam_launcher) if bam else []
    command += [args.gustann_bin_dir / binary, '--data_type', 'float',
                '--index', f'{args.index_prefix}_disk.index', '--pq_data', f'{args.index_prefix}_pq',
                '--query', args.sift_dir / 'sift_query.fvecs']
    if bam:
        io_size = SIMULATED.get(scenario, 4096)
        controllers = args.physical_ctrls if scenario == PHYSICAL else 1
        cache_pages = args.cache_mib * (1 << 20) // io_size
        command += ['--num_ctrls', controllers, '--cache_page', cache_pages,
                    '--bam-io-size', io_size, '--bam-queues', args.bam_queues]
        if scenario != PHYSICAL:
            command += ['--bam-sim']
    else:
        command += ['--io_backend', 'aio' if scenario == 'libaio' else 'uring',
                    '-B', args.mini_batch, '-T', args.threads, '-C', args.contexts]
    if search is None:
        return command + ['--copy_data', '--copy_only']
    return command + ['--ground_truth', args.sift_dir / 'sift_groundtruth.ivecs',
                      '--topk', TOPK, '--ef_search', search, '-R', 1, '--benchmark']


def diskann_command(args, search, result_prefix):
    return [args.diskann_bin_dir / 'search_disk_index', '--data_type', 'float', '--dist_fn', 'l2',
            '--index_path_prefix', args.index_prefix, '--result_path', result_prefix,
            '--query_file', args.out_dir / 'sift_query.fbin', '--gt_file', args.out_dir / 'sift_groundtruth.bin',
            '-K', TOPK, '-L', search, '-T', args.threads, '-W', args.beamwidth, '--num_nodes_to_cache', 0]


def diskann_recall(output_prefix, search, truth):
    data = Path(f'{output_prefix}_{search}_idx_uint32.bin').read_bytes()
    if len(data) != 8 + BATCH * TOPK * 4 or struct.unpack_from('<II', data) != (BATCH, TOPK):
        raise ValueError('unexpected DiskANN result dimensions')
    rows = struct.iter_unpack('<' + 'I' * TOPK, memoryview(data)[8:])
    return sum(len(set(row) & expected) for row, expected in zip(rows, truth)) / (BATCH * TOPK)


def parse_diskann(log, search, output_prefix, truth):
    pattern = re.compile(rf'^\s*{search}\s+\d+(?:\s+\d+(?:\.\d+)?){{7}}\s*$', re.MULTILINE)
    entries = pattern.findall(log)
    if len(entries) != 1:
        raise ValueError(f'expected one DiskANN table row for L={search}')
    _, _, qps, mean_us, p999_us, mean_ios, _, _, _ = map(float, entries[0].split())
    # The upstream CLI rounds Mean IOs. Its SIFT1M node reads are one 4 KiB sector.
    commands = round(mean_ios * BATCH)
    return dict(batch_seconds=BATCH / qps, recall=diskann_recall(output_prefix, search, truth),
                primary_read_commands=commands, primary_read_bytes=commands * 4096,
                extra_read_commands=0, extra_read_bytes=0, io_counter='rounded_diskann_mean_ios',
                query_mean_ms=mean_us / 1000, query_p999_ms=p999_us / 1000)


def parse_gustann(log, scenario, search):
    rows = [line.split()[1:] for line in log.splitlines() if line.startswith('GUSTANN_BENCH ')]
    if len(rows) != 1 or len(rows[0]) != 9:
        raise ValueError('expected one GUSTANN_BENCH row')
    ef, queries, seconds, recall, reads, read_bytes, extras, extra_bytes, result_hash = rows[0]
    if (int(ef), int(queries)) != (search, BATCH):
        raise ValueError(f'expected L={search} and batch={BATCH}, received {ef} and {queries}')
    io_size = SIMULATED.get(scenario, 4096)
    primary_commands = int(reads)
    primary_bytes = int(read_bytes)
    io_counter = 'gustann_application'
    if scenario in SIMULATED:
        expected = f'BaM I/O: {io_size} bytes, {io_size // 512} LBA(s) of 512 bytes'
        supplemental = ', 1 LBA per supplemental READ'
        if expected not in log or supplemental not in log or int(extra_bytes) != int(extras) * 512:
            raise ValueError(f'{scenario} did not report the expected namespace LBA and I/O sizes')
        if primary_commands == primary_bytes == 0:
            # Older binaries read the initialization range instead of the range updated by the kernel.
            _, timing, after_search = log.rpartition('Use time:')
            counters = re.findall(r'^#READ IOs:[ \t]*(\d+)[^\n]*#Misses:[ \t]*(\d+)[^\n]*CLSize:[ \t]*(\d+)[ \t]*$',
                                  after_search, re.MULTILINE) if timing else []
            if len(counters) == 1:
                logged_reads, misses, logged_page_size = map(int, counters[0])
                if logged_reads == misses > 0 and logged_page_size == io_size:
                    primary_commands = logged_reads
                    primary_bytes = logged_reads * io_size
                    io_counter = 'bam_cache_misses'
                    print(f'{scenario} L={search}: structured primary counters are zero. '
                          f'Using {logged_reads} primary reads from the post-search BaM cache counters.',
                          file=sys.stderr)
    if primary_commands <= 0 or primary_bytes != primary_commands * io_size:
        raise ValueError(f'{scenario} reported {primary_commands} primary reads and {primary_bytes} primary bytes. '
                         f'Expected a positive read count and {io_size} bytes per primary read.')
    return dict(batch_seconds=float(seconds), recall=float(recall), primary_read_commands=primary_commands,
                primary_read_bytes=primary_bytes, extra_read_commands=int(extras),
                extra_read_bytes=int(extra_bytes), result_hash=result_hash, io_counter=io_counter)


def summarize_row(scenario, search, trials):
    # The selected recall is the worst observed trial; throughput uses the median batch.
    seconds = statistics.median(t['batch_seconds'] for t in trials)
    primary_commands = statistics.median(t['primary_read_commands'] for t in trials)
    primary_bytes = statistics.median(t['primary_read_bytes'] for t in trials)
    extra_commands = statistics.median(t['extra_read_commands'] for t in trials)
    extra_bytes = statistics.median(t['extra_read_bytes'] for t in trials)
    result = dict(scenario=scenario, search_list=search, trials=len(trials), batch=BATCH,
                  recall_at_10=min(t['recall'] for t in trials), batch_latency_ms=seconds * 1000,
                  qps=BATCH / seconds, read_iops=(primary_commands + extra_commands) / seconds,
                  read_mib_s=(primary_bytes + extra_bytes) / seconds / (1 << 20),
                  primary_read_iops=primary_commands / seconds,
                  primary_read_mib_s=primary_bytes / seconds / (1 << 20),
                  supplemental_read_iops=extra_commands / seconds,
                  supplemental_read_mib_s=extra_bytes / seconds / (1 << 20),
                  io_counter=trials[0]['io_counter'])
    if scenario == 'diskann':
        for metric in ('query_mean_ms', 'query_p999_ms'):
            result[metric] = statistics.median(t[metric] for t in trials)
    if scenario in SIMULATED:
        result['simulator_io_unit_shift'] = trials[0]['simulator_io_unit_shift']
    return result


def write_results(out_dir, scenarios, rows):
    selected = []
    for scenario in scenarios:
        for target in RECALL_TARGETS:
            eligible = [row for row in rows if row['scenario'] == scenario and row['recall_at_10'] >= target]
            if eligible:
                best = max(eligible, key=lambda row: (row['qps'], -row['search_list']))
                selected.append(dict(target_recall=target, status='reached', **best))
            else:
                best_recall = max(row['recall_at_10'] for row in rows if row['scenario'] == scenario)
                selected.append(dict(scenario=scenario, target_recall=target, status='unreached',
                                     best_measured_recall=best_recall))
                print(f'{scenario}: {target:.2f} not reached (best {best_recall:.5f})', file=sys.stderr)
    for filename, payload in (('sweep.json', rows), ('recall_targets.json', selected)):
        (out_dir / filename).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Benchmark reports: {out_dir / "recall_targets.json"}')


def main():
    args = arguments()
    bam_scenario = next((scenario for scenario in args.scenarios if scenario.startswith('bam-')), None)
    simulator_shift = None
    if bam_scenario in SIMULATED and not args.dry_run:
        simulator_shift = check_swarmio_size(bam_scenario)
    truth = None
    if not args.dry_run:
        args.out_dir.mkdir(parents=True)
        query_bin = args.out_dir / 'sift_query.fbin'
        truth_bin = args.out_dir / 'sift_groundtruth.bin'
        convert_vecs(args.sift_dir / 'sift_query.fvecs', query_bin, BATCH, expected_dim=128)
        convert_vecs(args.sift_dir / 'sift_groundtruth.ivecs', truth_bin, BATCH, min_dim=TOPK)
        if 'diskann' in args.scenarios:
            matrix = truth_bin.read_bytes()
            _, width = struct.unpack_from('<II', matrix)
            truth = [set(struct.unpack_from('<' + 'I' * TOPK, matrix, 8 + 4 * width * i)) for i in range(BATCH)]
    prepare_index(args)
    if args.load_bam:
        run(gustann_command(args, bam_scenario), args.out_dir / 'bam_load.log', args.dry_run)

    rows = []
    for scenario in args.scenarios:
        for search in args.search_list:
            trials = []
            for trial in range(1, args.trials + 1):
                stem = args.out_dir / f'{scenario}-L{search}-trial{trial}'
                command = (diskann_command(args, search, stem) if scenario == 'diskann'
                           else gustann_command(args, scenario, search))
                log = run(command, stem.with_suffix('.log'), args.dry_run)
                if args.dry_run:
                    continue
                try:
                    record = (parse_diskann(log, search, stem, truth) if scenario == 'diskann'
                              else parse_gustann(log, scenario, search))
                    if record['batch_seconds'] <= 0 or not 0 <= record['recall'] <= 1:
                        raise ValueError('invalid search time or recall')
                except (ValueError, ZeroDivisionError) as exc:
                    raise ValueError(f'{stem.with_suffix(".log")}: {exc}') from exc
                if scenario in SIMULATED:
                    record['simulator_io_unit_shift'] = simulator_shift
                trials.append(record)
                with (args.out_dir / 'measurements.jsonl').open('a', encoding='utf-8') as raw:
                    raw.write(json.dumps(dict(scenario=scenario, search_list=search, trial=trial, **record),
                                         allow_nan=False) + '\n')
            if trials:
                rows.append(summarize_row(scenario, search, trials))
    if not args.dry_run:
        write_results(args.out_dir, args.scenarios, rows)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f'benchmark: {error}')
