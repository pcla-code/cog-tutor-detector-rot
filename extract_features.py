import os
from bisect import bisect
import argparse

import pandas as pd
import numpy as np
from scipy.stats import zscore


def approx_line_count(file_path: str, samples: int=1000, sample_size: int=4096,
                      line_ending: bytes=b'\n') -> int:
    """Quickly approximate the number of lines in a huge file via random sampling.

    Args:
        file_path (str): Path to file
        samples (int, optional): Number of samples to draw. Defaults to 1000.
        sample_size (int, optional): Size of each sample, in bytes. Defaults to 4096.
        line_ending (bytes, optional): Line ending byte string to search for. Defaults to b'\n'.

    Returns:
        int: Approximate number of lines in the file
    """
    fsize = os.path.getsize(file_path)
    match_count = 0
    with open(file_path, 'rb', buffering=0) as infile:
        for offset in np.random.randint(0, fsize - sample_size + 1, samples):
            infile.seek(offset)
            sample = infile.read(sample_size)
            match_count += sample.count(line_ending)
    return int(match_count / samples / sample_size * fsize)


# Fast comparison to previous value from https://stackoverflow.com/questions/41399538
def comp_prev(np_array: np.ndarray) -> np.ndarray:
    return np.concatenate(([False], np_array[1:] == np_array[:-1]))  # Fill first NaN with False


def get_features(one_student_df: pd.DataFrame) -> pd.DataFrame:
    assert not one_student_df.isna().values.any(), 'The input DataFrame contains NaNs'
    assert one_student_df.index[0] == 0 and one_student_df.index[-1] == len(one_student_df) - 1, \
        'Input DataFrame has a non-consecutive or non-zero-based index'
    feature_df = one_student_df.copy()
    feature_df['duration'] = feature_df.server_time.diff() / 1000
    feature_df['duration_sd'] = feature_df[['duration']].apply(zscore, nan_policy='omit')

    for outcome in ['OK', 'BUG', 'ERROR', 'INITIAL_HINT']:
        feature_df['assess_' + outcome] = (one_student_df.tutor_outcome == outcome) * 1

    feature_df['prob_first_att'] = 0
    feature_df.loc[feature_df[['problem_id', 'goalnode_id']].drop_duplicates().index,
                   'prob_first_att'] = 1
    feature_df['step_first_att'] = 0
    feature_df.loc[feature_df.goalnode_id.drop_duplicates().index, 'step_first_att'] = 1

    goalnode_group = feature_df.groupby('goalnode_id')
    correct_attempts = goalnode_group.assess_OK.sum()
    count_attempts = goalnode_group.assess_OK.count()
    wrong_attempts = count_attempts - correct_attempts
    error_perc = wrong_attempts / count_attempts
    max_attempts = feature_df.groupby(['problem_id', 'goalnode_id']).attempt.max()

    # This will be very slow; vectorization might help depending on how many goalnode IDs there are
    # Or maybe groupby and apply to each group in a view? Is that allowed?
    feature_df['wrong_attempts'] = [wrong_attempts[gn_id] for gn_id in feature_df.goalnode_id]
    feature_df['error_perc'] = [error_perc[gn_id] for gn_id in feature_df.goalnode_id]
    feature_df['numsteps'] = [max_attempts[(p_id, g_id)]
                              for p_id, g_id in zip(feature_df.problem_id, feature_df.goalnode_id)]

    feature_df['help_or_error'] = ((feature_df.tutor_outcome != 'OK_AMBIGUOUS') &
                                   (feature_df.tutor_outcome != 'OK')) * 1
    feature_df['help_and_errors_count'] = \
        feature_df.groupby(['user_id', 'goalnode_id']).help_or_error.transform('sum')

    feature_df['error_count'] = feature_df.assess_BUG + feature_df.assess_ERROR
    feature_df['error_count_last_5'] = feature_df.error_count.rolling(5).sum()
    feature_df['dur_sd_prev3'] = feature_df.duration_sd.rolling(3).sum()
    feature_df['dur_sd_prev5'] = feature_df.duration_sd.rolling(5).sum()
    feature_df['assess_HINT_LEVEL_CHANGE'] = (feature_df.tutor_outcome == 'HINT_LEVEL_CHANGE') * 1
    feature_df['help_attempts_last_8'] = feature_df.assess_HINT_LEVEL_CHANGE.rolling(8).sum()
    goalnode_id = feature_df.problem_id + feature_df.goalnode_id
    # TODO: The prob_step_last_5 count might be reversed (including in this redone version)
    #   -- Doesn't this actually measure the number of *same* problem steps, i.e., 5 - count?
    #   -- Not that it matters for information gain in ML
    feature_df['same_goalnode'] = comp_prev(goalnode_id.values) * 1
    feature_df['prob_step_last_5'] = feature_df.same_goalnode.rolling(5).sum()

    feature_df['pknow'] = feature_df.pknow.astype(float)
    feature_df['pknow_direct'] = feature_df.pknow.astype(float)
    feature_df.loc[feature_df.attempt != 1, 'pknow_direct'] = -1

    return feature_df


def get_clip_ids(feat_df: pd.DataFrame) -> dict:
    feat = feat_df.copy()
    refined_clips = {}
    for size in [8, 7, 6, 5, 4, 3, 2]:
        durations = feat.loc[~feat.index.duplicated(keep='first')].duration.rolling(size).sum()
        clips = feat.loc[durations <= 20].index
        if len(clips) == 0:
            continue
        refined_clips[size] = [clips[0]]
        for i in clips:
            if i - refined_clips[size][-1] >= size:
                refined_clips[size].append(i)
        dropIDs = []
        uniqueIDs = feat.loc[~feat.index.duplicated(keep='first')].index.to_list()
        for clip_id in refined_clips[size]:
            pos = bisect(uniqueIDs, clip_id)
            dropIDs.extend(uniqueIDs[pos-size:pos])
        feat = feat.drop(index=dropIDs)
    return refined_clips


def process_clips(feat_df: pd.DataFrame, clip_ids: dict) -> pd.DataFrame:
    funique = feat_df[~feat_df.index.duplicated()]  # These are all unique already?
    findex = funique.index
    cliplist = []
    for size in sorted(clip_ids.keys()):
        for ix in clip_ids[size]:
            start = findex[ix - size + 1]  # Valid because we asserted consecutive 0-based index
            cur = feat_df.loc[start:ix].describe().unstack().to_frame().T
            # describe() is by far the slowest part of the whole process (>95% of the runtime)
            # Hard to improve on it though...
            cur.index = [ix]
            cliplist.append(cur)
    clips = pd.concat(cliplist)
    clips.columns = [f'{i[0]}_{i[1]}' for i in clips.columns]
    for f in feat_df.columns:
        clips[f + '_sum'] = clips[f + '_mean'] * clips[f + '_count']
    return clips


# the order of features that the classifiers expect
ORDER = ['assess_BUG_25%', 'assess_BUG_50%', 'assess_BUG_75%', 'assess_BUG_count', 'assess_BUG_max',
         'assess_BUG_mean', 'assess_BUG_min', 'assess_BUG_std', 'assess_BUG_sum', 'assess_OK_25%',
         'assess_OK_50%', 'assess_OK_75%', 'assess_OK_count', 'assess_OK_max', 'assess_OK_mean',
         'assess_OK_min', 'assess_OK_std', 'assess_OK_sum', 'assess_ERROR_25%', 'assess_ERROR_50%',
         'assess_ERROR_75%', 'assess_ERROR_count', 'assess_ERROR_max', 'assess_ERROR_mean',
         'assess_ERROR_min', 'assess_ERROR_std', 'assess_ERROR_sum', 'assess_INITIAL_HINT_25%',
         'assess_INITIAL_HINT_50%', 'assess_INITIAL_HINT_75%', 'assess_INITIAL_HINT_count',
         'assess_INITIAL_HINT_max', 'assess_INITIAL_HINT_mean', 'assess_INITIAL_HINT_min',
         'assess_INITIAL_HINT_std', 'assess_INITIAL_HINT_sum', 'dur_sd_prev3_25%',
         'dur_sd_prev3_50%', 'dur_sd_prev3_75%', 'dur_sd_prev3_count', 'dur_sd_prev3_max',
         'dur_sd_prev3_mean', 'dur_sd_prev3_min', 'dur_sd_prev3_std', 'dur_sd_prev3_sum',
         'dur_sd_prev5_25%', 'dur_sd_prev5_50%', 'dur_sd_prev5_75%', 'dur_sd_prev5_count',
         'dur_sd_prev5_max', 'dur_sd_prev5_mean', 'dur_sd_prev5_min', 'dur_sd_prev5_std',
         'dur_sd_prev5_sum', 'duration_sd_25%', 'duration_sd_50%', 'duration_sd_75%',
         'duration_sd_count', 'duration_sd_max', 'duration_sd_mean', 'duration_sd_min',
         'duration_sd_std', 'duration_sd_sum', 'error_count_last_5_25%', 'error_count_last_5_50%',
         'error_count_last_5_75%', 'error_count_last_5_count', 'error_count_last_5_max',
         'error_count_last_5_mean', 'error_count_last_5_min', 'error_count_last_5_std',
         'error_count_last_5_sum', 'error_perc_25%', 'error_perc_50%', 'error_perc_75%',
         'error_perc_count', 'error_perc_max', 'error_perc_mean', 'error_perc_min',
         'error_perc_std', 'error_perc_sum', 'help_and_errors_count_25%',
         'help_and_errors_count_50%', 'help_and_errors_count_75%', 'help_and_errors_count_count',
         'help_and_errors_count_max', 'help_and_errors_count_mean', 'help_and_errors_count_min',
         'help_and_errors_count_std', 'help_and_errors_count_sum', 'help_attempts_last_8_25%',
         'help_attempts_last_8_50%', 'help_attempts_last_8_75%', 'help_attempts_last_8_count',
         'help_attempts_last_8_max', 'help_attempts_last_8_mean', 'help_attempts_last_8_min',
         'help_attempts_last_8_std', 'help_attempts_last_8_sum', 'numsteps_25%', 'numsteps_50%',
         'numsteps_75%', 'numsteps_count', 'numsteps_max', 'numsteps_mean', 'numsteps_min',
         'numsteps_std', 'numsteps_sum', 'pknow_25%', 'pknow_50%', 'pknow_75%', 'pknow_count',
         'pknow_direct_25%', 'pknow_direct_50%', 'pknow_direct_75%', 'pknow_direct_count',
         'pknow_direct_max', 'pknow_direct_mean', 'pknow_direct_min', 'pknow_direct_std',
         'pknow_direct_sum', 'pknow_max', 'pknow_mean', 'pknow_min', 'pknow_std', 'pknow_sum',
         'prob_first_att_25%', 'prob_first_att_50%', 'prob_first_att_75%', 'prob_first_att_count',
         'prob_first_att_max', 'prob_first_att_mean', 'prob_first_att_min', 'prob_first_att_std',
         'prob_first_att_sum', 'prob_step_last_5_25%', 'prob_step_last_5_50%',
         'prob_step_last_5_75%', 'prob_step_last_5_count', 'prob_step_last_5_max',
         'prob_step_last_5_mean', 'prob_step_last_5_min', 'prob_step_last_5_std',
         'prob_step_last_5_sum', 'step_first_att_25%', 'step_first_att_50%', 'step_first_att_75%',
         'step_first_att_count', 'step_first_att_max', 'step_first_att_mean', 'step_first_att_min',
         'step_first_att_std', 'step_first_att_sum', 'wrong_attempts_25%', 'wrong_attempts_50%',
         'wrong_attempts_75%', 'wrong_attempts_count', 'wrong_attempts_max', 'wrong_attempts_mean',
         'wrong_attempts_min', 'wrong_attempts_std', 'wrong_attempts_sum']


def process_one_student(df: pd.DataFrame, progress_prop: float, out_dir: str) -> None:
    fname = os.path.join(out_dir, df.iloc[0].user_id + '.csv')
    if os.path.exists(fname):
        print('%.2f%%' % (progress_prop * 100), 'Already done; skipping', df.iloc[0].user_id)
        return
    print('%.2f%%' % (progress_prop * 100), 'Processing', df.iloc[0].user_id)
    df_to_input = df.dropna().drop_duplicates().sort_values('server_time').reset_index(drop=True)
    if len(df_to_input) == 0:
        print('No rows to process due to NaNs!')
        return
    features = get_features(df_to_input)
    clip_ids = get_clip_ids(features)
    if not clip_ids:
        print('No clip IDs calculated!')
        return
    processed = process_clips(features.drop(columns=[
        'user_id', 'problem_id', 'tutor_outcome',
        'goalnode_id', 'server_time', 'attempt', 'help_level',
        'section_id', 'skill', 'help_or_error', 'semantic_event_id',
        'error_count'
    ]), clip_ids)
    processed[ORDER].dropna().rename_axis('orig_index').to_csv(fname)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Extract features from DataShop formatted MATHia logs')
    ap.add_argument('mathia_csv', help='Path to MATHia log file to use as input')
    ap.add_argument('output_dir', help='Output directory path')
    args = ap.parse_args()

    print('Loading')
    line_count = approx_line_count(args.mathia_csv, 10000)
    print('Approximately', line_count, 'lines in input file')

    prev_pid = None
    prev_df = None
    processed_pids = set()
    with pd.read_csv(args.mathia_csv, chunksize=100000) as df_reader:
        for chunk_i, chunk in enumerate(df_reader):
            chunk = chunk[[
                'Anon Student Id', 'Time', 'Level (Workspace Id)', 'Problem Name', 'Step Name',
                'Outcome', 'Help Level', 'Attempt At Step', 'KC Model(MATHia)',
                'CF (Skill New p-Known)', 'CF (Semantic Event Id)'
            ]]
            chunk.columns = ['user_id', 'server_time', 'section_id', 'problem_id', 'goalnode_id',
                             'tutor_outcome', 'help_level', 'attempt', 'skill', 'pknow',
                             'semantic_event_id']

            progress_prop = min(.9999, chunk_i * 100000 / line_count)
            for pid, pid_df in chunk.groupby('user_id', sort=False):  # Preserve original order
                if pid != prev_pid:  # First chunk or new PID
                    if prev_pid:  # Previous PID done; process
                        process_one_student(prev_df, progress_prop, args.output_dir)
                        assert prev_pid not in processed_pids, 'Ordering assumption violated'
                        processed_pids.add(prev_pid)
                    prev_pid = pid
                    prev_df = pid_df
                else:  # Continue on from previous chunk
                    prev_df = pd.concat([prev_df, pid_df])
    if prev_pid:  # One last PID to process
        process_one_student(prev_df, 1, args.output_dir)
