import inspect, pandas, numpy, re
import operator
from hashlib import sha1
from pathlib import Path

from document.report.scripts.image import get_pipeline_from_result
from engine.helpers import write_file, read_file
from engine.nn.training import NNModel
from model.testing import TestingResult
from utils.function import get_functions
from utils.report import create_sorted_result_set

TARGET_DIR = Path('../table/generated')

column_format_result_overview = """l
S[table-format=2.1]@{\,\( \pm \)\,}
S[table-format=1.1, table-number-alignment = left]
S[table-format=2.2]@{\,\( \pm \)\,}
S[table-format=1.2]
S[table-format=2.1]@{\,\( \pm \)\,}
S[table-format=1.1, table-number-alignment = left]
S[table-format=2.2]@{\,\( \pm \)\,}
S[table-format=1.2]
"""

column_format_syllable_duration_stats = """l
S[table-format=3.2]@{\,\( \pm \)\,}
S[table-format=2.2, table-number-alignment = left]
S[table-format=3.2]
S[table-format=3.2]
S[table-format=3.2]
S[table-format=3]
"""

# TARGET_DIR.mkdir(parents=True)


def accuracy(x):
    return '%2.1f' % (x * 100)


def loss(x):
    return '%1.2f' % x


def create_result_overview_table(data):
    with pandas.option_context("max_colwidth", 1000, 'display.precision', 2):
        clean_up_pattern = re.compile(r'^.*_remove_.*$', re.MULTILINE)

        formatters = numpy.repeat([accuracy, loss], 2)
        col_index = pandas.MultiIndex.from_tuples([('Model', '', '')])
        col_index = col_index.append(pandas.MultiIndex.from_product([['Validate', 'Test'], ['Accuracy (\si{\percent})', 'Loss'], ['_remove_', 'it']]))
        tex = pandas.DataFrame(data, columns=col_index) \
            .to_latex(escape=False, index=False, column_format=column_format_result_overview, formatters=numpy.concatenate(([None], formatters, formatters)).tolist())
        return clean_up_pattern.sub('', tex)


def create_syllable_duration_stats_table(result_name):
    pipeline: NNModel = get_pipeline_from_result(result_name)

    # hacky - input files are 2 tasks backward
    rows = [[file.metadata.label, file.metadata.duration * 1000] for file in pipeline.task.src_list.task.src_list.task.src_list.files]
    df = pandas.DataFrame(rows, columns=['syllable_type', 'duration'])
    stats = df.groupby(by='syllable_type').agg({'min', 'max', 'mean', 'median', 'std', 'count'})

    # droplevel(): removes the column level "duration" because of groupBy -> (duration, min), (duration, max) ....
    # reset_index() helps to create a realisation of the groupBy?
    stats = pandas.DataFrame(stats.reset_index()).droplevel(0, axis=1)
    stats.rename(columns={stats.columns[0]: "syllable_type"}, inplace=True)
    stats.sort_values(by='median', ascending=True, inplace=True)

    tex = stats[['syllable_type', 'mean', 'std', 'median', 'min', 'max', 'count']] \
        .to_latex(escape=False, index=False, column_format=column_format_syllable_duration_stats,
                  formatters=numpy.concatenate(([None], numpy.repeat('{:,.2f}'.format, 5), ['{:d}'.format])).tolist())\
        .replace('syllable_type', 'syllable type')\
        .replace('median', '{median/\si{\milli\second}}')\
        .replace('min', '{min/\si{\milli\second}}')\
        .replace('max', '{max/\si{\milli\second}}')
    return re.compile(r'mean.*std').sub(r'\\multicolumn{2}{l}{mean/\\si{\\milli\\second}}', tex)


def table_result_overview_sct_compressed():
    data = []
    for result in create_sorted_result_set('testing_sct_compressed'):
        result_model = TestingResult(result)
        data.append(result_model.get_stat_table_row('features', 'nr_sensitivity'))
    return create_result_overview_table(data)


def table_result_overview_sct_vl():
    data = []
    for result in create_sorted_result_set('testing_sct_vl'):
        result_model = TestingResult(result)
        data.append(result_model.get_stat_table_row('features', 'spectrogram'))
    return create_result_overview_table(data)


def table_result_overview_sct_padded():
    data = []
    for result in create_sorted_result_set('testing_sct_left_padded'):
        result_model = TestingResult(result)
        data.append(result_model.get_stat_table_row('features', 'spectrogram', 'nr_sensitivity'))
    return create_result_overview_table(data)


def table_result_overview_scs():
    data = [TestingResult(result).get_stat_table_row('features', 'split') for result in create_sorted_result_set('testing_scs_r3') if '_raw_' in result['id']]
    return create_result_overview_table(data)


def table_result_overview_scs_all():
    data = [TestingResult(result).get_stat_table_row('features', 'split') for result in create_sorted_result_set('testing_scs_r3')]
    return create_result_overview_table(data)


def table_result_overview_scs_r2():
    data = [TestingResult(result).get_stat_table_row('features', 'split') for result in create_sorted_result_set('testing_scs_all2')]
    return create_result_overview_table(data)


def table_result_overview_scs_r1():
    data = [TestingResult(result).get_stat_table_row('features', 'split') for result in create_sorted_result_set('testing_scs_all')]
    return create_result_overview_table(data)


def table_result_overview_sct():
    data = [TestingResult(result).get_stat_table_row('features', 'nr_sensitivity') for result in create_sorted_result_set('testing_sct_compressed')][0:5]
    data += [TestingResult(result).get_stat_table_row('features', 'spectrogram') for result in create_sorted_result_set('testing_sct_vl')][0:5]
    data += [TestingResult(result).get_stat_table_row('features', 'spectrogram', 'nr_sensitivity') for result in create_sorted_result_set('testing_sct_left_padded')][0:5]
    for i, row in enumerate(data):
        data[i][0] = row[0].split(',')[0] + ', ' + ('compressed' if i < 5 else ('variable length' if i < 10 else 'left padded')) + "," + row[0].split(',')[-1]

    data = sorted(data, key=operator.itemgetter(5, 2), reverse=True)
    return create_result_overview_table(data)


def table_syllable_duration_stats():
    # the pipeline for this experiments include all possible syllable
    return create_syllable_duration_stats_table('testing_sct_vl')


def table_syllable_duration_stats_compressed():
    return create_syllable_duration_stats_table('testing_sct_compressed')


def run(forced=None):
    TARGET_DIR.mkdir(exist_ok=True)
    forced = {} if forced is None else set(forced)

    for f, fx in get_functions(__name__, 'table_'):
        name = f[6:]
        dest = TARGET_DIR / f'{name}.tex'

        digest = sha1(inspect.getsource(fx).encode()).hexdigest()
        digest_path = Path(f'{dest}.sha1')

        if f in forced or not dest.is_file() or not (digest_path.is_file() and digest == read_file(digest_path)):
            print(f'Generating {dest}...')
            write_file(dest, fx())
            write_file(digest_path, digest)
        else:
            print(f'Skipped {dest}')
    print('Done')


if __name__ == '__main__':
    run()
