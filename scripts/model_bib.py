from engine.helpers import write_file
from model.testing import TestingResult
from utils.report import create_sorted_result_set


def run():
    bib_result = []
    index = 1
    for testing_result in ['testing_sct_compressed', 'testing_sct_vl', 'testing_sct_left_padded', 'testing_scs_r3']:
        for result in create_sorted_result_set(testing_result):
            try:
                bib_result.append(TestingResult(result).to_bib(index))
                index += 1
            except Exception:
                print('faild to handle model ' + result['id'])

    write_file('../biblio/model.bib', '\n'.join(bib_result))


if __name__ == '__main__':
    run()
