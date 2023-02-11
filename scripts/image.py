import inspect, json, re, pandas
import operator
import shutil
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict, Counter
from hashlib import sha1
from pathlib import Path

import librosa
from librosa.display import waveplot
from matplotlib.ticker import FuncFormatter
from tensorflow.keras.models import load_model
from tensorflow.python.keras.utils.vis_utils import model_to_dot

from engine.audio import load_audio
from engine.helpers import write_file, read_file
from engine.k_fold import get_kfold_set
from engine.metadata import metadata_db
from engine.nn.properties import count_by_label
from engine.plot import add_value_labels, colors

from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
import matplotlib.image as mpimg

from engine.processing.audio.silent import filter_and_transform_labels, get_silent_parts, filter_silent_parts
from engine.settings import BSC_ROOT_DATA_FOLDER
from plot.inputdescriptor import InputDescriptor
from plot.scatter import create_scatter_legend, create_scatter
from plot.settings import create_color_map_for_symbols
from utils.experiment import load_audio_and_db
from utils.function import get_functions
from plot.operator import ResultTaskPropGetter, FunctionMapper, ModelMatcher, Mapper, FeatureMatcher, DictMapper
from utils.report import create_sorted_result_set, get_best_model_pipeline_from_result, split_acc_loss_plots, \
    get_best_model_path_from_result, get_best_result_for_model, confusion_matrix, nn_cnn_2d_lrp

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Tex Gyre Pagella']
rcParams['svg.fonttype'] = 'none'
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Helvetica'] # Tahoma
rcParams['font.size'] = 10
rcParams['image.cmap'] = 'viridis'


def create_sample_pipline():
    ppl = load_audio_and_db('simple_call_test')
    ppl.files = [file for file in ppl.files if file.metadata.duration < .3 and len(file.metadata.labels) > 3]
    ppl.files = [ppl.files[0]]
    return ppl


def dataset_label_dist(ppl, size, x_max):
    kfold = get_kfold_set(ppl)
    tr = count_by_label(kfold.train.files)
    va = count_by_label(kfold.val.files)
    te = count_by_label(kfold.test.files)

    fig, ax = plt.subplots(figsize=size)
    classes = list(sorted(set(tr.keys()), reverse=True))
    ax.barh(classes, [tr[c] for c in classes], color=[colors[c] for c in classes])
    ax.barh(classes, [va[c] for c in classes], color='silver', left=[tr[c] for c in classes], label='Validation')
    ax.barh(classes, [te[c] for c in classes], color='black', left=[tr[c] + va[c] for c in classes], label='Testing')
    ax.set_ylabel('syllable'), ax.set_xlabel('number of samples')
    ax.set_xlim(0, x_max)
    ax.legend()
    add_value_labels(ax, horizontal=True, spacing=5)
    return fig


def create_model_distribution_per_result(result_name, sort=None, **descriptors: InputDescriptor):
    return create_scatter(plt, create_sorted_result_set(result_name), descriptors, sort)


def create_split_acc_vol(stats_file):
    data = json.loads(read_file(stats_file))
    fig, axes = split_acc_loss_plots(plt, range(1, len(data) + 1),
                                     [h['acc'] for h in data],
                                     [h['val_acc'] for h in data],
                                     [h['loss'] for h in data],
                                     [h['val_loss'] for h in data])
    return fig


def create_confusion_matrix(rec_report_file):
    data = json.loads(read_file(rec_report_file))
    return confusion_matrix(plt, data['confusion'])


def create_split_acc_vol_for_model(model_path: Path):
    for stats_file in model_path.parent.glob('*.stats.json'):
        # ignore empty or corrupt stats files
        if stats_file.stat().st_size > 50:
            return create_split_acc_vol(stats_file)


def create_split_acc_vol_for_best_model_of_result(result, model):
    model_path = get_best_model_path_from_result(get_best_result_for_model(result, model))
    return create_split_acc_vol_for_model(model_path)


def create_split_acc_vol_for_specific_model(result, model, hash):
    model_path = get_best_model_path_from_result(get_best_result_for_model(result, model))
    return create_split_acc_vol_for_model(model_path.parent.parent / hash / 'rec')


def create_confusion_for_specific_model(result, model, hash):
    model_path = get_best_model_path_from_result(get_best_result_for_model(result, model))
    for report_file in (model_path.parent.parent / hash / 'rec').glob('*.report.json'):
        return create_confusion_matrix(report_file)


def get_pipeline_from_result(result_path):
    return get_best_model_pipeline_from_result(create_sorted_result_set(result_path)[0])


def image_dataset_sct_compressed(path):
    return dataset_label_dist(get_pipeline_from_result('testing_sct_compressed'), (6, 0.9), 600)


def image_dataset_sct_variable_length(path):
    return dataset_label_dist(get_pipeline_from_result('testing_sct_vl'), (6, 1.5), 600)


def image_dataset_sct_padded(path):
    return dataset_label_dist(get_pipeline_from_result('testing_sct_padded'), (6, 1.5), 600)


def image_dataset_scs_r3_p30s5l60(path):
    return dataset_label_dist(get_pipeline_from_result('testing_scs_r3'), (6, 1.5), 20000)


def image_dataset_scs_r3_p30s5l80(path):
    pipeline = get_best_model_pipeline_from_result(get_best_result_for_model('testing_scs_r3', 'nn_densNet_scs_r3_p30s5l80%_raw_100'))
    return dataset_label_dist(pipeline, (6, 1.5), 20000)


def image_dataset_scs_r3_p30s10l60(path):
    pipeline = get_best_model_pipeline_from_result(get_best_result_for_model('testing_scs_r3', 'nn_densNet_scs_r3_p30s10l60%_raw_100'))
    return dataset_label_dist(pipeline, (6, 1.5), 20000)


def image_dataset_scs_r3_p30s10l80(path):
    pipeline = get_best_model_pipeline_from_result(get_best_result_for_model('testing_scs_r3', 'nn_densNet_scs_r3_p30s10l80%_raw_100'))
    return dataset_label_dist(pipeline, (6, 1.5), 20000)


def sort_model_distribution_sct_compressed(args):
    index = np.argsort([int(v) for v in args['x']])
    for arg_key in args:
        args[arg_key] = np.array(args[arg_key])[index].tolist()


def image_model_distribution_sct_compressed(path):
    return create_model_distribution_per_result(
        'testing_sct_compressed'
        , x=InputDescriptor('Sensitivity', ResultTaskPropGetter('NoiseReduceTask', 'sensitivity', 0, DictMapper({0:'0', 6:'6', 12:'12', 24:'24'})))
        , y=InputDescriptor('Test accuracy', operator.itemgetter('test_acc_m'))
        , c=InputDescriptor('Model', ModelMatcher())
        , markers=InputDescriptor('Features', FeatureMatcher())
        , sort=sort_model_distribution_sct_compressed
    )


def sort_model_distribution_sct_variable_length(args):
    index = np.lexsort((np.array(args["s"])*-1, args['x']))
    for arg_key in args:
        args[arg_key] = np.array(args[arg_key])[index].tolist()


def image_model_distribution_sct_variable_length(path):
    xpps_color_map = create_color_map_for_symbols([2000, 5000, 4000])
    return create_model_distribution_per_result(
        'testing_sct_vl'
        , x=InputDescriptor('Spectrogram height', ResultTaskPropGetter('CreateSpectrogramTask', 'height', 0, DictMapper({256:'256', 300:'300', 512:'512'})))
        , y=InputDescriptor('Test accuracy', operator.itemgetter('test_acc_m'))
        , c=InputDescriptor('Pixel per sec.', ResultTaskPropGetter('CreateSpectrogramTask', 'x_pixels_per_sec', 0, DictMapper(xpps_color_map)))
        , s=InputDescriptor('Pixel per sec.', ResultTaskPropGetter('CreateSpectrogramTask', 'x_pixels_per_sec', 0, DictMapper({2000:10, 4000:20, 5000:50})))
        , markers=InputDescriptor('Features', FeatureMatcher())
        , sort=sort_model_distribution_sct_variable_length
    )


def image_model_distribution_sct_padded(path):
    return create_model_distribution_per_result(
        'testing_sct_left_padded'
        , x=InputDescriptor('Model', ModelMatcher(Mapper()))
        , y=InputDescriptor('Test accuracy', operator.itemgetter('test_acc_m'))
        , c=InputDescriptor('Spectrogram height', ResultTaskPropGetter('CreateSpectrogramTask', 'height', 0, FunctionMapper(lambda v: v / 256)))
        , s=InputDescriptor('Sensitivity', ResultTaskPropGetter('NoiseReduceTask', 'sensitivity', 0, FunctionMapper(lambda v: v * 5 + 15)))
        , markers=InputDescriptor('Features', FeatureMatcher())
    )


def image_model_distribution_scs_r3(path):
    return create_model_distribution_per_result(
        'testing_scs_r3'
        , x=InputDescriptor('Model', ModelMatcher(Mapper()))
        , y=InputDescriptor('Test accuracy', operator.itemgetter('test_acc_m'))
        , c=InputDescriptor('Cover area', ResultTaskPropGetter('SplitIntoPartsTask', 'label_min_cover_length', 0))
        , s=InputDescriptor('Strides', ResultTaskPropGetter('SplitIntoPartsTask', 'strides', 0, FunctionMapper(lambda v: v * 7500)))
        , markers=InputDescriptor('Features', FeatureMatcher())
    )


def image_train_val_densNet_sct_compressed_nrs0_raw_100(path):
    return create_split_acc_vol_for_specific_model('testing_sct_compressed', 'nn_densNet_sct_compressed_nrs0_raw_100', '30e4d1fb298e02abe67bdff782c033720895eb16')


def image_confusion_densNet_sct_compressed_nrs0_raw_100(path):
    return create_confusion_for_specific_model('testing_sct_compressed', 'nn_densNet_sct_compressed_nrs0_raw_100', 'dc82913c523fdd7a54bf2aeb2a19e3f8d0338dd7')


def image_train_val_cnn_2d_sct_compressed_nrs0_raw_100(path):
    return create_split_acc_vol_for_specific_model('testing_sct_compressed', 'nn_cnn_2d_sct_compressed_nrs0_raw_100', 'dc82913c523fdd7a54bf2aeb2a19e3f8d0338dd7')


def image_confusion_cnn_2d_sct_compressed_nrs0_raw_100(path):
    return create_confusion_for_specific_model('testing_sct_compressed', 'nn_cnn_2d_sct_compressed_nrs0_raw_100', 'dc82913c523fdd7a54bf2aeb2a19e3f8d0338dd7')


def image_train_val_densNet_sct_compressed_nrs12_raw_100(path):
    return create_split_acc_vol_for_specific_model('testing_sct_compressed', 'nn_densNet_sct_compressed_nrs12_raw_100', 'd34000382959b81975c33cef4c8a6ff2134809ff')


def image_train_val_lstm_sct_compressed_nrs0_hog_100(path):
    return create_split_acc_vol_for_specific_model('testing_sct_compressed', 'nn_lstm_sct_compressed_nrs0_hog_100', '20416c807440e26c6b6059381d673fad5d4c1d10')


def image_confusion_lstm_sct_compressed_nrs0_hog_100(path):
    return create_confusion_for_specific_model('testing_sct_compressed', 'nn_lstm_sct_compressed_nrs0_hog_100', 'dc82913c523fdd7a54bf2aeb2a19e3f8d0338dd7')


def image_train_val_cnn_1d_sct_compressed_nrs0_raw_100(path):
    model_path = get_best_model_path_from_result(get_best_result_for_model('testing_sct_compressed', 'nn_cnn_1d_sct_compressed_nrs0_raw_100'))
    model_path = model_path.parent.parent / 'c705a68d6b660af11476f6d6e76df7837f882209' / 'rec'
    for stats_file in model_path.parent.glob('*.stats.json'):
        # ignore empty or corrupt stats files
        if stats_file.stat().st_size > 50:
            data = json.loads(read_file(stats_file))
            plt.rcParams['image.cmap'] = 'viridis'
            fig, axes = split_acc_loss_plots(plt, range(1, len(data) + 1),
                                             [h['acc'] for h in data],
                                             [h['val_acc'] for h in data],
                                             [h['loss'] for h in data],
                                             [h['val_loss'] for h in data])

            for i, gap in enumerate(['acc', 'loss']):
                gap_index = 54
                arrow = patches.FancyArrowPatch((gap_index+1, data[gap_index][gap]), (gap_index+1, data[gap_index]['val_'+gap]), arrowstyle='<->', mutation_scale=10)
                axes[i].add_patch(arrow)
                axes[i].text(gap_index+3, data[gap_index][gap] + (data[gap_index]['val_'+gap] - data[gap_index][gap])/2, 'gap')
                # axes[i].vlines([gap_index+1], data[gap_index][gap], data[gap_index]['val_'+gap], color='C2', linestyles="solid", label='gap')
                # axes[i].legend([arrow], ['gap'], loc='lower right')

            return fig


def image_train_val_sct_vl_best_lstm(path):
    return create_split_acc_vol_for_best_model_of_result('testing_sct_vl', 'nn_lstm_')


def image_confusion_lstm_sct_vl_xpps4000_h300_hog_100(path):
    return create_confusion_for_specific_model('testing_sct_vl', 'nn_lstm_sct_vl_xpps4000_h300_hog_100', 'a8508399caabca79c1693271c3a59c060efa7581')


def image_train_val_lstm_sct_vl_xpps5000_h512_hog_100(path):
    return create_split_acc_vol_for_specific_model('testing_sct_vl', 'nn_lstm_sct_vl_xpps5000_h512_hog_100', 'c09f8c4197ac94746535924339ac85187098569a')


def image_confusion_lstm_sct_vl_xpps5000_h512_hog_100(path):
    return create_confusion_for_specific_model('testing_sct_vl', 'nn_lstm_sct_vl_xpps5000_h512_hog_100', 'c09f8c4197ac94746535924339ac85187098569a')


def image_train_val_cnn_2d_sct_left_padded_nrs6xpps5000h512_raw_100(path):
    return create_split_acc_vol_for_specific_model('testing_sct_left_padded', 'nn_cnn_2d_sct_left_padded_nrs6xpps5000h512_raw_100', '847d88a3a8b6972e1090f000d951870468990557')


def image_lrp_cnn_2d_sct_left_padded_nrs6xpps5000h512_raw_100(path):
    model_path = get_best_model_path_from_result(get_best_result_for_model('testing_sct_left_padded', 'cnn_2d_sct_left_padded_nrs6xpps5000h512_raw_100'))
    return nn_cnn_2d_lrp(model_path.parent.parent / '847d88a3a8b6972e1090f000d951870468990557' / 'epoch_100.h5.json', np.array([14, 23]))


def image_train_val_cnn_1d_sct_left_padded_nrs0xpps5000h512_raw_100(path):
    return create_split_acc_vol_for_specific_model('testing_sct_left_padded', 'nn_cnn_1d_sct_left_padded_nrs0xpps5000h512_raw_100', '718b61b8f8318b4687facde6c61b244c54bacde6')


def image_confusion_densNet_sct_left_padded_nrs0xpps2000h256_raw_100(path):
    return create_confusion_for_specific_model('testing_sct_left_padded', 'nn_densNet_sct_left_padded_nrs0xpps2000h256_raw_100', '93688afba2bc3012f83c1906d3ef725f143c75ec')


def image_confusion_lstm_sct_left_padded_nrs0xpps5000h512_hog_100(path):
    return create_confusion_for_specific_model('testing_sct_left_padded', 'nn_lstm_sct_left_padded_nrs0xpps5000h512_hog_100', '0371c671cf46db48c8e325eabf542802c218f829')


def image_confusion_cnn_2d_sct_left_padded_nrs6xpps5000h512_raw_100(path):
    return create_confusion_for_specific_model('testing_sct_left_padded', 'nn_cnn_2d_sct_left_padded_nrs6xpps5000h512_raw_100', 'f984bd3e912d2f458263edbac2bbe939bf6434ef')


def image_train_val_densNet_scs_r3_p30s5l60_raw_100(path):
    return create_split_acc_vol_for_specific_model('testing_scs_r3', 'nn_densNet_scs_r3_p30s5l60%_raw_100', 'cd4a62024eb456eab7ba246c6b82676936bf78b4')


def image_confusion_densNet_scs_r3_p30s5l60_raw_100(path):
    return create_confusion_for_specific_model('testing_scs_r3', 'nn_densNet_scs_r3_p30s5l60%_raw_100', 'cd4a62024eb456eab7ba246c6b82676936bf78b4')


def image_lrp_densNet_scs_r3_p30s5l60_raw_100(path):
    model_path = get_best_model_path_from_result(get_best_result_for_model('testing_scs_r3', 'nn_densNet_scs_r3_p30s5l60%_raw_100'))
    return nn_cnn_2d_lrp(model_path.parent.parent / 'cd4a62024eb456eab7ba246c6b82676936bf78b4' / 'epoch_100.h5.json', np.array([18, 19, 4, 6, 25, 27, 15]))


def image_train_val_lstm_scs_r3_p30s5l60_raw_100(path):
    return create_split_acc_vol_for_specific_model('testing_scs_r3', 'nn_lstm_scs_r3_p30s5l60%_raw_100', 'f418ba78428049e2ee2f41c930b9c5acb7ab863e')


def image_confusion_lstm_scs_r3_p30s5l60_raw_100(path):
    return create_confusion_for_specific_model('testing_scs_r3', 'nn_lstm_scs_r3_p30s5l60%_raw_100', 'f418ba78428049e2ee2f41c930b9c5acb7ab863e')


def image_train_val_cnn_1d_scs_r3_p30s5l60_raw_100(path):
    return create_split_acc_vol_for_specific_model('testing_scs_r3', 'nn_cnn_1d_scs_r3_p30s5l60%_raw_100', '3ce311e6f776a352e3bbc7d1e92aed57c96940fe')


def image_confusion_cnn_1d_scs_r3_p30s5l60_raw_100(path):
    return create_confusion_for_specific_model('testing_scs_r3', 'nn_cnn_1d_scs_r3_p30s5l60%_raw_100', '3ce311e6f776a352e3bbc7d1e92aed57c96940fe')


def image_silent_detection_sample1(path):
    mdb = metadata_db('audio/simpleCall/metadata.json')
    ppl = load_audio(mdb, 'simple_call_test', 'audio/simpleCall')
    file = ppl.files[0]
    y, sr = librosa.load(file.path(), sr=500000, duration=1)

    rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=1024)
    mse = rms ** 2
    mse = librosa.power_to_db(mse.squeeze(), ref=np.max, top_db=None)

    labels = filter_and_transform_labels(file.metadata.labels, 0, 1, sr)
    cmap = plt.cm.get_cmap('viridis', 2)
    silent_color, label_color = [cmap(i) for i in range(2)]

    silence, cut_off = get_silent_parts(y, labels)
    pad_time = .005
    pad = librosa.time_to_samples(pad_time, sr)
    silence[...] += [pad, -pad]
    silence = list(filter(lambda part: part[1] - part[0] > 10, silence))

    if len(silence) > 0:
        silence = filter_silent_parts(silence, filter_and_transform_labels(file.metadata.labels, 0, 1000, sr))

    fig, ax = plt.subplots(nrows=1, figsize=(10, 2.5))
    times = librosa.times_like(rms, sr=sr, hop_length=1024)
    ax.semilogy(times, mse, label='MSE')
    ax.hlines(cut_off, 0, max(times),colors='darkorange',linestyles="dashed", label='silent threshold')
    ax.set_yscale('linear')
    ax.set_xlim(0, max(times))

    for start, end in librosa.samples_to_time(silence, sr):
        rect = patches.Rectangle((start - pad_time,0), pad_time, -80, linewidth=0, edgecolor='r',facecolor='CornflowerBlue', alpha=.3)
        ax.add_patch(rect)
        rect = patches.Rectangle((start,0), end-start, -80, linewidth=0, edgecolor='r',facecolor=silent_color)
        ax.add_patch(rect)
        rect = patches.Rectangle((end,0), pad_time, -80, linewidth=0, edgecolor='r',facecolor='CornflowerBlue', alpha=.3)
        ax.add_patch(rect)

    for start, end in librosa.samples_to_time(labels, sr):
        rect = patches.Rectangle((start,0), end-start, -80, linewidth=0, edgecolor='r',facecolor=label_color)
        ax.add_patch(rect)

    ax.legend()
    return fig


# def image_lstm_model(path):
#     pattern = re.compile(r'<title>[^<]*</title>')
#     model_path = get_best_model_path_from_result(get_best_result_for_model('testing_sct_compressed', 'nn_lstm_'))
#     model = load_model(model_path)
#     dot = model_to_dot(model, show_shapes=True, show_layer_names=True, dpi=300)
#     dot.write(path, format='svg')
#     write_file(path, pattern.sub('', read_file(path)))


def image_compressed_spectorgram_b2(path):
    ppl = load_audio_and_db('simple_call_test').extract_label_parts(False)
    ppl.files = [file for file in ppl.files if file.metadata.duration > .2]

    spectrogram_file_list = ppl.create_spectrogram(sampling_rate=500000, width=100, window='Ham').run()
    shutil.copy(spectrogram_file_list.files[0].path(), str(path).replace('.svg', '_compressed.png'))

    # when 22ms = 100pixel, we get 4545,454545 pixel per sec
    spectrogram_file_list = ppl.create_spectrogram(sampling_rate=500000, x_pixels_per_sec=4545, window='Ham').run()
    shutil.copy(spectrogram_file_list.files[0].path(), str(path).replace('.svg', '.png'))


def image_moving_window(path):
    xpps = 2000
    ppl = create_sample_pipline()
    split_parts = ppl.split_into_parts(.030, .034, .8)
    spectrogram_file_list = ppl.create_spectrogram(sampling_rate=500000, height=129, x_pixels_per_sec=xpps, window='Ham').run()
    fig = plt.figure(figsize=(10, 2.5))
    ax = fig.add_subplot()
    cmap = plt.cm.get_cmap('viridis', 2)

    plt.imshow(mpimg.imread(spectrogram_file_list.files[0].path()))
    ticks = FuncFormatter(lambda x, pos: '{0:g}'.format(x / xpps))
    ax.xaxis.set_major_formatter(ticks)
    ax.set_ylabel('height in pixel')
    ax.set_xlabel('ms')

    for label in ppl.files[0].metadata.labels:
        label['start'] *= xpps
        label['end'] *= xpps
        rect = patches.Rectangle((label['start'], 0), label['end'] - label['start'], 129, linewidth=0, edgecolor=None, facecolor=cmap(1), alpha=.3)
        ax.add_patch(rect)

    label_rect = patches.Rectangle((20, 140), 10, 20, linewidth=1, edgecolor='black', facecolor=cmap(1), alpha=1)

    for file in split_parts.files:
        start = file.metadata.start * xpps
        rect = patches.Rectangle((start, 0), file.metadata.duration*xpps, 128, linewidth=2, edgecolor=cmap(0), fill=False, alpha=.8)
        ax.add_patch(rect)
        ax.text(start+25, -13, file.metadata.label, horizontalalignment='left', verticalalignment='top')

    ax.legend([label_rect, rect], ['annotated label', 'moving window'])

    return fig


def image_pipline_audio_waveform(path):
    ppl = create_sample_pipline()
    y, sr = librosa.load(ppl.files[0].path(), duration=.03)

    fig, ax = plt.subplots(figsize=(4,1))
    ax.label_outer()
    waveplot(y, sr=sr, ax=ax)
    ax.set(title='audio')
    ax.set_yticklabels([])
    return fig


def image_pipline_audio_spectrogram(path):
    ppl = create_sample_pipline().extract_label_parts()\
        .create_spectrogram(sampling_rate=500000, height=129, x_pixels_per_sec=4000, window='Ham').run()

    l = 4
    fig, ax = plt.subplots(ncols=l, figsize=(4,1))
    for i in range(0, l):
        ax[i].set(title=ppl.files[i].metadata.label)
        ax[i].imshow(mpimg.imread(ppl.files[i].path()))
        ax[i].set_axis_off()

    return fig


def create_syllable_plot(name):
    image = mpimg.imread(BSC_ROOT_DATA_FOLDER/'simple_call_test'/'audio'/ 'label_variable' / 'sr500000_xpps5000_h512_wHam'/ name)
    fig = plt.figure(figsize=(5*image.shape[0]/image.shape[1], 5))
    plt.imshow(image[:, :, 0], cmap='viridis')
    plt.axis('off')
    return fig


def image_syllable_b2(path):
    return create_syllable_plot('C_TO_Jt2_16_06_12_018_048562.51_B2.png')


def image_syllable_b3(path):
    return create_syllable_plot('C_TR_Jt7_16_07_07_037_115411.70_B3.png')


def image_syllable_b4(path):
    return create_syllable_plot('B4_C_TR_Jt1_16_06_02_108_000129.77_B4.png')


def image_syllable_vs(path):
    return create_syllable_plot('VS_Curu_B_Jto_16_07_04_006_000246.27_VS.png')


def image_syllable_vsv(path):
    return create_syllable_plot('VSV_C_TR_Jt1_16_06_02_109_000035.96_VSV.png')


def image_syllable_ups(path):
    return create_syllable_plot('T_Curu_TR_Jt7_16_06_23_023_000710.77_UPS.png')


FIGURES_DIR = Path('../image/generated')


def run(forced=None):
    FIGURES_DIR.mkdir(exist_ok=True)
    forced = {} if forced is None else set(forced)

    for f, fx in get_functions(__name__, 'image_'):
        name = f[6:]
        dest = FIGURES_DIR / f'{name}.svg'

        digest = sha1(inspect.getsource(fx).encode()).hexdigest()
        digest_path = Path(f'{dest}.sha1')

        if f in forced or not (digest_path.is_file() and digest == read_file(digest_path)):
            print(f'Generating {dest}...')

            plot = fx(dest)
            if plot:
                plot.savefig(dest, bbox_inches='tight', dpi=300)

            write_file(digest_path, digest)
        else:
            print(f'Skipped {dest}')
    print('Done')


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate diagrams for the report')
    parser.add_argument('forced', nargs='*', help='images that must be regenerated even when source is unchanged')
    args = parser.parse_args()
    run(**vars(args))
