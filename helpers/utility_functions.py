import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sc
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std


def borda_rank(fi_lists):
    """
    fi_lists: [[], [], ...] list of lists per algorithm feature importances
    """
    assert len(fi_lists) >= 2
    #lists_lengths = [len(i) for i in fi_lists]
    #assert all([lists_lengths[0] == i for i in lists_lengths])
    ranks = _create_list_of_rank_dicts(fi_lists)
    return  _total_ranks(ranks)


def sorted_borda_rank(fi_lists):
    total_score = borda_rank(fi_lists)
    result = [(k,v) for k,v in sorted(total_score.items(), key=lambda item: item[1], reverse=True)]
    return result


def _total_ranks(ranks):
    total_score = {}
    for ranka in ranks:
        for key, val in ranka.items():
            total_score[key] =  total_score.get(key, 0) + val
    return total_score


def _create_list_of_rank_dicts(fi_lists):
    ranks = []
    for imm in fi_lists:
        copied_imm = imm[:]
        copied_imm.reverse()
        rank_d = {}
        for i, el in enumerate(copied_imm):
            rank_d[el] = i
        ranks.append(rank_d)
    return ranks


# Read dataset from .mtx file
def read_dataset(dataname):
    fnc = sc.mmread(dataname)
    B = fnc.todense()
    data = pd.DataFrame(B)
    data = data.transpose()
    return(data)


def show_cross_val_scores(alg_name_scores, alg_name='Xgboost'):
    print(f"{alg_name} scores")
    print("%0.3f accuracy with a standard deviation of %0.3f" % (alg_name_scores['test_accuracy'].mean(), alg_name_scores['test_accuracy'].std()))
    print("%0.3f f1 score with a standard deviation of %0.3f" % (alg_name_scores['test_f1'].mean(), alg_name_scores['test_f1'].std()))
    print("%0.3f roc_auc with a standard deviation of %0.3f" % (alg_name_scores['test_roc_auc'].mean(), alg_name_scores['test_roc_auc'].std()))

    
def create_box_plot(vals, labels, pdf_name, title='XgBoost accuracy'):
    fig, ax = plt.subplots()
    datasets = ['pre-processed', 'consensus']
    fig.subplots_adjust(left=0.2, right=1.3, bottom=0.2, top=1.5,
                        hspace=0.05, wspace=0.05)
    box = ax.boxplot(vals, patch_artist=True)
    ax.set_title(title, fontsize=28)
    ax.set_xticklabels(labels, rotation='horizontal', fontsize=24)
    
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.savefig(pdf_name, dpi=150, bbox_inches='tight')
    plt.show()



# Delete columns with more than specified number of zero values
def reduce_columns(data, zero_num):
    cols = []
    for i in range(len(data.columns)):
        if data[i].value_counts()[0] < zero_num:
            cols.append(i)
    return cols


# delete columns with more than specified number of zero values (percentage method -> more accurate)
def reduce_columns_percentage(data, zero_percentage):
    cols = []
    zero_num = int(len(data.index) * zero_percentage)
    for i in range(len(data.columns)):
        if data[i].value_counts()[0] < zero_num:
            cols.append(i)
    return cols


# Order columns by the number of zeros that they have (lowest comes first)
def order_columns_by_zero(data):
    cols = {}
    for i in range(len(data.columns)):
        cols[i] = data[i].value_counts()[0]
    return cols


# Dataset normalization with min-max scaler
def data_normalization(dataset):
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(dataset)
    normalized_data = pd.DataFrame(x_scaled)
    return normalized_data

def data_tag_separator(dataset_tags):
    tags = dataset_tags['result'].to_frame()
    dataset = dataset_tags.drop(['result'], axis=1)
    return dataset, tags