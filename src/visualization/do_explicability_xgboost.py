from utils import directory_path, define_h2o_dataframe, load_pickle, dump_pickle
import shap
import pandas as pd
from matplotlib import pyplot as plt


def shap_visualisation_xgboost(version:str, point:int, do_train=True):
    test = pd.read_csv(f"{directory_path}data/processed/application_test.csv")
    model = load_pickle(f"{directory_path}/models/{version}/xgboostclassifier")

    # understanding model predictions
    X = test
    if do_train:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        dump_pickle(f"{directory_path}visualisation/xgboost/explainer", explainer)
        dump_pickle(f"{directory_path}visualisation/xgboost/shap_values", shap_values)
    else:
        shap_values = load_pickle(f"{directory_path}visualisation/xgboost/shap_values")
        explainer = load_pickle(f"{directory_path}visualisation/xgboost/explainer")

    image_name = f"{directory_path}visualisation/xgboost/plots/plot"
    plots_shap(image_name, shap_values, explainer, X, point)




def plots_shap(image_name, shap_values, explainer, X, point):
    # Visualize explanations for a specific point of your data set,
    save_plot_specific_point(f'{image_name}_specific_point_{point}.png', X,
                             explainer.expected_value, shap_values, point)

    save_plot_local_waterfall(f'{image_name}_waterfall_point_{point}.png', X,
                              explainer.expected_value, shap_values, point)

    # Visualize explanations for all points of your data set at once,
    save_plot_all_points(f'{image_name}_allpoint.png', X,
                         explainer.expected_value, shap_values)

    # Visualize a summary plot for each class on the whole dataset.
    save_plot_all_featureimportance(f'{image_name}_summary.png', X, shap_values)
    save_plot_summary(f'{image_name}_summary_bar.png', X, shap_values)


def save_plot_specific_point(image_name, set, expected_value, shap_values, row):
    point = shap.force_plot(expected_value, shap_values[row], features=set.iloc[-1:],
                                feature_names=list(set.columns),
                                matplotlib=True, show=False
                        )
    plt.savefig(image_name,  bbox_inches='tight')
    plt.close()

def save_plot_local_waterfall(image_name, set, expected_value, shap_values, row):
    fig=plt.gcf()
    shap.plots._waterfall.waterfall_legacy(expected_value, shap_values[row],
                                                   feature_names=set.columns,
                                                   )
    fig.savefig(image_name,  bbox_inches='tight')

def save_plot_all_points(image_name, set, expected_value, shap_values):
    shap_plot = shap.force_plot(expected_value,
                                shap_values[-1:], features=set.iloc[-1:],
                                feature_names=list(set.columns),
                                matplotlib=True, show=False)

    plt.savefig(image_name,  bbox_inches='tight')
    plt.close()

def save_plot_all_featureimportance(image_name, set, shap_values):
    fig = plt.gcf()
    shap.summary_plot(shap_values, set.values, class_names=[0,1], feature_names=set.columns
                             )
    fig.savefig(image_name,  bbox_inches='tight')


def save_plot_summary(image_name, set, shap_values):
    fig = plt.gcf()
    shap.summary_plot(shap_values, set.values, plot_type="bar",
                             class_names=[0,1], feature_names=set.columns)
    fig.savefig(image_name,  bbox_inches='tight')
