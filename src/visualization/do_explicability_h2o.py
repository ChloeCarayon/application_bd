from utils import get_model
from models.predict_model import preprocess_testset
from utils import directory_path, define_h2o_dataframe, load_pickle, dump_pickle
import shap
import pandas as pd
from matplotlib import pyplot as plt
import h2o

def shap_visualisation_h2o(version: str, point:int):
    label = 'TARGET'

    test = pd.read_csv(f"{directory_path}data/processed/application_test.csv")
    test_h2o = h2o.H2OFrame(test)

    model = get_model("xgboost", version)

    image_name = f"{directory_path}visualisation/h2o_xgboost/plots/plot"

    # H2o integrating Shap functionnalities

    # Visualize explanations for a specific point of your data set,
    save_plot_specificpoint_h2o(model, f"{image_name}_specific_point_{point}.png", test_h2o, point)

    # Visualize a summary plot for each class on the whole dataset.
    save_plot_allpoint_h2o(model,f"{image_name}_summary.png", test_h2o)

    image_name_kernel = f"{directory_path}visualisation/h2o_xgboost/plots/kernel/plot"

    #test_set = test[:100]
    #explainer = shap.KernelExplainer(lambda x : h2opredict(x, label, list(test.columns), model), test_set)
    #shap_values = explainer.shap_values(test_set)
    #plots_shap(image_name_kernel, shap_values, explainer, test_set.drop(columns=label), point)



def transform_h2o_contribute_to_shape(test:pd.DataFrame, label: str):
    contributions_df = pd.read_csv(f"{directory_path}visualisation/csv_files/contribution.csv")
    # shap values are calculated for all features
    shap_values = contributions_df.drop(columns=['BiasTerm']).values
    # expected values is the last returned column
    expected_values = contributions_df['BiasTerm'].values

    X_values = test.drop(columns=[label])

    return shap_values, expected_values, X_values


def h2opredict(nf, label, feature_names, model): #nf is numpy array
    df = pd.DataFrame(nf, columns = feature_names) #numpy to pandas
    df[label] = 0
    hf = h2o.H2OFrame(df) #pandas to h2o frame
    predictions = model.predict(test_data = hf)
    predictions_pd = predictions[predictions.columns[1:]].as_data_frame() #h2o frame to pandas
    return predictions_pd.values


def h2omodel_prediction(model_type: str, version: str):
    model = get_model(model_type, version)
    df_to_predict = preprocess_testset()
    predictions_h2o = model.predict(df_to_predict)
    predictions_pd = predictions_h2o[predictions_h2o.columns[1:]].as_data_frame()
    return predictions_pd.values


def save_plot_allpoint_h2o(model,image_name, test_h2o):
    perf = model.shap_summary_plot(test_h2o)
    plt.savefig(image_name)
    plt.close()

def save_plot_specificpoint_h2o(model,image_name, test_h2o, row_index):
    perf = model.shap_explain_row_plot(test_h2o, row_index=row_index)
    plt.savefig(image_name)
    plt.close()

def save_plot_summary_h2o(model,image_name, test_h2o):
    perf = model.shap_summary_plot(test_h2o, plot_type="bar")
    plt.savefig(image_name)
    plt.close()



