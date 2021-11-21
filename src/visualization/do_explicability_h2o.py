from utils import get_model
from models.predict_model import preprocess_testset
from utils import directory_path, define_h2o_dataframe, load_pickle, dump_pickle
import shap
import pandas as pd
from matplotlib import pyplot as plt
import h2o

def shap_visualisation_h2o(version: str, point:int):
    """This function build shap visualisation on h2o models.

    :param version: model version
    :type version: str
    :param point: a specific row of the dataset
    :type point: int
    """
    test = pd.read_csv(f"{directory_path}data/processed/application_test.csv")
    test_h2o = h2o.H2OFrame(test)
    model = get_model("xgboost", version)

    image_name = f"{directory_path}visualisation/h2o_xgboost/plots/plot"

    # H2o integrating Shap functionnalities
    # Visualize explanations for a specific point of your data set,
    save_plot_specificpoint_h2o(model, f"{image_name}_specific_point_{point}.png", test_h2o, point)
    # Visualize a summary plot for each class on the whole dataset.
    save_plot_allpoint_h2o(model,f"{image_name}_summary.png", test_h2o)


def save_plot_allpoint_h2o(model,image_name: str, test_h2o: h2o.H2OFrame):
    """This function do shap summary plot and store
    the plot in a file

    :param model: H2O xgboost model
    :type model:
    :param image_name: path to the image
    :type image_name: str
    :param test_h2o: test set
    :type test_h2o: h2o.H2OFrame
    """
    perf = model.shap_summary_plot(test_h2o)
    plt.savefig(image_name)
    plt.close()

def save_plot_specificpoint_h2o(model,image_name:str , test_h2o: h2o.H2OFrame, row_index: int):
    """This function do visualisation plot of a specific point and
    store the result.

    :param model: H2O xgboost model
    :type model:
    :param image_name: path to the image
    :type image_name: str
    :param test_h2o: test set
    :type test_h2o: h2o.H2OFrame
    :param row_index: row index
    :type row_index: int
    """
    perf = model.shap_explain_row_plot(test_h2o, row_index=row_index)
    plt.savefig(image_name)
    plt.close()
