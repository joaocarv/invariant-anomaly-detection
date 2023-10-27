from omegaconf import OmegaConf
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pytorch_lightning import Callback
from train import *
import string
import random
import shutil
log_folder = "logs"



'''
class MetricTracker(Callback):

    def __init__(self):
        self.metrics = []

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        #print(pl_module.image_metrics['AUROC']())
        #self.metrics.append([pl_module.image_metrics['image_AUROC'], pl_module.image_metrics['image_F1Score']])
        #self.metrics.append([trainer.logged_metrics['image_AUROC'], trainer.logged_metrics['image_F1Score']])
        #print(trainer.callbacks)
        print(trainer.callbacks[-1].best_model_score)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        #self.on_validation_end(trainer, pl_module)
        #print(trainer.callbacks)
        print(trainer.callbacks[-1].best_model_score)
'''




# hyperparameter file consists of key value pairs, where keys are the names of hyperparameters, values are list of
# search ranges e.g.
# clamp_alpha:
# - 0.5
# - 1
def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to a model config template")
    parser.add_argument("--dataset", type=str, required=True, help="Path to a dataset config template")
    parser.add_argument("--hyperparameter", type=str, required=True,
                        help="Path to hyperparameter space")
    args = parser.parse_args()
    return args


def hyperparameter_search():
    config_template, hyperparameter = _load_files()
    # generating configs for each model
    configs = _generate_configs_gridwise(deepcopy(config_template), hyperparameter)
    test_results = []
    random_string = ''.join(random.choices(string.ascii_uppercase +
                                           string.digits, k=6))
    file_name = config_template.model.name + "_" + config_template.dataset.name + "_" + random_string + ".txt"
    file_path = os.path.join(log_folder, file_name)
    os.makedirs(log_folder, exist_ok=True)
    with open(file_path, "a+") as myfile:
        myfile.write("-------Experiments with "+config_template.model.name+" on "+config_template.dataset.name+" -------------------------\n")
    for config in configs:
        with open(file_path, "a+") as myfile:
            myfile.write("---------------------Current Config---------------------\n")
            myfile.write(str(config.model) + "\n")
            myfile.write("---------------------End---------------------\n")
        true_config = deepcopy(config)  # the anomalib module modifies the config file a bit
        # true_config.logging.logger = ["csv"]
        true_config.visualization.save_images = False
        true_config.visualization.log_images = False
        true_config.visualization.show_images = False
        true_config.trainer.enable_progress_bar = False
        result_dir = true_config.project.path

        seed_everything_at_once(config_template.project.seed if config_template.project.seed else 42)
        auroc = _train_and_test_and_get_result(true_config)
        test_results.append(auroc)
        # remove checkpoints and visualized images of hyperparameter search
        best_config = _print_best_config(test_results, configs, file_path)
        shutil.rmtree(result_dir, ignore_errors=True)

    # Train and Test Again with Best Config
    seed_everything_at_once(config_template.project.seed if config_template.project.seed else 42)
    true_best_config = deepcopy(best_config)
    _train_and_test_and_get_result(true_best_config)


def _print_best_config(test_results, configs, file_path):
    max_result = max(test_results)
    index = test_results.index(max_result)
    best_config = configs[index]
    with open(file_path, "a+") as myfile:
        myfile.write("---------------------The Best Config---------------------\n")
        myfile.write(str(test_results) + "\n")
        myfile.write(str(max_result) + "\n")
        myfile.write(str(best_config.model) + "\n")
        myfile.write("---------------------End---------------------\n")
    return best_config


def _train_and_test_and_get_result(config):
    trainer, _, _ = train_and_test(config)
    mdck_cb = None
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint) and cb.best_model_score is not None:
            mdck_cb = cb
            break
    return mdck_cb.best_model_score


def _load_files():
    args = get_args()
    config_template = OmegaConf.load(args.model)
    config_template.dataset = OmegaConf.load(args.dataset)
    hyperparameter = OmegaConf.load(args.hyperparameter)
    return config_template, hyperparameter


# we do grid hyperparameter search
def _generate_configs_gridwise(config_template, model_hyperparameter):
    hyperparameter_dict = OmegaConf.to_container(model_hyperparameter, resolve=True)
    keys = list(hyperparameter_dict.keys())
    # generate the cartesian product [value1] x [value2] x ....
    config_grid = _generate_grid(keys, hyperparameter_dict)
    res = []
    for config in config_grid:
        tmp = deepcopy(config_template)
        for key, value in config.items():  # all hyperparameters should be under model section
            tmp['model'][key] = value
        res.append(tmp)
    return res


def _generate_grid(keys, hyperparameter_dict):
    if len(keys) == 1:
        _, value_list = _get_curr_key_value(hyperparameter_dict, keys)
        configs = []
        for value in value_list:
            configs.append({keys[0]: value})
        return configs
    else:
        key, value_list = _get_curr_key_value(hyperparameter_dict, keys)
        prev_configs = _generate_grid(keys[1:], hyperparameter_dict)
        configs = []
        for value in value_list:
            tmp = deepcopy(prev_configs)
            _grid_step(tmp, key, value)
            configs += tmp
        del prev_configs
        return configs


def _get_curr_key_value(hyperparameter_dict, keys):
    key = keys[0]
    value_list = hyperparameter_dict[key]
    return key, value_list


def _grid_step(config_list, key, value):
    for config in config_list:
        config[key] = value


if __name__ == "__main__":
    hyperparameter_search()
