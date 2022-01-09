import os
import time
import json
import argparse
import logging
from dotenv import load_dotenv
from datetime import datetime, date, timedelta

import pandas as pd

from tf_agents.environments import gym_wrapper  # wrap OpenAI gym
from tf_agents.environments import tf_py_environment  # gym to tf gym

from reinforcement import _version
from reinforcement.environment.envs import OptionEnvironment
from reinforcement.training.baseline import Baseline
from reinforcement.training.trainer import DQN
from reinforcement.utils.logging_utils import initialise_logger

project_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
env_file = os.path.join(project_path, "config.env")
hyperparameter_file = os.path.join(project_path, "hyperparameters.env")
load_dotenv(env_file)
load_dotenv(hyperparameter_file)


logger = logging.getLogger("root")


def main():

    # initialize logger
    global logger
    logger = initialise_logger(log_level=logging.INFO)
    logger.info(f"Experiment started at {datetime.now()}")

    start_time = time.time()

    option_settings = {
        "Option settings": "",
        "S0": float(os.getenv("S0")),
        "K": float(os.getenv("K")),
        "r": float(os.getenv("r")),
        "sigma": float(os.getenv("sigma")),
        "d": float(os.getenv("d")),
        "T": float(os.getenv("T")),
        "N": int(os.getenv("N")),
    }

    hyperparameter_settings = {
        "Hyperparameters": "",
        "learning_rate": float(os.getenv("learning_rate")),
        "replay_buffer_max_length": int(os.getenv("replay_buffer_max_length")),
        "batch_size": int(os.getenv("batch_size")),
        "num_iterations": int(os.getenv("num_iterations")),
        "num_eval_episodes": int(os.getenv("num_eval_episodes")),
        "collect_steps_per_iteration": int(os.getenv("collect_steps_per_iteration")),
        "final_eval_episodes": int(os.getenv("final_eval_episodes")),
    }

    # setting today'a date
    today_date = date.today()
    # calling baseline model evaluations
    baseline = Baseline(
        today_date,
        option_settings.get("S0"),
        option_settings.get("K"),
        option_settings.get("r"),
        option_settings.get("sigma"),
        option_settings.get("d"),
        option_settings.get("T"),
    )
    df = baseline.baseline_model()

    train_env_gym = OptionEnvironment(
        option_settings.get("S0"),
        option_settings.get("K"),
        option_settings.get("r"),
        option_settings.get("sigma"),
        option_settings.get("T"),
        option_settings.get("N"),
    )
    eval_env_gym = OptionEnvironment(
        option_settings.get("S0"),
        option_settings.get("K"),
        option_settings.get("r"),
        option_settings.get("sigma"),
        option_settings.get("T"),
        option_settings.get("N"),
    )

    train_env_wrap = gym_wrapper.GymWrapper(train_env_gym)
    eval_env_wrap = gym_wrapper.GymWrapper(eval_env_gym)

    train_env = tf_py_environment.TFPyEnvironment(train_env_wrap)
    eval_env = tf_py_environment.TFPyEnvironment(eval_env_wrap)

    # defining network
    dqn = DQN(
        train_env,
        eval_env,
        hyperparameter_settings.get("learning_rate"),
        hyperparameter_settings.get("replay_buffer_max_length"),
        hyperparameter_settings.get("batch_size"),
        hyperparameter_settings.get("num_iterations"),
        hyperparameter_settings.get("num_eval_episodes"),
        hyperparameter_settings.get("collect_steps_per_iteration"),
    )
    trained_policy = dqn.train()

    npv = dqn.compute_avg_return(
        eval_env,
        trained_policy,
        num_episodes=hyperparameter_settings.get("collect_steps_per_iteration"),
    )
    df["ReinforcementAgent"] = npv
    df = pd.DataFrame.from_dict(df, orient="index")
    df.columns = ["Price"]

    name = os.path.join(
        f"experiments/{os.getenv('EXPERIMENT_NO')}/history", "pricing.csv"
    )
    df.to_csv(name, index=True, encoding="utf-8")

    logger.info(f"The experiment was completed")

    training_time = time.time() - start_time
    # saving configurations for the experiment
    model_metadata = {
        "version": _version.__version__,
        "major_version": _version.MAJOR_VERSION,
        "minor_version": _version.MINOR_VERSION,
        "patch_version": _version.PATCH_VERSION,
        "algorithm": "DQN",
        "model_date": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "training_time": str(training_time / 60) + " minutes",
    }

    model_metadata = dict(model_metadata, **option_settings, **hyperparameter_settings)
    with open(
        f"experiments/{os.getenv('EXPERIMENT_NO')}/metadata.json", "w", encoding="utf-8"
    ) as f:
        json.dump(model_metadata, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        "--experiment_no",
        type=str,
        required=True,
        help="Please provide the experiment number",
    )

    args = parser.parse_args()

    # set experiment number in environment variable
    os.environ["EXPERIMENT_NO"] = args.experiment_no

    if not os.path.exists(f"experiments/{os.getenv('EXPERIMENT_NO')}"):
        os.makedirs(f"experiments/{os.getenv('EXPERIMENT_NO')}")

    if not os.path.exists(f"experiments/{os.getenv('EXPERIMENT_NO')}/history"):
        os.makedirs(f"experiments/{os.getenv('EXPERIMENT_NO')}/history")

    if not os.path.exists(f"experiments/{os.getenv('EXPERIMENT_NO')}/policy"):
        os.makedirs(f"experiments/{os.getenv('EXPERIMENT_NO')}/policy")

    if not os.path.exists(f"experiments/{os.getenv('EXPERIMENT_NO')}/tensorboard"):
        os.makedirs(f"experiments/{os.getenv('EXPERIMENT_NO')}/tensorboard")

    main()
