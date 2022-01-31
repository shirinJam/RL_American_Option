# Reinforcement learning application to option pricing for American options

## Pre-requisites

- Python >= 3.8 version must be installed 

## Preparation of local setup

Please follow the below steps to set up the Reinforcement learning Python code locally.

1.	Copy the source code to a local directory.
2.	Open Terminal or CMD on the local machine, and navigate to the folder directory of the code, i.e., RL_American_Option_Code. This folder contains the src folder, requirements file and  Readme file. 
3.	Given the system has python distribution installed along with pip, please execute the below command to install the dependencies to run the code:

    ```
    pip install -r requirements.txt
    ```

## Evaluating the Trained RL policies

### Saved policies

Following policies were saved:
1. Experiment 1 – DQN Agent (Put Option) 
        - Name: exp_1_dqn_put
2. Experiment 1 – REINFORCE Agent (Put Option)
        - Name: exp_1_reinforce_put
3. Experiment 1 – DQN Agent (Call Option) 
        - Name: exp_1_dqn_call
4. Experiment 1 – REINFORCE Agent (Call Option)
        - Name: exp_1_reinforce_call
5. Experiment 2 – Best hyperparameter tuned DQN Agent (Put Option)
        - Name: exp_2_dqn_put
6. Experiment 3 – SABR based DQN Agent (Put Option)
        - Name: exp_3_dqn_sabr_put

The name corresponds to the folder name where the experiment and its respective policies are saved in the experiments folder.

### Evaluate the saved policies

1.	Open Terminal or CMD on the local machine, and navigate to the src folder inside the RL_American_Option_Code folder.
2.	To test and evaluate the trained policy for the DQN agent in the 1st experiment specifically to put option, run the following command:

    ```
    python3 -m reinforcement.train_model --experiment_no exp_1_dqn_put –-option_type put -–flow evaluate
    ```

    These parameters passed in the command line helps the Python code to understand which experiment to use, which option type to use and if it needs to do training or just evaluation. 
3.	Similarly, to run other experiments follow the same command as below:

    ```
    python3 -m reinforcement.train_model –-experiment_no <EXPERIMENT_NAME> –-option_type <OPTION_TYPE_BASED_ON_THE_EXPERIMENT> -–flow evaluate
    ```

	In order to run other experiments with call option, then use option_type as call.
4.	Since the default evaluation path is set to 2000, the evaluation takes time (15 mins). Once the evaluation is complete, the results are saved in pricing_evaluation.csv along with the benchmark model results. The results can be viewed under the history folder in the respective experiments folder.
 
    Repeat steps 1 to 4 for other experiments by changing the parameters in the command line: `<EXPERIMENT_NAME>` and `<OPTION_TYPE_BASED_ON_THE_EXPERIMENT>`

## Training a New Agent Policy


As the previous section illustrated to navigate through the code and evaluate the existing trained policies, this section focuses on using the Python code to train a new policy based on the set option settings and hyperparameters as per the requirements. In order to train a new reinforcement learning agent policy, follow the below steps:

1.	Navigate to the RL_American_Option_Code -> scr -> reinforcement folder. Open and edit the `config.env` file as per the requirement to set the option settings for the experiment
2.  Once the option settings for the experiment are set using the `config.env` file, save the file. Next, open and edit the `hyperparameters.env` file as per requirements.
    **Note:** Here, `num_iterations` is equal to the number of training episodes
3.  Once these hyperparameters are set and saved, a new experiment can be initiated through Terminal or CMD using the below command
    
    ```
    python3 -m reinforcement.train_model --experiment_no <EXPERIMENT_NO> --policy_based <POLICY_FLAG> --option_type <OPTION_TYPE>
    ```

    Here, `<EXPERIMENT_NO>` can be set to any name or number, e.g. new_exp_1. Based on the set entry, the Code will create a new folder under RL_American_Option_Code -> scr -> experiments to place all the related files and trained policy under that experiment folder. It is mandatory to provide `<EXPERIMENT_NO>` to initiate training.
    
    `<POLICY_FLAG>` can be set to yes or no. If it is set to yes, training using the REINFORCE agent is initiated. If it is set to no, training using the DQN agent is initiated. The default setting is set to no.
    
    `<OPTION_TYPE>` can be set to put or call as per the requirements. The default setting is set to put.
    **Note:** There is no need to pass --flow parameter, as by default, it is set to training and evaluation.

4.	Based on these parameters passed in the Command line, and through the `config.env` and `hyperparameters.env` file, the training and evaluation of the RL agent is initiated. The training loss and evaluated average return, i.e., the option price based on `num_eval_episodes` in `hyperparameters.env`, are printed in the command line while the RL agent is training.
5.	Once the training is completed, the results are stored in the `experiments` folder.
6.	The stored results consist of the trained RL agent policy in the greedy_policy folder. The history folder stores the `loss_history.cs`v and `return_history.csv`, which are the same results printed while training the RL agent along with the episodes. 
The history folder also stores `pricing.csv`, which consists of the evaluated benchmark values based on the passed parameters, and the final evaluation of the performance of the trained RL agent policy on `final_eval_eposides` paths. 
7.	The code also captures all option settings and hyperparameters set for the experiments and are stored in a `metadata.json` JSON format file. This allows to easily keep track of the experiments in the future by knowing the passed parameters, code version and the results obtained. It also stores the time taken by the experiment and the time when the experiment was initiated. 
