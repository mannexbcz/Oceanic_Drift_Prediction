import os
import yaml

CONFIG_PATH = '../configs/config_data_driven/NOAA'
base_config_file = 'configs_data_driven/NOAA/config_NOAA.yml'

if __name__ == "__main__":

    with open(base_config_file, 'r') as f:
            config = yaml.safe_load(f)
    config['chkpt_folder'] = '../checkpoints/NOAA/gridsearch'
    i = 0
    for channel1 in [16,32,64]:
        for channel2 in [16,32,64]:
            for hidden1 in [128,256]:
                for hidden2 in [64,128]:
                    for hidden3 in [128,256]:
                        for hidden4 in [64,128]:
                                    config['ch1'] = channel1
                                    config['ch2'] = channel2
                                    config['hidden1'] = hidden1
                                    config['hidden2'] = hidden2
                                    config['hidden3'] = hidden3
                                    config['hidden4'] = hidden4
                                    data_file = 'config_' + str(i) + '.yml'
                                    with open(os.path.join(CONFIG_PATH,data_file), 'w') as f:
                                        yaml.dump(config,f)
                                    i = i+1