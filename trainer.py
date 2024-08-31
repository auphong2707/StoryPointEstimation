import itertools
import json
from IPython.display import clear_output
from sklearn.model_selection import cross_val_score
import os

class GridSearchCVTrainer:
    __directory = 'trainer_checkpoint_data/'
    def __init__(self, name, model, param_grid, cv=5, n_jobs=-1):
        self.name = name
        
        self.model = model
        self.cv = cv
        self.n_jobs = n_jobs
        
        self.__param_combinations = self.__get_param_combinations(param_grid)
        self.__last_combination = -1
        
        self.best_params_ = None
        self.best_score_ = None
        
    def __get_param_combinations(self, param_grid):
        if(type(param_grid) == dict):
            keys, values = zip(*param_grid.items())
            combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            return combinations
        elif(type(param_grid) == list):
            combinations = []
            for pg in param_grid:
                if(type(pg) == dict):
                    keys, values = zip(*pg.items())
                    combinations.extend([dict(zip(keys, v)) for v in itertools.product(*values)])
                else:
                    raise ValueError("param_grid must be a dictionary or a list of dictionaries.")
            return combinations
        else:
            raise ValueError("param_grid must be a dictionary or a list of dictionaries.")
        
    def fit(self, X, y):
        for i in range(self.__last_combination + 1, len(self.__param_combinations)):
            clear_output(wait=True)
            print(f"Training combination {i + 1}/{len(self.__param_combinations)}")
            params = self.__param_combinations[i]
            
            model = self.model
            model.set_params(**params)
            
            score = cross_val_score(model, X, y, cv=self.cv, n_jobs=self.n_jobs, scoring='neg_mean_squared_error').mean()
            if(self.best_score_ == None or score > self.best_score_):
                self.best_score_ = score
                self.best_params_ = params
            self.__last_combination = i
            self.save_to_json(self.__directory + self.name.lower() + '_checkpoint.json')
            
    def to_dict(self):
        return {
            'last_combination': self.__last_combination,
            'best_params': self.best_params_,
            'best_score': self.best_score_
        }

    def save_to_json(self, filename):
        # Ensure the directory exists
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save data to the JSON file
        data = self.to_dict()
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    
    def load_if_exists(self):
        filename = self.__directory + self.name.lower() + '_checkpoint.json'
        if os.path.exists(filename):
            with open(filename, 'r') as json_file:
                data = json.load(json_file)
            
            self.__last_combination = data['last_combination']
            self.best_params_ = data['best_params']
            self.best_score_ = data['best_score']
        else:
            print("There is no checkpoint file for this model.")
            
    @property
    def best_estimator_(self):
        model = self.model
        model.set_params(**self.best_params_)
        return model