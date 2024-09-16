

project_names = ['appceleratorstudio', 'aptanastudio', 'bamboo', 'clover', 
                'datamanagement', 'duracloud', 'jirasoftware', 'mesos', 
                'moodle', 'mule', 'mulestudio', 'springxd', 
                'talenddataquality', 'talendesb', 'titanium', 'usergrid']
preprocess_types = ['BoW', 'doc2vec']

for preprocess_type in preprocess_types:  
    for project_name in project_names:
        result_directory = 'results/' + preprocess_type
        result_file = result_directory + '/' + project_name + '.txt'
        number_of_scores = 8

        with open(result_file, 'r') as f:
            lines = f.readlines()
            
        scores = []
        for line in lines:
            array = line.split()
            if len(array) > 0:
                try:
                    float_value = float(array[-1])
                    scores.append(float_value)
                except ValueError:
                    continue

        import numpy as np
        scores = np.array(scores)
        scores = scores.reshape(-1, number_of_scores)

        scores = scores.tolist()

        for score_line in scores:
            output_file = 'extract_results/' + preprocess_type + '/' + project_name + '_extracted.txt'
            with open(output_file, 'w') as out_f:
                for score_line in scores:
                    out_f.write('\t'.join(map(str, score_line)) + '\n')