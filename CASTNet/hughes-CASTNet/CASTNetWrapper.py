import numpy as np
import pandas as pd
from pandas import IndexSlice as idx
import os
import tensorflow as tf
import pickle
import hughes_castnet_model

class CASTNetWrapper:

    def __init__(self, conf):
        """
        Initializes the CASTNetWrapper class.
        """
        self.conf = conf
        self.sess = tf.compat.v1.Session()  
        self.castnet_model = hughes_castnet_model.CASTNet(self.sess, conf)  # initialize CASTNet model
    
    def train(self, train_svi_local, train_svi_global, train_static, train_sample_indices, train_dist, train_y,
              valid_svi_local, valid_svi_global, valid_static, valid_sample_indices, valid_dist, valid_y,
              test_svi_local, test_svi_global, test_static, test_sample_indices, test_dist, test_y, batches=True):
        """
        Trains  CASTNet model

        Args:
        batches: Whether to use mini-batches during training (default=True).
        """
        self.castnet_model.train(train_svi_local, train_svi_global, train_static, train_sample_indices, train_dist, train_y,
                                 valid_svi_local, valid_svi_global, valid_static, valid_sample_indices, valid_dist, valid_y,
                                 test_svi_local, test_svi_global, test_static, test_sample_indices, test_dist, test_y, batches)

    def predict(self, svi_data_local, svi_data_global, static_data, sample_indices, dist_data):
        """
        Makes predictions using the trained CASTNet model
        """
        return self.castnet_model.predict(svi_data_local, svi_data_global, static_data, sample_indices, dist_data)
    
    def save(self, save_path):
        self.castnet_model.save(save_path)
    
    def load(self, load_path):
        self.castnet_model.load(load_path)
    
    def save_results(cook, test_times):
        """
        @param cook (bool)
        @param test_times would be [1,2] for 2 test times"""
        data_dir = os.environ.get('DATA_DIR', '/Users/jyontika/Desktop/opioid-overdose-models/CASTNet/hughes-CASTNet/')
        
        if cook:
            results_folder = 'Chicago'
        else:
            results_folder = 'MA'

        results = []
        for test_time in test_times:
            results_path = os.path.join(data_dir, f'Results/{results_folder}_lead_time{test_time}.pkl')

            with open(results_path, 'rb') as file:
                result = pickle.load(file, encoding='bytes')
                results.append(result)

        all_predictions = [result['preds'] for result in results]
        prediction_matrices = [np.array(predictions) for predictions in all_predictions]
        prediction_matrices = [np.maximum(matrix, 0) for matrix in prediction_matrices]
        prediction_leads = [matrix[-1, :] for matrix in prediction_matrices]

        locations_path = os.path.join(data_dir, f'Data/{results_folder}/locations.txt')
        locations = []
        with open(locations_path, 'rb') as file:
            for line in file:
                line = line.rstrip().decode("utf-8").split("\t")
                locations.append(line[1])

        df_predictions = []
        for idx, prediction_lead in enumerate(prediction_leads):
            year = str(2021 + test_times[idx]) if cook else str(2019 + test_times[idx])
            df_prediction = pd.DataFrame({
                'geoid': locations,
                'prediction': prediction_lead,
                'year': year
            })
            df_predictions.append(df_prediction)

        combined_df = pd.concat(df_predictions)

        csv_filename = f"Results/{'cook-county' if cook else 'MA'}-predictions.csv"
        csv_filepath = os.path.join(data_dir, csv_filename)

        combined_df.to_csv(csv_filepath, index=False)


        
    def load_results_and_locations(self, Cook):
        """
        Loads CASTNet results and location information.

        @param Cook (bool): Whether to load results for Cook County (True) or MA (False).

        Returns:
            CN_results: df containing CASTNet results.
            CN_locations: list of location information.
        """
        data_dir = os.environ.get('DATA_DIR', '/Users/jyontika/Desktop/opioid-overdose-models/CASTNet/hughes-CASTNet/')
        if Cook:
            results_path = os.path.join(data_dir, 'Results/cook-county-predictions.csv') 
            locations_path = os.path.join(data_dir, 'Data/Chicago/locations.txt')
        else:
            results_path = os.path.join(data_dir, 'Results/MA-predictions.csv') 
            locations_path = os.path.join(data_dir, 'Data/MA/locations.txt')
        
        CN_results = pd.read_csv(results_path)
        CN_results['geoid'] = CN_results['geoid'].astype(str)
        
        CN_locations = []
        with open(locations_path, 'rb') as file:
            for line in file:
                line = line.rstrip().decode("utf-8").split("\t")
                CN_locations.append(line[1])
        
        return CN_results, CN_locations

    
    def close_session(self):
        """
        Closes the TensorFlow session.
        """
        self.sess.close()
