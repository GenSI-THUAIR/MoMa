import torch
import numpy as np
import cvxpy as cp
import json
import pickle
import pandas as pd


epsilon = 1e-6
lambda_reg = 0
N = 18 # change to the number of modules
T = 1
QUOTA = 800

module_names = ['mp_eform','mp_bandgap','n_Seebeck','n_avg_eff_mass','n_th_cond','p_Seebeck','p_avg_eff_mass','p_e_cond','n_e_cond','p_th_cond','mp_gvrh','castelli_eform','jarvis_gvrh','jarvis_bandgap','jarvis_eform','jarvis_kvrh','mp_kvrh','jarvis_dielectric_opt']

data_names = ['piezoelectric_tensor','expt_eform','mp_poly_total','mp_poisson_ratio','jarvis_2d_dielectric_opt','jarvis_2d_eform','jarvis_2d_exfoliation','jarvis_2d_gap_opt','mp_elastic_anisotropy','mp_eps_electronic','mp_eps_total','mp_phonons','mp_poly_electronic','expt_bandgap','jarvis_3d_eps_tbmbj','jarvis_3d_gap_tbmbj','mp_dielectric']
splits = ['0','1','2','3','4']

K = 18  # Set to an integer if you want to limit the number of modules

optimized_obj_csv_path = "optimized_obj_csv_path"
optimized_obj_results = []

# define base paths
labels_base_path = './outputs/labels'
predictions_base_path = './outputs/predictions'

for split in splits:
    module_dict = {}
    weight_dict = {}
    num_modules = {}
    for data_name in data_names:
        print(data_name, split)

        predictions = torch.load(f"{predictions_base_path}/{data_name}_split{split}.pt")

        # if prediction is NaN, give a very large number
        for i in range(len(predictions)):
            result = sum(predictions[i]).numpy()
            
            if np.isnan(result).any():
                print(f"Group {i} contains NaN, replacing with a large value.")
                predictions[i] = torch.ones_like(predictions[i]) * 1e5

        # print(np.isnan(sum(predictions[3]).numpy()))

        # if prediction is NaN, give a very large number
        # predictions[3] = torch.ones_like(predictions[3]) * 1e5
        # predictions[6] = torch.ones_like(predictions[6]) * 1e5
        
        with open(f'{labels_base_path}/{data_name}_split{split}_module_mp_eform.pkl', 'rb') as f: # all labels are the same, and we take mp_eform here
            labels = pickle.load(f)
        y = labels.numpy()
        hat_y = torch.cat(predictions, dim=1).numpy()
        m = y.shape[0]
        y = y.reshape(m)

        m = min(m, QUOTA)
        y = y[:m]
        hat_y = hat_y[range(m),:]
        print("hat_y.shape:", hat_y.shape)

        # Define variables
        w = cp.Variable(N * T)
        z = cp.Variable(N * T, boolean=True)

        # Compute E_D
        residuals = hat_y @ w - y
        E_D = (1 / m) * cp.sum_squares(residuals)
        # E_D = (1 / m) * cp.sum(residuals)

        # Objective function
        objective = cp.Minimize(E_D + lambda_reg * cp.sum(z))

        # Constraints
        constraints = [
            w >= 0,
            w <= z,            # Ensures w_t = 0 when z_t = 0
            w >= epsilon * z,
            cp.sum(w) == 1,
            cp.sum(z) <= K,
        ]

        # Define and solve the problem
        prob = cp.Problem(objective, constraints)

        prob.solve(solver=cp.CPLEX, verbose=True)

        # Optimal weights and selected modules
        w_opt = w.value
        z_opt = z.value
        optimized_obj = prob.value
        print("Optimized Objective Value: ", optimized_obj)

        non_zero_indices = np.nonzero(z_opt)
        non_zero_indices = non_zero_indices[0].tolist()
        
        non_zero_indices = [i for i in non_zero_indices]

        non_zero_weights = w_opt[non_zero_indices]
        print("non_zero_weights: ", non_zero_weights)

        module_dict[data_name] = ','.join([str(num) for num in non_zero_indices])
        num_modules[data_name] = len(non_zero_indices)
        weight_dict[data_name] = non_zero_weights.tolist()
        if sum(weight_dict[data_name]) != 1:
            weight_dict[data_name][-1] = 1 - sum(weight_dict[data_name][:-1])
        assert sum(weight_dict[data_name]) == 1

        optimized_obj_results.append({
            'data_name': data_name,
            'split': split,
            'optimized_objective': optimized_obj,
            'module_dict': json.dumps(module_dict[data_name]),
            'weight_dict': json.dumps(weight_dict[data_name]),
            'num_modules': num_modules[data_name]
        })        
    
    with open(f'json/module_ids/split_{split}.json', 'w') as json_file:
        json.dump(module_dict, json_file, indent=4)
    
    with open(f'json/module_num/split_{split}.json', 'w') as json_file:
        json.dump(num_modules, json_file, indent=4)

    with open(f'json/module_weights/split_{split}.json', 'w') as json_file:
        json.dump(weight_dict, json_file, indent=4)
    
    print(module_dict)
    print(num_modules)
    print(weight_dict)


# df = pd.DataFrame(optimized_obj_results)
# df.to_csv(optimized_obj_csv_path, index=False)