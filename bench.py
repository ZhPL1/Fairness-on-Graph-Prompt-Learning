from prompt_graph.tasker import NodeTask, GraphTask
from prompt_graph.utils import seed_everything
from torchsummary import summary
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args
from prompt_graph.data import load4node,load4graph, split_induced_graphs
import pickle
import random
import numpy as np
import os
import json
import pandas as pd
def load_induced_graph(dataset_name, data, device):

    min_size = 100
    max_size = 300

    folder_path = './Experiment/induced_graph/' + dataset_name
    if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    file_path = folder_path + '/induced_graph_min'+str(min_size)+'_max'+str(max_size) +'.pkl'
    if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                print('loading induced graph...')
                graphs_list = pickle.load(f)
                print('Done!!!')
    else:
        print('Begin split_induced_graphs.')
        split_induced_graphs(data, folder_path, device, smallest_size=min_size, largest_size=max_size)
        with open(file_path, 'rb') as f:
            graphs_list = pickle.load(f)
    graphs_list = [graph.to(device) for graph in graphs_list]
    return graphs_list


args = get_args()
seed_everything(args.seed)

param_grid = {
    'learning_rate': 10 ** np.linspace(-3, -1, 1000),
    # 'learning_rate':  [0.0011325413151528113],
    'weight_decay':  10 ** np.linspace(-5, -6, 1000),
    # 'weight_decay':  [2.6633327251749805e-06],
    'batch_size': [32,64,128],
    # 'batch_size': [32]
}
# if args.dataset_name in ['PubMed']:
#      param_grid = {
#     'learning_rate': 10 ** np.linspace(-3, -1, 1000),
#     'weight_decay':  10 ** np.linspace(-5, -6, 1000),
#     'batch_size': np.linspace(128, 512, 200),
#     }
if args.dataset_name in ['ogbn-arxiv','Flickr']:
     param_grid = {
    # 'learning_rate': 10 ** np.linspace(-3, -1, 1),
    'learning_rate': 0.001,
    'weight_decay':  10 ** np.linspace(-5, -6, 1),
    'batch_size': np.linspace(512, 512, 200),
    }


num_iter=10
print('args.dataset_name', args.dataset_name)
# if args.prompt_type in['MultiGprompt','GPPT']:
#     print('num_iter = 1')
#     num_iter = 1
# if args.dataset_name in ['ogbn-arxiv','Flickr']:
#     print('num_iter = 1')
#     num_iter = 1
best_params = None
best_loss = float('inf')
best_acc = float('-inf')
final_acc_mean = 0
final_acc_std = 0
final_f1_mean = 0
final_f1_std = 0
final_roc_mean = 0
final_roc_std = 0
all_results = []


# args.task = 'GraphTask'
# # # # # args.prompt_type = 'MultiGprompt'
# args.dataset_name = 'COLLAB'
# # args.dataset_name = 'Cora'
# # num_iter = 1
# args.shot_num = 1
# args.pre_train_model_path='./Experiment/pre_trained_model/DD/DGI.GCN.128hidden_dim.pth' 


if args.task == 'NodeTask':
    data, input_dim, output_dim, sens = load4node(args.dataset_name)   
    data = data.to(args.device)
    if args.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']:
        graphs_list = load_induced_graph(args.dataset_name, data, args.device) 
    else:
        graphs_list = None 
    

         

if args.task == 'GraphTask':
    input_dim, output_dim, dataset = load4graph(args.dataset_name)
    
print('num_iter',num_iter)
for a in range(num_iter):
    params = {k: random.choice(v) for k, v in param_grid.items()}
    print(params)

    # 返回平均损失
    dicts = []
    for j in range(1):
        if args.task == 'NodeTask':
            tasker = NodeTask(pre_train_model_path = args.pre_train_model_path, 
                        dataset_name = args.dataset_name, num_layer = args.num_layer,
                        gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type,
                        epochs = args.epochs, shot_num = args.shot_num, device=args.device, lr = params['learning_rate'], wd = params['weight_decay'],
                        batch_size = int(params['batch_size']), data = data, input_dim = input_dim, output_dim = output_dim, graphs_list = graphs_list, if_fair = args.fair, sensitive = sens)


        if args.task == 'GraphTask':
            tasker = GraphTask(pre_train_model_path = args.pre_train_model_path, 
                        dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type, epochs = args.epochs,
                        shot_num = args.shot_num, device=args.device, lr = params['learning_rate'], wd = params['weight_decay'],
                        batch_size = int(params['batch_size']), dataset = dataset, input_dim = input_dim, output_dim = output_dim)
        pre_train_type = tasker.pre_train_type

        dict_j = tasker.run()
        dicts.append(dict_j)
    keys = dicts[0].keys()
    test_result = {}
    for key in keys:
        test_result[key] = sum(d[key] for d in dicts) / len(dicts)

    print(f"For {a}th searching, Tested Params: {params}, Avg Best Loss: {test_result['mean_best_loss']}")

    # Append current iteration results to all_results list
    all_results.append({
        'params': params,
        'mean_best_loss': test_result['mean_best_loss'],
        'mean_acc': test_result['mean_acc'],
        'std_acc': test_result['std_acc'],
        'mean_f1': test_result['mean_f1'],
        'std_f1': test_result['std_f1'],
        'mean_roc': test_result['mean_roc'],
        'std_roc': test_result['std_roc'],
        'mean_dp': test_result['mean_dp'],
        'std_dp': test_result['std_dp'],
        'mean_eo': test_result['mean_eo'],
        'std_eo': test_result['std_eo'],
    })


    if test_result['mean_best_loss'] < best_loss:
        best_loss = test_result['mean_best_loss']

        best_params = params
        final_acc_mean = test_result['mean_acc']
        final_acc_std = test_result['std_acc']
        final_f1_mean = test_result['mean_f1']
        final_f1_std = test_result['std_f1']
        final_roc_mean = test_result['mean_roc']
        final_roc_std = test_result['std_roc']
        final_dp_mean = test_result['mean_dp']
        final_dp_std = test_result['std_dp']
        final_eo_mean = test_result['mean_eo']
        final_eo_std = test_result['std_eo']


print("After searching, Final Accuracy {:.4f}±{:.4f}(std)".format(final_acc_mean, final_acc_std)) 
print("After searching, Final F1 {:.4f}±{:.4f}(std)".format(final_f1_mean, final_f1_std)) 
print("After searching, Final AUROC {:.4f}±{:.4f}(std)".format(final_roc_mean, final_roc_std)) 

print("After searching, Final DP {:.4f}±{:.4f}(std)".format(final_dp_mean, final_dp_std)) 
print("After searching, Final EO {:.4f}±{:.4f}(std)".format(final_eo_mean, final_eo_std)) 

print('best_params ', best_params)
print('best_loss ',best_loss)

output_result = {
        'best_loss': best_loss, 
        'final_acc_mean': final_acc_mean, 
        'final_acc_std': final_acc_std, 
        'final_f1_mean': final_f1_mean, 
        'final_f1_std': final_f1_std, 
        'final_roc_mean': final_roc_mean, 
        'final_roc_std': final_roc_std, 
        'final_dp_mean': final_dp_mean, 
        'final_dp_std': final_dp_std, 
        'final_eo_mean': final_eo_mean, 
        'final_eo_std': final_eo_std,
        'all_results': all_results}

# with open('./{}+{}.pickle'.format(args.dataset_name, args.prompt_type), 'wb') as file:
#     pickle.dump(output_result, file)
# print('Saved!')
with open('./{}+{}.json'.format(args.dataset_name, args.prompt_type), 'w') as file:
    json.dump(output_result, file, indent=4)
print('Saved!')
