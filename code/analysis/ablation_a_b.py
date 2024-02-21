import torch.nn as nn
import sys
sys.path.append('..')  # Add the parent folder to the import pathfrom utils import *
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from models import *
import argparse
import wandb
import pickle


def main():
    if args.use_wandb:
        run = wandb.init(project='')
        config = wandb.config
        SEED = config.seed
        LR = config.lr
        DROPOUT = config.dropout
        HIDDEN_DIM = config.hidden_dim
        WEIGHT_DECAY = config.weight_decay
        NUM_HEADS = config.num_heads
        NUM_LAYERS = config.num_layers
        run.name = f"Run_with_{config.seed}_{config.lr}_{config.hidden_dim}_{config.weight_decay}_{config.num_heads}_{config.num_layers}"
        run.save()
    else:
        SEED = args.seed
        LR = args.lr
        DROPOUT = args.dropout
        HIDDEN_DIM = args.hidden_dim
        WEIGHT_DECAY = args.weight_decay
        NUM_HEADS = args.num_heads
        NUM_LAYERS = args.num_layers

    set_seed(SEED)
    graph = torch.load('../processed_data/heterogeneous_graph_768_no_med_balanced_with_prompt.pt')

    UFU_edge_list = generate_neighbors(graph, ('user', 'eats', 'food'), ('food', 'eaten', 'user'), shared_threshold=5)
    UHU_edge_list = generate_neighbors(graph, ('user', 'has', 'habit'), ('habit', 'from', 'user'), shared_threshold=8)

    # Create a refined graph for HAN part
    share_food = torch.vstack([torch.tensor(np.array([x[0] for x in UFU_edge_list]), dtype=torch.int64),
                               torch.tensor(np.array([x[1] for x in UFU_edge_list]), dtype=torch.int64)])
    share_habit = torch.vstack([torch.tensor(np.array([x[0] for x in UHU_edge_list]), dtype=torch.int64),
                                torch.tensor(np.array([x[1] for x in UHU_edge_list]), dtype=torch.int64)])
    refined_graph = HeteroData()
    refined_graph['user'].x = graph['user'].x
    refined_graph[('user', 'share_same_food', 'user')].edge_index = share_food
    refined_graph[('user', 'share_same_habit', 'user')].edge_index = share_habit

    # Back to the original code
    user_pair_edge_index = edge_concat(UFU_edge_list, UHU_edge_list)
    train_edge_index, val_edge_index, test_edge_index = split_edges(graph, user_pair_edge_index, test_val_size=0.6)
    train_labels = get_labels(graph, train_edge_index)
    val_labels = get_labels(graph, val_edge_index)
    test_labels = get_labels(graph, test_edge_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model and optimizer
    if args.model_type == 'RGCN':
        node_feature_dims, feature_dict, edge_index, edge_type, num_relations, user_labels = load_data_for_RGCN(graph)
        model = mod_RGCN(num_relations, node_feature_dims, num_classes=2, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)

        feature_dict = {key: x.to(device) for key, x in feature_dict.items()}
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
    elif args.model_type == 'HGT':
        model = AttHGT(graph, refined_graph, num_classes=2, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                       hidden_dim=HIDDEN_DIM, dropout=DROPOUT)

        feature_dict_refined = refined_graph.x_dict
        edge_dict_refined = refined_graph.edge_index_dict
        feature_dict = graph.x_dict
        edge_dict = graph.edge_index_dict
        feature_dict_refined = {key: x.to(device) for key, x in feature_dict_refined.items()}
        edge_dict_refined = {key: x.to(device) for key, x in edge_dict_refined.items()}
        feature_dict = {key: x.to(device) for key, x in feature_dict.items()}
        edge_dict = {key: x.to(device) for key, x in edge_dict.items()}
    else:
        raise ValueError('Unknown model type')
        
    
    
    train_indices, val_indices, test_indices, train_labels, val_labels, test_labels, weights = \
        set_split(graph['user'].y, balanced_test=False, test_val_size=0.8)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Set up device
    model = model.to(device)
    criterion = criterion.to(device)
    
    train_edge_index = train_edge_index.to(device)
    val_edge_index = val_edge_index.to(device)
    test_edge_index = test_edge_index.to(device)
    train_labels = train_labels.to(device)
    val_labels = val_labels.to(device)
    test_labels = test_labels.to(device)
    

    # Edge Processing Phase
    best_model_state = None
    best_f1_val = 0.0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        # Forward pass using all nodes
        if args.model_type == 'RGCN':
            embeddings = model.forward(feature_dict, edge_index, edge_type)
        elif args.model_type == 'HGT':
            embeddings, _ = model.forward(feature_dict, edge_dict, feature_dict_refined, edge_dict_refined)
        else:
            raise ValueError('Unknown model type')
        train_preds = embeddings[train_indices]
        
        # Compute loss only for the subset of nodes
        loss = criterion(train_preds, train_labels)
        loss.backward()
        optimizer.step()

        predictions = train_preds.max(1)[1].cpu().numpy()
        truth = train_labels.cpu().numpy()
        f1_train = f1_score(truth, predictions)

        with torch.no_grad():
            if args.model_type == 'RGCN':
                embeddings = model.forward(feature_dict, edge_index, edge_type)
            elif args.model_type == 'HGT':
                embeddings, _ = model.forward(feature_dict, edge_dict, feature_dict_refined, edge_dict_refined)
            else:
                raise ValueError('Unknown model type')
                
                
            val_preds = embeddings[val_indices]
            val_predictions = val_preds.max(1)[1].cpu().numpy()
            test_truth = val_labels.cpu().numpy()
            f1_val = f1_score(val_predictions, test_truth)

            print(f"Epoch {epoch + 1}: Train Loss: {loss.item()}, Train F1-Score: {f1_train} Val F1-Score: {f1_val}")
            if args.use_wandb:
                wandb.log({
                    'train_loss': loss,
                    'f1_train': f1_train,
                    'f1_val': f1_val
                })

            if f1_val > best_f1_val:
                best_f1_val = f1_val
                best_model_state = model.state_dict().copy()

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        if args.model_type == 'RGCN':
            embeddings = model.forward(feature_dict, edge_index, edge_type)
        elif args.model_type == 'HGT':
            embeddings, attention_scores = model.forward(feature_dict, edge_dict, feature_dict_refined, edge_dict_refined)
        else:
            raise ValueError('Unknown model type')
            
            
        test_preds = embeddings[test_indices]
        
        
        test_predictions = test_preds.max(1)[1].cpu().numpy()
        test_probabilities = F.softmax(test_preds, dim=1)[:, 1]
        test_truth = test_labels.cpu().numpy()
    
        f1_test = f1_score(test_predictions, test_truth)
        print('Final Result: F1 Score - {}'.format(f1_test))
        auc_test = roc_auc_score(test_truth, test_probabilities.cpu().numpy())
        print('Final Result: AUC Score - {}'.format(auc_test))
        # Calculating Accuracy, Precision, and Recall
        accuracy_test = accuracy_score(test_truth, test_predictions)
        precision_test = precision_score(test_truth, test_predictions)
        recall_test = recall_score(test_truth, test_predictions)
        print('Final Result: Accuracy - {}'.format(accuracy_test))
        print('Final Result: Precision - {}'.format(precision_test))
        print('Final Result: Recall - {}'.format(recall_test))

        if args.use_wandb:
            wandb.log({
                'f1_test': f1_test,
                'auc_test': auc_test,
                'accuracy_test': accuracy_test,
                'precision_test': precision_test,
                'recall_test': recall_test
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Number of hidden dimension.')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--use_wandb', type=bool, default=True,
                        help='whether to use wandb for a sweep.')
    parser.add_argument('--model_type', type=str, default='HGT',
                        help='model type, RGCN or HGT')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='number of heads in HGT')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in HGT')
    args = parser.parse_args()

    if args.use_wandb:
        wandb.login(key='USE YOUR OWN KEY HERE')
        sweep_config = {
            'name': 'sweep-try-edge-prediction',
            'method': 'grid',
            'parameters': {
                'lr': {'values': [1e-3]},
                'hidden_dim': {'values': [256]},
                'num_heads': {'values': [4]},
                'num_layers': {'values': [3]},
                'dropout': {'values': [0.6]},
                'weight_decay': {'values': [1e-3]},
                'seed': {'values': [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]}
            }
        }
        sweep_id = wandb.sweep(sweep_config, entity='', project='')
        wandb.agent(sweep_id, function=main)
    else:
        main()

