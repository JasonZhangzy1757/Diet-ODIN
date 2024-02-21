import torch.nn as nn
from utils import *
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from models import *
import argparse
import wandb


def main():
    if args.use_wandb:
        run = wandb.init(project='ablation_only_b_244_3')
        config = wandb.config
        SEED = config.seed
        LR = config.lr
        DROPOUT = config.dropout
        HIDDEN_DIM = config.hidden_dim
        WEIGHT_DECAY = config.weight_decay
        MODEL_TYPE = args.model_type
        run.name = f"Run_with_{config.lr}_{config.dropout}_{config.hidden_dim}_{config.weight_decay}"
        run.save()
    else:
        SEED = args.seed
        LR = args.lr
        DROPOUT = args.dropout
        HIDDEN_DIM = args.hidden_dim
        WEIGHT_DECAY = args.weight_decay
        MODEL_TYPE = args.model_type

    set_seed(SEED)
    graph = torch.load('../processed_data/heterogeneous_graph_768_no_med_balanced_with_prompt.pt')
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if MODEL_TYPE == 'MLP':
        node_features, user_labels = process_features_for_MLP(graph)
        model = simple_MLP(num_features=node_features.size(1), num_classes=2, hidden_dim=HIDDEN_DIM)

        node_features = node_features.to(device)
    elif MODEL_TYPE == 'GCN':
        node_features, edge_index, user_labels = load_data_for_GCN(graph)
        model = GCN(num_features=node_features.size(1),
                    num_classes=2,
                    hidden_dim=HIDDEN_DIM,
                    dropout=DROPOUT)

        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
    elif MODEL_TYPE == 'SAGE':
        node_features, edge_index, user_labels = load_data_for_GCN(graph)
        model = SAGE(num_features=node_features.size(1),
                     num_classes=2,
                     hidden_dim=HIDDEN_DIM,
                     dropout=DROPOUT)

        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
    elif MODEL_TYPE == 'GAT':
        node_features, edge_index, user_labels = load_data_for_GCN(graph)
        model = GAT(num_features=node_features.size(1),
                    num_classes=2,
                    hidden_dim=HIDDEN_DIM,
                    dropout=DROPOUT)

        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
    elif MODEL_TYPE == 'RGCN':
        node_feature_dims, feature_dict, edge_index, edge_type, num_relations, user_labels = load_data_for_RGCN(graph)
        model = RGCN(num_relations, node_feature_dims, num_classes=2, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)

        feature_dict = {key: x.to(device) for key, x in feature_dict.items()}
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
    elif MODEL_TYPE == 'HAN':
        graph = metapath_generation(graph, sample_ratio=0.2)
        # UFU_edge_list = generate_neighbors(graph, ('user', 'eats', 'food'), ('food', 'eaten', 'user'),
        #                                    shared_threshold=3)
        # UFU_edge_index = edge_concat(UFU_edge_list, [])
        # UHU_edge_list = generate_neighbors(graph, ('user', 'has', 'habit'), ('habit', 'from', 'user'),
        #                                    shared_threshold=3)
        # UHU_edge_index = edge_concat(UHU_edge_list, [])
        # refined_graph = HeteroData()
        # refined_graph['user'].x = graph['user'].x
        # refined_graph['user'].y = graph['user'].y
        # refined_graph[('user', 'UFU', 'user')].edge_index = UFU_edge_index
        # refined_graph[('user', 'UHU', 'user')].edge_index = UHU_edge_index
        model = HAN(graph, in_channels=-1, out_channels=2)

        feature_dict = graph.x_dict
        edge_dict = graph.edge_index_dict
        user_labels = graph['user'].y

        feature_dict = {key: x.to(device) for key, x in feature_dict.items()}
        edge_dict = {key: x.to(device) for key, x in edge_dict.items()}
    elif MODEL_TYPE == 'HGT':
        model = HGT(graph, hidden_channels=HIDDEN_DIM, out_channels=2, num_heads=4, num_layers=2)

        feature_dict = graph.x_dict
        edge_dict = graph.edge_index_dict
        user_labels = graph['user'].y

        feature_dict = {key: x.to(device) for key, x in feature_dict.items()}
        edge_dict = {key: x.to(device) for key, x in edge_dict.items()}
    else:
        raise NotImplementedError

    train_indices, val_indices, test_indices, train_labels, val_labels, test_labels, weights = \
        set_split(user_labels, balanced_test=False, test_val_size=0.8)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    train_labels = train_labels.to(device)
    val_labels = val_labels.to(device)
    test_labels = test_labels.to(device)
    
    if MODEL_TYPE == 'HGT' or MODEL_TYPE == 'HAN':
        with torch.no_grad(): 
          out = model(feature_dict, edge_dict)

    # Training Starts here
    best_model_state = None
    best_f1_val = 0.0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        if MODEL_TYPE == 'RGCN':
            out = model(feature_dict, edge_index, edge_type)
        elif MODEL_TYPE == 'MLP':
            out = model(node_features)
        elif MODEL_TYPE == 'HGT' or MODEL_TYPE == 'HAN':
            out = model(feature_dict, edge_dict)
        else:
            out = model(node_features, edge_index)
        # Compute loss only for the subset of nodes
        loss = criterion(out[train_indices], train_labels)
        loss.backward()
        optimizer.step()

        predictions = out[train_indices].max(1)[1].cpu().numpy()
        truth = train_labels.cpu().numpy()
        f1_train = f1_score(truth, predictions)

        with torch.no_grad():
            if MODEL_TYPE == 'RGCN':
                out = model(feature_dict, edge_index, edge_type)
            elif MODEL_TYPE == 'MLP':
                out = model(node_features)
            elif MODEL_TYPE == 'HGT' or MODEL_TYPE == 'HAN':
                out = model(feature_dict, edge_dict)
            else:
                out = model(node_features, edge_index)
            val_output = out[val_indices]
            val_predictions = val_output.max(1)[1].cpu().numpy()
            test_truth = val_labels.cpu().numpy()
            f1_val = f1_score(val_predictions, test_truth)
            if args.use_wandb:
                wandb.log({
                    'train_loss': loss,
                    'f1_val': f1_val
                })

            print(f"Epoch {epoch + 1}: Train Loss: {loss.item()}, Train F1-Score: {f1_train} Val F1-Score: {f1_val}")

            if f1_val > best_f1_val:
                best_f1_val = f1_val
                best_model_state = model.state_dict().copy()

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        if MODEL_TYPE == 'RGCN':
            out = model(feature_dict, edge_index, edge_type)
        elif MODEL_TYPE == 'MLP':
            out = model(node_features)
        elif MODEL_TYPE == 'HGT' or MODEL_TYPE == 'HAN':
            out = model(feature_dict, edge_dict)
        else:
            out = model(node_features, edge_index)
        test_output = out[test_indices]
        test_probabilities = F.softmax(test_output, dim=1)[:, 1]
        test_predictions = test_output.max(1)[1].cpu().numpy()
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
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Number of hidden dimension.')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--use_wandb', type=bool, default=True,
                        help='whether to use wandb for a sweep.')
    parser.add_argument('--model_type', type=str, default='HGT',
                        help='The baseline model to use.')
    args = parser.parse_args()

    if args.use_wandb:
        wandb.login(key='2a0863bcb6510c5d64bb4c57e14b278e8fbe3fb6')
        sweep_config = {
            'name': 'sweep-try-edge-prediction',
            'method': 'grid',
            'parameters': {
                'lr': {'values': [1e-3]},
                'hidden_dim': {'values': [256]},
                'dropout': {'values': [0.6]},
                'weight_decay': {'values': [1e-2]},
                'seed': {'values': [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]}
            }
        }
        sweep_id = wandb.sweep(sweep_config, entity='jasonzhangzy1920', project='ablation_only_b_244_3')
        wandb.agent(sweep_id, function=main)
    else:
        main()
