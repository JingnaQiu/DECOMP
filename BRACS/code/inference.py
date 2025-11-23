from utils import *
from dataloader import make_data_loader, make_al_data_loader
from histocartography.ml import HACTModel


def predict(cfg, model_list, model_path=None):
    """
    determine the best f1 test score from a list of models that are stored when the validation loss/acc/f1 achieved the best score
    preserve the best model to model_path, delete all other models

    if model_path != None: copy the best model in the model_list to model_path and delete the rest
    """
    print(f"inference------------------------------")
    test_dataloaders = []
    print("loading test data...")
    test_dataloader = make_data_loader(
        cg_path=os.path.join(cfg.cg_path, 'test'),
        tg_path=os.path.join(cfg.tg_path, 'test'),
        assign_mat_path=os.path.join(cfg.assign_mat_path, 'test'),
        batch_size=cfg.batch_size,
        shuffle=False,
        load_in_ram=False,
        task_3_classes=True if cfg.task=="3-classes" else False,
    )
    print(f"loaded {len(test_dataloader.dataset)} test images.")
    test_dataloaders.append(test_dataloader)

    print("creating model...")
    model = HACTModel(
        cg_gnn_params=cfg.model_config['cg_gnn_params'],
        tg_gnn_params=cfg.model_config['tg_gnn_params'],
        classification_params=cfg.model_config['classification_params'],
        cg_node_dim=NODE_DIM,
        tg_node_dim=NODE_DIM,
        num_classes=cfg.n_classes,
    ).to(DEVICE)

    print("start testing...")
    test_f1_list = []
    test_f1_classes_list = []

    for m in model_list:
        model.eval()
        model.load_state_dict(torch.load(m))

        all_test_logits = []
        all_test_labels = []
        for test_dataloader in test_dataloaders:
            for batch in test_dataloader:
                labels = batch[-1]
                data = batch[:-1]
                with torch.no_grad():
                    logits = model(*data)
                all_test_logits.append(logits)
                all_test_labels.append(labels)

        all_test_logits = torch.cat(all_test_logits).cpu()
        all_test_preds = torch.argmax(all_test_logits, dim=1)
        all_test_labels = torch.cat(all_test_labels).cpu()

        test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
        test_f1_classes = f1_score(all_test_labels, all_test_preds, average=None)
        test_f1_list.append(test_f1)
        test_f1_classes_list.append(test_f1_classes)
        print(f"testing model: {m.split('/')[-1]}",
              '- weighted F1 score {}'.format(np.round(test_f1, 4)))
        torch.cuda.empty_cache()

    best_model_index = np.argmax(test_f1_list)
    test_f1 = test_f1_list[best_model_index]
    test_f1_classes = test_f1_classes_list[best_model_index]

    if model_path and not os.path.exists(model_path):
        shutil.copy(src=model_list[best_model_index], dst=model_path)
        for m in model_list:
            os.remove(m)

    return test_f1, test_f1_classes
