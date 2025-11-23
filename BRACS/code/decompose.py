from dataloader import make_al_data_loader
from histocartography.ml import HACTModel
from utils import *



def helper(batch, model, checkpoint, return_pred=False, return_uncertainty=False):
    preds, uncertainties = None, None
    model.eval()
    model.load_state_dict(torch.load(checkpoint))

    with torch.no_grad():
        logits = model(*batch)  # (bs, n_classes)
        probs = torch.softmax(logits, dim=1)  # (bs, n_classes)

    if return_pred:
        preds = probs.argmax(dim=1)

    if return_uncertainty:
        uncertainties = Categorical(probs=probs).entropy()

    return probs, preds, uncertainties

def predict_train_val(cfg, AL_annotations, checkpoint, tau=None, return_pred=True, return_uncertainty=False, return_conf=True):
    """
    return_pred: return the predicted labels for each unlabeled train/val sample
    return_uncertainty: return the uncertainties for each unlabeled train/val sample
    return_conf: return the confidence level for each class based on the training sample predictions
    """
    print(f"predict_train_val...")
    print("creating model...")
    model = HACTModel(
        cg_gnn_params=cfg.model_config['cg_gnn_params'],
        tg_gnn_params=cfg.model_config['tg_gnn_params'],
        classification_params=cfg.model_config['classification_params'],
        cg_node_dim=NODE_DIM,
        tg_node_dim=NODE_DIM,
        num_classes=cfg.n_classes,
    ).to(DEVICE)

    pred_dict = {"train": {}, "val": {}}
    confidence_rate = None
    for set_name in ["train", "val"]:
        print(f"{set_name} set: start predicting...")
        print("loading training data...")
        dataloader = make_al_data_loader(
            AL_annotations=AL_annotations,
            set_name=set_name,
            cg_path=os.path.join(cfg.cg_path, set_name),
            tg_path=os.path.join(cfg.tg_path, set_name),
            assign_mat_path=os.path.join(cfg.assign_mat_path, set_name),
            batch_size=cfg.batch_size,
            shuffle=False,
            load_in_ram=False,
            load_unannotated=True,
            task_3_classes=True if cfg.task == "3-classes" else False,
        )
        
        pred_list = []
        conf_list = []
        uncertainty_list = []

        for batch in tqdm(dataloader):
            data = batch[:-1]  # cell_graph, tissue_graph, assignment_matrix, label
            probs, preds, uncertainties = helper(data, model, checkpoint, return_pred=True, return_uncertainty=return_uncertainty)
            if return_pred or (return_conf and set_name == "train"):
                pred_list.extend(preds)
                conf_list.extend(torch.amax(probs, dim=1))
            if return_uncertainty:
                uncertainty_list.extend(uncertainties)

        if return_pred or return_uncertainty:
            name_list = dataloader.dataset.files

            for i, name in enumerate(name_list):
                res = {}
                if return_pred:
                    res.update({"pred": pred_list[i]})
                    res.update({"conf": conf_list[i]})
                if return_uncertainty:
                    res.update({"uncertainty": uncertainty_list[i]})
                pred_dict[set_name].update({name: res})

            # sort pred_dict from {ROI_1_name: {"pred": 1, "uncertainty": 0}, ROI_2_name: {"pred": 1, "uncertainty": 0} ...} to {WSI_1: [{WSI_1_ROI_1_name: {"pred": 1, "uncertainty": 0}}, ...]}
            tmp_dict = {}
            RoI_list = list(pred_dict[set_name].keys())
            WSIs = set([RoI.split("_")[1] for RoI in RoI_list])
            for WSI in WSIs:
                WSI_ROIs = [{key: value} for key, value in pred_dict[set_name].items() if key.__contains__(f"BRACS_{WSI}")]
                tmp_dict.update({f"BRACS_{WSI}": WSI_ROIs})
            tmp_dict = {WSI_name: {RoI_name: RoI_value for RoI in WSI_value for RoI_name, RoI_value in RoI.items()} for WSI_name, WSI_value in tmp_dict.items()}  # from list to dict
            pred_dict[set_name] = tmp_dict
            del tmp_dict
        if return_conf and set_name == "train":
            conf_list = torch.tensor(conf_list)
            pred_list = torch.tensor(pred_list)
            class_conf_list = [conf_list[pred_list==c] for c in range(cfg.n_classes)]

            confident_samples = torch.tensor([torch.sum(class_conf_list[c]>tau) for c in range(cfg.n_classes)])
            total_samples = torch.tensor([len(class_conf_list[c]) for c in range(cfg.n_classes)])

            confidence_rate = torch.full_like(confident_samples, fill_value=0., dtype=torch.float32)
            mask = total_samples != 0
            confidence_rate[mask] = confident_samples[mask] / total_samples[mask]
            confidence_rate = confidence_rate.cpu().numpy()
            confidence_rate = np.round(confidence_rate, 2)
            print(f"\nconfidence_rate = {confidence_rate}")

        torch.cuda.empty_cache()
        del pred_list, conf_list, uncertainty_list
    return pred_dict, confidence_rate


def compute_img_value(cfg, pred_dict, class_select_weights=None):
    print("compute_img_value...")
    if class_select_weights is not None:
        class_select_weights = torch.tensor(list(class_select_weights.values()))
        if torch.all(class_select_weights.isclose(torch.tensor(1/cfg.n_classes))):
            print(f"same weights for all classes, skip from compute_img_value...")
            return None  # random image selection, no image has priority

    select_filenames = {}
    for set_name in ["train", "val"]:
        WSIs = list(pred_dict[set_name].keys())
        print(f"{set_name} set: {len(WSIs)} WSIs...")

        WSI_values = []
        for WSI in WSIs:
            if cfg.image_sampling_strategy.__contains__("uncertainty"):
                RoI_uncertainties = torch.tensor([RoI_value["uncertainty"] for RoI_name, RoI_value in pred_dict[set_name][WSI].items()])
                WSI_value = torch.mean(RoI_uncertainties)
            elif cfg.image_sampling_strategy.__contains__("decomposition"):
                RoI_preds = torch.tensor([RoI_value["pred"].item() for RoI_name, RoI_value in pred_dict[set_name][WSI].items()])
                RoI_preds = torch.tensor([torch.sum(RoI_preds == c) for c in range(cfg.n_classes)])
                RoI_preds = torch.tensor([p>0 for p in RoI_preds], dtype=int)  # counts if a certain class is predicted among the RoIs of a WSI, avoid the impact of number of RoIs contained by each WSI
                WSI_value = torch.sum(RoI_preds * class_select_weights)

            else:
                raise ValueError
            WSI_values.append(WSI_value)

        top_value_indices = torch.argsort(torch.tensor(WSI_values), descending=True)
        select_list = [WSIs[idx] for idx in top_value_indices]
        select_filenames.update({set_name: select_list})

    return select_filenames
