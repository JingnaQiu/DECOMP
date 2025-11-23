from compared_methods import * 
from utils import *

CLASSES = ["N", "PB", "UDH", "ADH", "FEA", "DCIS", "IC"]
CLASSES_3 = {
    "N": "BT",
    "PB": "BT", 
    "UDH": "BT",
    "ADH": "AT", 
    "FEA": "AT", 
    "DCIS": "MT", 
    "IC": "MT",
}
def get_dist(RoIs):
    dist = np.array([RoI.split('_')[2] for RoI in RoIs])
    dist = [int(np.sum(dist == c)) for c in CLASSES]
    return dist


def select(cfg, AL_annotations, set_name, n_images, select_WSIs, pred_dict=None, class_select_weights=None, checkpoint=None):
    if select_WSIs is None:
        # random image selection
        candidate_WSIs_json = AL_annotations.replace("_select.json", "_candidate_WSIs.json")
        with open(candidate_WSIs_json, "r") as rf:
            select_WSIs = json.load(rf)[0]
        candidate_WSIs = select_WSIs[set_name]
        random.shuffle(candidate_WSIs)
    else:
        candidate_WSIs = select_WSIs[set_name]
            
    # in case of all WSIs have been selected in the train/val set
    if len(select_WSIs[set_name]) == 0:
        return
    
    n_classes = len(cfg.classes)
    print(f"region selection: {cfg.region_sampling_strategy}---------------------------------------- {set_name} set")
    
    selected = []
    N = min(n_images, len(candidate_WSIs)) 
    if cfg.region_sampling_strategy == "random":
        with open(AL_annotations, 'r') as f:
            already_selected_RoIs = json.load(f)[0][set_name]

        with open("code/filenames.json", "r") as f:
            WSI_RoI_dict = json.load(f)[0][set_name] # {WSI_1: [WSI_1_ROI_1, ..., WSI_1_ROI_n], WSI_2: [...]}

        for n in range(N):
            WSI = candidate_WSIs[n]
            RoIs = list(set(WSI_RoI_dict[WSI]) - set([RoI for RoI in already_selected_RoIs if WSI+"_" in RoI]))
            print(f"\n{WSI}: {len(RoIs)} RoIs ({get_dist(RoIs)})")

            k = min(len(RoIs), cfg.max_query_per_WSI)
            WSI_selected = random.sample(RoIs, k=k)
            print(f"--------------------------------------------selected {k} RoIs ({get_dist(WSI_selected)})")
            selected.extend(WSI_selected)

    elif cfg.region_sampling_strategy.__contains__("decomposition"):
        for n in range(N):
            WSI = candidate_WSIs[n]
            RoIs = [key for key, _ in pred_dict[set_name][WSI].items()]
            print(f"{WSI}: {len(RoIs)} RoIs ({get_dist(RoIs)})")
            
            preds = np.array([value["pred"].item() for _, value in pred_dict[set_name][WSI].items()])
            confs = np.array([value["conf"].item() for _, value in pred_dict[set_name][WSI].items()])
            
            sampling_classes = list(set(preds))
            sampling_classes_probs = np.array([class_select_weights[c] for c in sampling_classes])
            nonzero_prob_classes = np.where(sampling_classes_probs > 0.)[0]
            sampling_classes = np.array(sampling_classes)[nonzero_prob_classes]
            sampling_classes_probs = np.array(sampling_classes_probs)[nonzero_prob_classes]
            sampling_classes_probs = sampling_classes_probs / np.sum(sampling_classes_probs)
            sampling_dict = dict(zip(sampling_classes, sampling_classes_probs))
            if sampling_dict == {}:
                sampling_dict = {c: 1/n_classes for c in range(n_classes)}
                print("no target class in predictions, uniform selection for all classes")
            print("sampling dict created: ", [f"{cfg.classes[key]}: {value:.2f}" for key, value in sampling_dict.items()])

            WSI_selected = []
            k = min(len(RoIs), cfg.max_query_per_WSI)
            while len(WSI_selected) < k:
                # select a class
                c = random.choices(list(sampling_dict.keys()), list(sampling_dict.values()), k=1)[0]
                c_indices = np.argwhere(preds == c).flatten()
                c_indices = [idx for idx in c_indices if idx not in WSI_selected]

                if len(c_indices) == 0:
                    del sampling_dict[c]
                    # print(f"deleting class since no more RoIs can be selected: {CLASSES[c]}")
                    if sampling_dict == {}:
                        sampling_dict = {c: 1/n_classes for c in range(n_classes)}
                        print("no target class in predictions, uniform selection for all classes")
                else:
                    selected_index = c_indices[np.argmax(confs[c_indices])]
                    WSI_selected.append(selected_index)
                    if n_classes == 7:
                        print(f"selecting {len(WSI_selected)}/{k}: sampled class {cfg.classes[c]} ({len(c_indices)}) -- real {RoIs[selected_index].split('_')[2]}")
                    else:
                        print(f"selecting {len(WSI_selected)}/{k}: sampled class {cfg.classes[c]} ({len(c_indices)}) -- real {CLASSES_3[RoIs[selected_index].split('_')[2]]}")

            selected.extend(list(np.array(RoIs)[WSI_selected])) 

    else:
        if cfg.region_sampling_strategy=="uncertainty":
            for n in range(N):
                WSI = candidate_WSIs[n]
                RoIs = [key for key, _ in pred_dict[set_name][WSI].items()]
                print(f"{WSI}: {len(RoIs)} RoIs ({get_dist(RoIs)})")
                    
                uncertainties = np.array([value["uncertainty"].item() for _, value in pred_dict[set_name][WSI].items()])
                top_uncertain_RoIs = np.argsort(uncertainties)
                k = min(len(RoIs), cfg.max_query_per_WSI)
                WSI_selected = list(np.array(RoIs)[top_uncertain_RoIs[-k:]])
                print(f"--------------------------------------------selected {k} RoIs ({get_dist(WSI_selected)})")
                selected.extend(WSI_selected)

        else:
            candidate_RoI_metas = []
            id = -1
            for n in range(N):
                WSI = candidate_WSIs[n]
                RoIs = [key for key, _ in pred_dict[set_name][WSI].items()]
                print(f"{WSI}: {len(RoIs)} RoIs ({get_dist(RoIs)})")
                    
                uncertainties = np.array([value["uncertainty"].item() for _, value in pred_dict[set_name][WSI].items()])
                top_uncertain_RoIs = np.argsort(uncertainties)
                k = min(len(RoIs), cfg.max_query_per_WSI*3)
                print(f"------------------------------------------- add {k} candidate RoIs into the pool")
                top_uncertain_RoIs = top_uncertain_RoIs[-k:]

                for roi_idx in top_uncertain_RoIs:
                    id+=1
                    candidate_RoI_metas.append({
                        "id": id,
                        "WSI": WSI,
                        "RoI": RoIs[roi_idx],
                        "uncertainty": uncertainties[roi_idx],
                    })
                
            if len(candidate_RoI_metas) <= n_images * cfg.max_query_per_WSI:
                selected_RoI_metas = candidate_RoI_metas
            else:
                if cfg.region_sampling_strategy == "BADGE":
                    candidate_RoI_features = calculate_region_gradient_embedding(cfg, checkpoint, candidate_RoI_metas, set_name)
                else:
                    candidate_RoI_features = calculate_region_feature(cfg, checkpoint, candidate_RoI_metas, set_name)

                if cfg.region_sampling_strategy.__contains__("clustering") or cfg.region_sampling_strategy == "BADGE":
                    selected_RoI_metas = region_clustering(candidate_RoI_features, candidate_RoI_metas, N * cfg.max_query_per_WSI)
                elif cfg.region_sampling_strategy.__contains__("setcover"):
                    selected_RoI_metas = region_setcover(candidate_RoI_features, candidate_RoI_metas, N * cfg.max_query_per_WSI)
                else:
                    raise ValueError

            for n in range(N):
                WSI = candidate_WSIs[n]
                WSI_selected = [meta["RoI"] for meta in selected_RoI_metas if meta["WSI"] == WSI]
                if len(WSI_selected) > cfg.max_query_per_WSI:
                    WSI_selected_uncertainties = [meta["uncertainty"] for meta in selected_RoI_metas if meta["WSI"] == WSI]
                    WSI_selected = np.array(WSI_selected)[np.argsort(WSI_selected_uncertainties)[-cfg.max_query_per_WSI:]]
                print(f"{WSI}: selected {len(WSI_selected)} RoIs ({get_dist(WSI_selected)})")
                selected.extend(WSI_selected)
    
    with open(AL_annotations, 'r') as f:
        f_data = json.load(f)[0]
    f_data[set_name].extend(selected)
    with open(AL_annotations, 'w') as f:
        json.dump([f_data], f)
