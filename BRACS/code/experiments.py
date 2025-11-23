from AL_select import select
from decompose import *
from setup import setup
from train import train
from inference import *
from utils import *
# Reduce VRAM usage by reducing fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__== "__main__":
    cfg = setup()
    anno_static_csv_file = os.path.join(cfg.csv_root, f'{cfg.exp_name}.csv')

    if cfg.sampling_strategy.__contains__("decomposition"):
        tau = float(cfg.sampling_strategy.split("threshold_")[1].split("_")[0])
    else:
        tau = 0.
    ################################################################################################
    for cycle in range(1, cfg.cycles+1):
        if cycle == 1 and (not cfg.sampling_strategy == cfg.init_sampling_strategy):
            continue

        if not os.path.exists(anno_static_csv_file):
            with open(anno_static_csv_file, "a") as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(["time", "exp_ID", "cycle", "sampling_strategy", "n_query", "tau", "annotated_WSIs (%)", "annotated_RoIs (%)",
                                "---", "test_w_f1",
                                 "---", *[f"test_f1_{c}" for c in cfg.classes],
                                "---", "anno_w_f1",
                                "---", *[f"anno_f1_{c}" for c in cfg.classes],
                                "---", "train_conf",
                                "---", *[f"train_conf_{c}" for c in cfg.classes],
                                 "---", *[f"anno_{c}" for c in cfg.classes]])
                
            if not cfg.sampling_strategy == cfg.init_sampling_strategy:
                assert cycle == 2
                first_cycle_anno_static_csv_file = anno_static_csv_file.replace(cfg.sampling_strategy, cfg.init_sampling_strategy)
                with open(anno_static_csv_file, "a") as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow(pd.read_csv(first_cycle_anno_static_csv_file, sep=",").iloc[0].values.tolist())  # copy the result of first cycle

        already_processed_cycles = pd.read_csv(anno_static_csv_file, sep=",")["cycle"].values
        if len(already_processed_cycles) > 0:
            if cycle <= already_processed_cycles[-1]:
                print(f"cycle_{cycle} finished already, skipping...")
                continue

        if 'SLURM_JOB_END_TIME' in os.environ:
            job_time_left = (float(os.environ['SLURM_JOB_END_TIME']) - time.time()) / 3600
            print(f"job_time_left: {job_time_left:.2f}h")
            if job_time_left < 1.5:
                break

        cycle_exp_name = f"cycle_{cycle}_{cfg.exp_name}"
        print(f"#==============EXPERIMENT {cycle_exp_name} ======================")
        cycle_exp_dir = os.path.join(cfg.work_dir, cycle_exp_name)
        os.makedirs(cycle_exp_dir, exist_ok=True)

        checkpoint = os.path.join(cycle_exp_dir, cycle_exp_name+".pt")
        AL_annotations = os.path.join(cycle_exp_dir, cycle_exp_name + '_select.json')
        candidate_WSIs_json = AL_annotations.replace("_select.json", "_candidate_WSIs.json")

        train_conf = None
        train_select_weights = None
        pred_dict = None
        anno_f1 = 0
        anno_f1_classes = [0] * cfg.n_classes
        cycle_select_WSIs = None
        
        mode = [
            'select',
            'anno_perc_calculation',
            'training',
            'testing',
        ]

        for m in mode:
            print(f"-----------------------------------------------------------------------{m}")
            if m == 'select':
                if cfg.sampling_strategy.__contains__("full"):
                    AL_annotations = 'code/full_select.json'
                    last_AL_annotations = None
                    last_checkpoint = None
                    continue

                if cycle == 1:
                    last_AL_annotations = None
                    last_checkpoint = None
                    last_candidate_WSIs_json = None
                else:
                    last_AL_annotations = AL_annotations.replace(f"cycle_{cycle}", f"cycle_{cycle-1}")
                    last_checkpoint = checkpoint.replace(f"cycle_{cycle}", f"cycle_{cycle-1}")
                    last_candidate_WSIs_json = candidate_WSIs_json.replace(f"cycle_{cycle}", f"cycle_{cycle-1}")

                    if cycle == 2:
                        last_AL_annotations = last_AL_annotations.replace(cfg.sampling_strategy, cfg.init_sampling_strategy)
                        last_checkpoint = last_checkpoint.replace(cfg.sampling_strategy, cfg.init_sampling_strategy)
                        last_candidate_WSIs_json = last_candidate_WSIs_json.replace(cfg.sampling_strategy, cfg.init_sampling_strategy)

                if os.path.exists(AL_annotations):
                    if last_AL_annotations is not None and \
                            get_anno_statics(cfg.classes, cfg.full_annotations, AL_annotations, return_percentage=True)[1] > \
                            get_anno_statics(cfg.classes, cfg.full_annotations, last_AL_annotations, return_percentage=True)[1]:
                        continue

                prepare_AL_annotations(AL_annotations, last_AL_annotations)
                prepare_candidate_WSIs_json(candidate_WSIs_json, last_candidate_WSIs_json, last_AL_annotations)
                
                # image selection
                if (cfg.sampling_strategy.__contains__("decomposition") or cfg.image_sampling_strategy.__contains__("uncertainty")):
                    pred_dict, train_conf = predict_train_val(cfg=cfg, 
                                                              AL_annotations=AL_annotations, 
                                                              checkpoint=last_checkpoint, 
                                                              tau=tau, 
                                                              return_pred=cfg.sampling_strategy.__contains__("decomposition"),
                                                              return_conf=cfg.sampling_strategy.__contains__("decomposition"),
                                                              return_uncertainty=cfg.sampling_strategy.__contains__("uncertainty"))

                    if cfg.sampling_strategy.__contains__("decomposition"):
                        train_select_weights = 1 - train_conf
                        train_select_weights = update_class_select_weights(cfg.classes, train_select_weights)

                    cycle_select_WSIs = compute_img_value(cfg,
                                                          pred_dict=pred_dict, 
                                                          class_select_weights=train_select_weights if cfg.image_sampling_strategy.__contains__("decomposition") else None)

                for set_name in ["train", "val"]:
                    select(cfg=cfg,
                           AL_annotations=AL_annotations, 
                           set_name=set_name,
                           n_images=cfg.n_query if set_name=="train" else max(1, int(cfg.n_query/3)),
                           select_WSIs=cycle_select_WSIs,
                           pred_dict=pred_dict,
                           class_select_weights=train_select_weights,
                           checkpoint=last_checkpoint)
                pred_dict = None
                if cycle > 3 and os.path.exists(last_checkpoint):
                    print(f"deleting {last_checkpoint}... ")
                    os.remove(last_checkpoint)
                
            if m == 'anno_perc_calculation':
                if last_AL_annotations:
                    anno_WSIs, anno_files, anno_files_classes = get_anno_statics(
                        cfg.classes, cfg.full_annotations, AL_annotations=last_AL_annotations, return_percentage=True)
                print("="*50)
                anno_WSIs, anno_files, anno_files_classes = get_anno_statics(
                    cfg.classes, cfg.full_annotations, AL_annotations=AL_annotations, return_percentage=True)
                for set_name in ["train", "val"]:
                    candidate_WSIs_json = os.path.join(*AL_annotations.split("/")[:-2], f"candidate_WSIs_{set_name}.json")
                    if os.path.exists(candidate_WSIs_json):
                        with open(candidate_WSIs_json, 'r') as f:
                            candidate_WSIs = json.load(f)
                        print(f"candidate_WSIs ({set_name}): {len(candidate_WSIs)}")
                
            if m == 'training':
                if not os.path.exists(checkpoint):
                    train(cfg, AL_annotations, checkpoint, plot_training_curves=False)

            if m == 'testing':
                model_list = glob(f"{cycle_exp_dir}/*pt")
                test_f1, test_f1_classes = predict(cfg, model_list, checkpoint)
                test_metric = ["---"] + [f"{test_f1*100:.2f}"] +\
                              ["---"] + [f"{f1*100:.2f}" for f1 in test_f1_classes]
                anno_metric = ["---"] + [f"{anno_f1*100:.2f}"] +\
                              ["---"] + [f"{f1*100:.2f}" for f1 in anno_f1_classes]

                anno_WSIs, anno_files, anno_files_classes = get_anno_statics(
                    cfg.classes, cfg.full_annotations, AL_annotations=AL_annotations, return_percentage=True)

                if train_conf is None:
                    train_conf = ["---"] + ["0"] + ["---"] + ["0"]*cfg.n_classes
                else:
                    train_conf *= 100
                    train_conf = ["---"] + [f"{np.mean(train_conf):.2f}"] +\
                                 ["---"] + [f"{value:.2f}" for value in train_conf]

                head = [time.strftime("%Y-%m-%d %H:%M:%S"), cfg.exp_id, cycle, cfg.sampling_strategy, cfg.n_query, tau,f"{anno_WSIs:.2f}", f"{anno_files:.2f}"]
                       
                row = head + test_metric +  anno_metric + train_conf +  ["---"] +  [f"{anno_files_class:.2f}" for anno_files_class in anno_files_classes]

                with open(anno_static_csv_file, "a") as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow(row)
                    
