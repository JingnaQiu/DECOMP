
from dataloader import make_al_data_loader
from utils import *
from histocartography.ml import HACTModel


def train(cfg, AL_annotations, model_path, plot_training_curves=True):
    # print_AL_annotations(AL_annotations)
    # make data loaders (train, validation & test)
    print("loading training data...")
    train_dataloader = make_al_data_loader(
        AL_annotations=AL_annotations,
        set_name='train',
        cg_path=os.path.join(cfg.cg_path, 'train'),
        tg_path=os.path.join(cfg.tg_path, 'train'),
        assign_mat_path=os.path.join(cfg.assign_mat_path, 'train'),
        batch_size=cfg.batch_size,
        load_in_ram=cfg.in_ram,
        task_3_classes=True if cfg.task == "3-classes" else False,
    )

    print("loading validation data...")
    val_dataloader = make_al_data_loader(
        AL_annotations=AL_annotations,
        set_name='val',
        cg_path=os.path.join(cfg.cg_path, 'val'),
        tg_path=os.path.join(cfg.tg_path, 'val'),
        assign_mat_path=os.path.join(cfg.assign_mat_path, 'val'),
        batch_size=cfg.batch_size,
        load_in_ram=cfg.in_ram,
        task_3_classes=True if cfg.task == "3-classes" else False,
    )

    print("creating model...")
    model = HACTModel(
        cg_gnn_params=cfg.model_config['cg_gnn_params'],
        tg_gnn_params=cfg.model_config['tg_gnn_params'],
        classification_params=cfg.model_config['classification_params'],
        cg_node_dim=NODE_DIM,
        tg_node_dim=NODE_DIM,
        num_classes=cfg.n_classes,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    step = 0
    best_val_loss = 10e5
    best_val_acc = 0.
    best_val_f1 = 0.
    model_best_val_loss = model_path.replace('.pt', '_best_val_loss.pt')
    model_best_val_acc = model_path.replace('.pt', '_best_val_acc.pt')
    model_best_val_f1 = model_path.replace('.pt', '_best_val_f1.pt')
    print(time.strftime("%Y-%m-%d %H:%M:%S"),
          '{0:<10}'.format('Epoch'),
          '{0:>10}'.format('lr'),
          '{0:>10}'.format('train_loss'),
          '{0:>10}'.format('val_loss'),
          '{0:>10}'.format('val_acc'),
          '{0:>10}'.format('val_f1'))
    lr_curve = []
    train_loss_curve = []
    val_loss_curve = []
    val_acc_curve = []
    val_f1_curve = []
    for epoch in range(cfg.epochs):
        lr = np.round(optimizer.param_groups[0]['lr'], 5)
        lr_curve.append(lr)

        train_loss = 0.
        model = model.to(DEVICE)
        model.train()
        for batch in train_dataloader:
            labels = batch[-1]  # cg, tg, assign_mat, label
            data = batch[:-1]
            logits = model(*data)

            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            train_loss += loss.item()
        train_loss /= train_dataloader.dataset.__len__()
        train_loss_curve.append(train_loss)

        model.eval()
        all_val_logits = []
        all_val_labels = []
        for batch in val_dataloader:
            labels = batch[-1]
            data = batch[:-1]
            with torch.no_grad():
                logits = model(*data)
            all_val_logits.append(logits)
            all_val_labels.append(labels)

        all_val_logits = torch.cat(all_val_logits).cpu()
        all_val_preds = torch.argmax(all_val_logits, dim=1)
        all_val_labels = torch.cat(all_val_labels).cpu()

        with torch.no_grad():
            val_loss = loss_fn(all_val_logits, all_val_labels).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_best_val_loss)
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_best_val_acc)
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_best_val_f1)
        print(time.strftime("%Y-%m-%d %H:%M:%S"),
              f"{epoch+1:3d}/{cfg.epochs}      ",
              '{0:>10}'.format(lr),
              '{0:>10}'.format(np.round(train_loss, 4)),
              '{0:>10}'.format(np.round(val_loss, 4)),
              '{0:>10}'.format(np.round(val_acc, 4)),
              '{0:>10}'.format(np.round(val_f1, 4)))
        val_loss_curve.append(val_loss)
        val_acc_curve.append(val_acc)
        val_f1_curve.append(val_f1)

        lr_scheduler.step()
        torch.cuda.empty_cache()

    if plot_training_curves:
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        axes[0].plot(lr_curve, label="lr")
        axes[1].plot(train_loss_curve, label="train_losss")
        axes[2].plot(val_loss_curve, label="val_loss")
        axes[3].plot(val_acc_curve, label="val_acc")
        axes[3].plot(val_f1_curve, label="val_f1")
        for ax in axes.ravel():
            ax.legend()
        plt.tight_layout()
        plt.savefig(model_path.replace('.pt', '_curve.png'))
        plt.close()
