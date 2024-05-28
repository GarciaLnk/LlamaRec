import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pytorch_lightning import seed_everything

from config import EXPERIMENT_ROOT, PROJECT_NAME, args, set_template
from dataloader import dataloader_factory
from model import BERT, NARM, LRURec, SASRec
from trainer import BERTTrainer, LRUTrainer, RNNTrainer, SASTrainer

try:
    os.environ["WANDB_PROJECT"] = PROJECT_NAME
except:
    print("WANDB_PROJECT not available, please set it in config.py")


def main(args, export_root=None):
    seed_everything(args.seed)
    train_loader, val_loader, test_loader = dataloader_factory(args)

    if args.model_code == "lru":
        model = LRURec(args)
    elif args.model_code == "bert":
        model = BERT(args)
    elif args.model_code == "sas":
        model = SASRec(args)
    elif args.model_code == "narm":
        model = NARM(args)

    if export_root == None:
        export_root = EXPERIMENT_ROOT + "/" + args.model_code + "/" + args.dataset_code

    if args.model_code == "lru":
        trainer = LRUTrainer(
            args,
            model,
            train_loader,
            val_loader,
            test_loader,
            export_root,
            args.use_wandb,
        )
    elif args.model_code == "bert":
        trainer = BERTTrainer(
            args,
            model,
            train_loader,
            val_loader,
            test_loader,
            export_root,
            args.use_wandb,
        )
    elif args.model_code == "sas":
        trainer = SASTrainer(
            args,
            model,
            train_loader,
            val_loader,
            test_loader,
            export_root,
            args.use_wandb,
        )
    elif args.model_code == "narm":
        args.num_epochs = 100
        trainer = RNNTrainer(
            args,
            model,
            train_loader,
            val_loader,
            test_loader,
            export_root,
            args.use_wandb,
        )

    trainer.train()
    trainer.test()

    # the next line generates val / test candidates for reranking
    trainer.generate_candidates(os.path.join(export_root, "retrieved.pkl"))


if __name__ == "__main__":
    set_template(args)

    if args.hyperparam_search:
        # searching best hyperparameters
        for decay in [0, 0.01]:
            for dropout in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                args.weight_decay = decay
                args.bert_dropout = dropout
                args.bert_attn_dropout = dropout
                export_root = (
                    EXPERIMENT_ROOT
                    + "/"
                    + args.model_code
                    + "/"
                    + args.dataset_code
                    + "/"
                    + str(decay)
                    + "_"
                    + str(dropout)
                )
                main(args, export_root=export_root)
    else:
        main(args, export_root=None)
