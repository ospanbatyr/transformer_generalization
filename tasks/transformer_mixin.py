import torch.nn
from layers.transformer import Transformer, UniversalTransformer, RelativeTransformer, UniversalRelativeTransformer
from models import TransformerEncDecModel
from interfaces import TransformerEncDecInterface
import os
import os.path as osp


class TransformerMixin:
    def create_model(self) -> torch.nn.Module:
        rel_args = dict(pos_embeddig=(lambda x, offset: x), embedding_init="xavier")
        trafos = {
            "scaledinit": (Transformer, dict(embedding_init="kaiming", scale_mode="down")),
            "opennmt": (Transformer, dict(embedding_init="xavier", scale_mode="opennmt")),
            "noscale": (Transformer, {}),
            "universal_noscale": (UniversalTransformer, {}),
            "universal_scaledinit": (UniversalTransformer, dict(embedding_init="kaiming", scale_mode="down")),
            "universal_opennmt": (UniversalTransformer, dict(embedding_init="xavier", scale_mode="opennmt")),
            "relative": (RelativeTransformer, rel_args),
            "relative_universal": (UniversalRelativeTransformer, rel_args)
        }

        constructor, args = trafos[self.helper.args.transformer.variant]

        model = TransformerEncDecModel(len(self.train_set.in_vocabulary),
                                      len(self.train_set.out_vocabulary), self.helper.args.state_size,
                                      nhead=self.helper.args.transformer.n_heads,
                                      num_encoder_layers=self.helper.args.transformer.encoder_n_layers,
                                      num_decoder_layers=self.helper.args.transformer.decoder_n_layers or \
                                                         self.helper.args.transformer.encoder_n_layers,
                                      ff_multiplier=self.helper.args.transformer.ff_multiplier,
                                      transformer=constructor,
                                      tied_embedding=self.helper.args.transformer.tied_embedding, **args)
        
        finetune_steps = self.helper.args.finetune_steps
        finetune_sweep_name = self.helper.args.finetune_sweep_name

        if finetune_steps is None or finetune_sweep_name is None:
            return model

        # load the trained model
        checkpoint_fname = self.trained_model_address()
        checkpoint = model.load_state_dict(torch.load(checkpoint_fname)["model"])
        print(f"Loaded model from {checkpoint_fname}")
        return model


    def create_model_interface(self):
        self.model_interface = TransformerEncDecInterface(self.model, label_smoothing=self.helper.args.label_smoothing)

    
    def trained_model_address(self):
        sweep_folder = osp.join("wandb", f"sweep-{self.helper.args.finetune_sweep_name}")
        # find the name of the yaml file in the sweep folder
        sweep_file = [f for f in os.listdir(sweep_folder) if f.endswith(".yaml")][0]
        # the name of the yaml file is in the format config-{run_name}.yaml, get the run_name
        run_name = sweep_file.split("-")[1].split(".")[0]
        # find the folder under wandb folder that has run name in its name
        run_folder = [f for f in os.listdir("wandb") if run_name in f][0]
        # find the name of the checkpoint file in the run folder
        checkpoint_fname = osp.join("wandb", run_folder, "files", "checkpoint", f"model-{self.helper.args.finetune_steps}.pth")
        return checkpoint_fname

