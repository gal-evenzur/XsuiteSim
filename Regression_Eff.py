# %% IMPORTS #
print("Importing libraries...")
import time
import sys
sys.modules['horovod'] = None
sys.modules['horovod.torch'] = None
start_import_time = time.time()
from NNfunctions import *

from torch.optim import Adam, AdamW
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
print("Time for torch imports:", time.time() - start_import_time)

from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Loss

import os
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'afmhot'

start_script_time = time.time()
print(f"Imports done in {start_script_time - start_import_time:.2f} seconds")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    choice = sys.argv[1]
    cluster_flag = True
except:
    cluster_flag = False
# %%% PARAMS # 
hyperVar = {
    # Data parameters
    'batch_size': 8, # Bigger = stable gradients and smaller updates
    'device': device,
    'cluster_flag': cluster_flag,
    'num_workers': 4,  # Number of DataLoader workers

    # Model parameters
    'Bnumber': 0,  # 0 for B0, 1 for B1, 2 for B2


    # Optimiser parameters
    'optimizer': Adam, # AdamW or Adam
    'h_lr': 5e-2,
    'b_lr': 5e-2,
    'weight_decay': 0,
    'wd_off_below_lr': 0,
    'beta1': 0.9, # ++ smoother training but slower response to changes
    'beta2': 0.999, # -- faster adaptation of learning rates but potentially less stability
    'amsgrad': False, 

    # Training procedure parameters
    'freeze_backbone_epochs': 0,    # set to 0 to disable; no reinit needed since lr=0 during freeze
    'n_epochs': 1000,
    'patience': 50,
    'score_metric': 'val_loss',  # 'val_loss'
    'lr_factor': 0.5,
    'lr_patience': 10, # number of no improvement rounds before lowering lr
    'min_lr': 1e-5,

    # Plotting parameters
    'n_plotted': 10,
}

criterion = nn.L1Loss()

hyperVar['suffix'] = f'[tests]{criterion.__class__.__name__}_B_{hyperVar["Bnumber"]}_wd{hyperVar["weight_decay"]}_bs{hyperVar["batch_size"]}'



# %% &&&&&& IMPORTING DATA &&&&&&
start_time = time.time()
data_path = '/storage/agrp/galeven/Data_new/uncompressed_merged_data_2.h5'
trainSet, validateSet, testSet = CreateTrainValTest(data_path, 0.8, 0.17, seed=120112)

dataloader = DataLoader(trainSet, 
                        batch_size=hyperVar["batch_size"], 
                        shuffle=True, 
                        num_workers=hyperVar['num_workers']) # Use > 0 workers!

validate_loader = DataLoader(validateSet, 
                             batch_size=hyperVar["batch_size"], 
                             shuffle=False, 
                             num_workers=hyperVar['num_workers'])

test_loader = DataLoader(testSet, 
                         batch_size=hyperVar["batch_size"], 
                         shuffle=False, 
                         num_workers=hyperVar['num_workers'])

load_time = time.time() - start_time
print(f"Data loaded in {load_time:.2f} seconds")
print(f"Dataset sizes: Train={len(trainSet)}, Validation={len(validateSet)}, Test={len(testSet)}")


# %% Structure for the NN: #
class EfficientNet(nn.Module):
    def __init__(self, n_classes=29, pretrained=True, freeze_pretrained=False):
        """
        Initializes a EfficientNet model adapted for single-channel input.
        Args:
            n_classes (int): The number of output values to predict.
            pretrained (bool): Whether to use pre-trained weights from ImageNet.
            freeze_pretrained (bool): If True, freezes all layers except the modified
                                     input and output layers.
        """
        super().__init__()
        # Load a pre-trained EfficientNet model
        if hyperVar['Bnumber'] == 0:
            self.net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        elif hyperVar['Bnumber'] == 1:
            self.net = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT if pretrained else None)
        elif hyperVar['Bnumber'] == 2:
            self.net = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT if pretrained else None)
        elif hyperVar['Bnumber'] == 3:
            self.net = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
        elif hyperVar['Bnumber'] == 4:
            self.net = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT if pretrained else None)
        elif hyperVar['Bnumber'] == 5:
            self.net = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT if pretrained else None)


        # --- Freeze pre-trained layers if requested ---
        if pretrained and freeze_pretrained:
            for param in self.net.parameters():
                param.requires_grad = False

        if pretrained:
            # --- Adapt for 1-channel (grayscale) input ---
            original_conv = self.net.features[0][0]  # First conv layer in EfficientNet
            self.net.features[0][0] = nn.Conv2d(
                in_channels=3,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            with torch.no_grad(): #assign channels
                w = original_conv.weight.data
                # Initialize each of the 3 channels with the same averaged weights
                averaged_weights = w.mean(dim=1, keepdim=True)
                self.net.features[0][0].weight.data = averaged_weights.repeat(1, 3, 1, 1)

        # --- Adapt the final layer for regression ---
        num_ftrs = self.net.classifier[1].in_features
        self.net.classifier[1] = nn.Linear(num_ftrs, n_classes)
        # Add activation
        

        # --- Unfreeze the new layers so they can be trained ---
        if pretrained and freeze_pretrained:
            # for param in self.net.features[0][0].parameters():
            #     param.requires_grad = True
            for param in self.net.classifier[1].parameters():
                param.requires_grad = True

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1, freq_bins, time_bins)
        Returns:
            torch.Tensor: The model's output.
        """
        return self.net(x)

def build_param_groups(model, lr_head, lr_body, weight_decay):
    """
    4 groups:
      0: backbone weights (with weight decay)
      1: backbone bias/norm (no decay)
      2: head weights (with weight decay)
      3: head bias/norm (no decay)
    """
    bb_wd, bb_nd, hd_wd, hd_nd = [], [], [], []

    def is_norm_or_bias(name):
        n = name.lower()
        return n.endswith(".bias") or "bn" in n or "norm" in n or "gn" in n

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = name.startswith("net.classifier")
        if is_head:
            (hd_nd if is_norm_or_bias(name) else hd_wd).append(p)
        else:
            (bb_nd if is_norm_or_bias(name) else bb_wd).append(p)

    groups = [
        {'params': bb_wd, 'lr': lr_body, 'weight_decay': weight_decay, 'backbone': True,  'decay': True},
        {'params': bb_nd, 'lr': lr_body, 'weight_decay': 0.0,        'backbone': True,  'decay': False},
        {'params': hd_wd, 'lr': lr_head, 'weight_decay': weight_decay/10, 'backbone': False, 'decay': True},
        {'params': hd_nd, 'lr': lr_head, 'weight_decay': 0.0,        'backbone': False, 'decay': False},
    ]
    # drop any empty groups
    return [g for g in groups if len(g['params']) > 0]

def set_bn_eval(m: nn.Module):
    # Put all BatchNorm variants in eval mode to freeze running_mean/var updates
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()
        pass

# %% DELETING OLD PARAMETERS !!! 
# Get number of outputs from dataloader
sample_batch = next(iter(dataloader))
_, sample_targets = sample_batch
hyperVar['n_outputs'] = sample_targets.shape[1]
model = EfficientNet(n_classes=hyperVar['n_outputs'], pretrained=False, freeze_pretrained=False)
print(f"Model initialized with EfficientNet B{hyperVar['Bnumber']} backbone.")
# %% ENGINE SETUP #

losses = []
val_losses = []
eps = []
lr = [hyperVar['h_lr'], hyperVar['b_lr']]


param_groups = build_param_groups(
    model,
    lr_head=hyperVar['h_lr'],
    lr_body=hyperVar['b_lr'],
    weight_decay=hyperVar['weight_decay']
)

optimizer = hyperVar['optimizer'](
    param_groups,
    betas=(hyperVar['beta1'], hyperVar['beta2']),
    amsgrad=hyperVar['amsgrad'],
)

def supervised_train(model, device="cpu"): 
    model.to(device)
    optimizer.zero_grad() 

    def _update(engine, batch):
        model.train()
        if engine.state.epoch <= hyperVar['freeze_backbone_epochs']:
            model.apply(set_bn_eval)  # Freeze BatchNorm running stats for the first epoch
        optimizer.zero_grad()  ## This zeroes out the gradient stored in "model".
        X_batch, Y_batch = batch
        X, Y = X_batch.to(device), Y_batch.to(device)
        Y_pred = model(X)
        # Measure the loss/error, gonna be high at first
        loss = criterion(Y_pred, Y)  # predicted values vs the y_train
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)


def supervised_evaluator(model, device="cpu"):

    def _interference(engine, batch):
        model.eval()
        with torch.no_grad():
            X_batch, Y_batch = batch
            X, Y = X_batch.to(device), Y_batch.to(device)
            Y_pred = model(X)
            return Y_pred, Y

    engine = Engine(_interference)

    # Attach loss metric
    Loss(criterion,
    ).attach(engine, 'loss')

    return engine

trainer = supervised_train(model, device=device)
evaluator = supervised_evaluator(model, device=device)

def score(engine):
    return -engine.state.metrics['loss']

handler = EarlyStopping(
    patience=hyperVar['patience'],  # Number of epochs to wait for improvement
    score_function=score,
    trainer=trainer
)
evaluator.add_event_handler(Events.COMPLETED, handler)

# Add ReduceLROnPlateau scheduler that tracks validation loss (lower is better)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',           # minimize validation loss
    factor=hyperVar['lr_factor'],           # on plateau: lr <- lr * 0.2
    patience=hyperVar['lr_patience'],           # wait 2 validation epochs with no improvement
    threshold=1e-4,       # minimal change to qualify as improvement
    cooldown=0,           # no cooldown
    min_lr=hyperVar['min_lr'],          # do not go below this lr
)

def maybe_disable_wd(opt, threshold):
    for pg in opt.param_groups:
        if pg.get('decay', False) and pg['lr'] <= threshold and pg['weight_decay'] > 0.0:
            pg['weight_decay'] = 0.0
            pg['wd_disabled'] = True
            print("Disabled weight decay for a param group (lr below threshold)")


# Setup Model Checkpoint handler to save the best model
import os
pydir = os.path.dirname(os.path.abspath(__file__)) # This results "fresh-start/"
checkpoint_dir = f"{pydir}/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_handler = ModelCheckpoint(
    checkpoint_dir,
    filename_prefix=f"{hyperVar['suffix']}",
    require_empty=False,
    score_function=score,  # Using the same score_function as early stopping
    score_name=hyperVar['score_metric'],
    n_saved=1,  # Only save the best model
    global_step_transform=lambda engine, _: engine.state.epoch
)
evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, {'model': model})

# Dictionary to store the best model parameters and corresponding score
best_model_info = {
    'params': None,
    'score': float('-inf'),  # Initialize with worst possible score (we want to maximize -loss)
    'epoch': 0,
    'history': {
        'eps': [],
        'losses': [],
        'val_losses': [],
        'lr': []
    }
}


# ---- Backbone freeze just for the first epoch (no optimizer rebuild) ----
@trainer.on(Events.EPOCH_STARTED)
def freeze_backbone_for_starting_epochs(engine):
    # Record epoch start time
    engine.state.epoch_start_time = time.time()
    
    if engine.state.epoch <= hyperVar['freeze_backbone_epochs']:
        for pg in optimizer.param_groups:
            if pg.get('backbone', False):
                # Save originals once
                pg.setdefault('orig_lr', pg['lr'])
                pg.setdefault('orig_wd', pg['weight_decay'])
                # Disable updates (and WD) for backbone during epoch 1
                pg['lr'] = 0.0
                pg['weight_decay'] = 0.0
        print("Backbone frozen: lr=0, weight_decay=0; BN running stats frozen via eval()")


@trainer.on(Events.EPOCH_COMPLETED)
def run_validation_loss_track(engine):
    # Calculate epoch time
    epoch_time = time.time() - engine.state.epoch_start_time

    if engine.state.epoch <= hyperVar['freeze_backbone_epochs']:
        for pg in optimizer.param_groups:
            if pg.get('backbone', False):
                # Restore original lr and weight decay
                pg['lr'] = pg.get('orig_lr', hyperVar['b_lr'])
                pg['weight_decay'] = pg.get('orig_wd', hyperVar['weight_decay'])
        print("Backbone unfrozen: restored lr and weight_decay; BN back to train() behavior")


    # Track losses for plotting
    eps.append(trainer.state.epoch)
    losses.append(trainer.state.output)

    # Run validation
    evaluator.run(validate_loader)
    val_loss = evaluator.state.metrics['loss']
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)  # Step the scheduler based on validation loss
    maybe_disable_wd(optimizer, hyperVar['wd_off_below_lr'])

    lr.append([pg['lr'] for pg in optimizer.param_groups])
    
    # Check if this is the best model so far
    current_score = score(evaluator)  # Negative because we want to maximize score (minimize loss)
    if current_score > best_model_info['score']:
        best_model_info['score'] = current_score
        best_model_info['params'] = {k: v.clone().detach() for k, v in model.state_dict().items()}
        best_model_info['epoch'] = trainer.state.epoch
        best_model_info['history'] = {
            'epochs': eps.copy(),
            'train_losses': losses.copy(),
            'val_losses': val_losses.copy(),
            'learning_rates': lr.copy()
        }
        print(f"New best model found at epoch {trainer.state.epoch}!")
    print(f"Epoch {trainer.state.epoch} ({epoch_time:.2f}s) - Training Loss: {trainer.state.output:.4f}, "
            f"Validation Loss: {val_loss:.4f}, Best Epoch: {best_model_info['epoch']}")
    print("Current LRs:", [f"{a:.2e}" for a in lr[-1]])

if hyperVar['cluster_flag']:
    ProgressBar().attach(trainer, output_transform=lambda x: {'loss': x})

# %% CHECKPOINTS #

print("CHOOSE CHECKPOINT...")
# Ask user if they want to resume training from a checkpoint
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
if os.path.exists(checkpoint_dir):
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') or f.endswith('.pth')]
    
    if files:
        print("\nAvailable checkpoint files:")
        for i, file in enumerate(files):
            print(f"{i+1}. {file}")
        
        while True:
            try:

                try:
                    choice = int(sys.argv[1])
                except:
                    choice = int(input("\nSelect checkpoint number (or 0 to cancel): "))

                if choice == 0:
                    print("Canceled. Starting training from scratch...")
                    break
                elif 1 <= choice <= len(files):
                    checkpoint_path = os.path.join(checkpoint_dir, files[choice-1])
                    print(f"Loading checkpoint: {checkpoint_path}")
                    # Load the model state
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint)
                    break
                else:
                    print(f"Invalid choice. Please select a number between 0 and {len(files)}.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        print("No checkpoint files found in the directory.")
else:
    print(f"Checkpoint directory {checkpoint_dir} does not exist.")
    
print("NUM WORKERS:", hyperVar['num_workers'], "BATCH SIZE:", hyperVar['batch_size'])


# %% ********TRAINING*********
print("--TRAINING---")
# trainer.run(dataloader, max_epochs=hyperVar['n_epochs'])

# Restore the best model
if best_model_info['params'] is not None:
    print(f"Restoring best model from epoch {best_model_info['epoch']} with score: {best_model_info['score']:.4f}")
    model.load_state_dict(best_model_info['params'])
else:
    print("Warning: No best model was found during training!")

 # %%   ------Evaluation of the model------
set = testSet
perc_error_per_parameter(
    # Need to add correct transform function
    output_transform=lambda x: (x[0], x[1]), device=device
).attach(evaluator, 'perc_pp')

prediction_diff_histograms(
    output_transform=lambda x: (x[0], x[1]), device=device
).attach(evaluator, 'param_hists')
print("Python <<<<< legit everything else (except C)\n")


if hyperVar['cluster_flag']:
    ProgressBar().attach(evaluator, output_transform=lambda x: {'loss': x})

state = evaluator.run(test_loader)
test_loss = state.metrics['loss']
print(f"Test Loss: {test_loss:.4f}")


# %% [] [] [] Plotting [] [] []
# Plot training and validation loss over epochs

def plot_loss_curves(eps, losses, val_losses, name):
    plt.figure(figsize=(10, 6))
    plt.plot(eps, losses, label='Training Loss')
    plt.plot(eps, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)

    os.makedirs('pdfs', exist_ok=True)
    plt.savefig(f'pdfs/{name}.pdf', bbox_inches='tight', dpi=300)

plot_loss_curves(eps, losses, val_losses, name=f'loss_curves_{hyperVar["suffix"]}')


def plot_predictions_vs_actual(model, dataloader, device, name, n_samples=5):
    """
    Plot histograms with predictions vs actual parameters.
    
    Args:
        model: The trained neural network model
        dataloader: DataLoader containing test data
        device: Device to run inference on
        n_samples: Number of samples to plot (default: 5)
    """
    model.eval()
    
    # Get n_samples from the dataloader
    samples_collected = 0
    all_inputs = []
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            batch_size = X_batch.size(0)
            remaining = n_samples - samples_collected
            
            if remaining <= 0:
                break
                
            # Take only what we need from this batch
            take = min(batch_size, remaining)
            X = X_batch[:take].to(device)
            Y = Y_batch[:take]
            
            # Get predictions
            Y_pred = model(X).cpu()
            
            all_inputs.append(X_batch[:take].cpu())
            all_targets.append(Y)
            all_predictions.append(Y_pred)
            
            samples_collected += take
    
    # Concatenate all collected samples
    inputs = torch.cat(all_inputs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    predictions = torch.cat(all_predictions, dim=0)
    
    # Create subplots
    fig, axes = plt.subplots(n_samples, 3, figsize=(18, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Get the first channel (assuming multi-channel input)
        img = inputs[i, 0].numpy()  # Take first channel
        target = targets[i].numpy()
        pred = predictions[i].numpy()
        
        # Calculate differences
        diff = np.abs(pred - target)
        
        # Plot histogram
        ax_hist = axes[i, 0]
        im = ax_hist.imshow(img, origin='lower', aspect='auto')
        ax_hist.set_title(f'Sample {i+1}: Histogram')
        ax_hist.set_xlabel('X bins')
        ax_hist.set_ylabel('Y bins')
        plt.colorbar(im, ax=ax_hist)
        
        # Plot parameter comparison
        ax_params = axes[i, 1]
        param_indices = np.arange(len(target))
        width = 0.35
        
        ax_params.bar(param_indices - width/2, target, width, label='Actual', alpha=0.7)
        ax_params.bar(param_indices + width/2, pred, width, label='Predicted', alpha=0.7)
        ax_params.set_xlabel('Parameter Index')
        ax_params.set_ylabel('Parameter Value (scaled)')
        ax_params.set_title(f'Sample {i+1}: Parameters Comparison')
        ax_params.legend()
        ax_params.grid(True, alpha=0.3)
        
        # Plot differences
        ax_diff = axes[i, 2]
        ax_diff.bar(param_indices, diff, color='red', alpha=0.7)
        ax_diff.set_xlabel('Parameter Index')
        ax_diff.set_ylabel('Absolute Difference')
        ax_diff.set_title(f'Sample {i+1}: |Predicted - Actual|')
        ax_diff.grid(True, alpha=0.3)
        
        # Add MSE text
        mse = np.mean(diff**2)
        ax_diff.text(0.98, 0.98, f'MSE: {mse:.6f}', 
                    transform=ax_diff.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()

    os.makedirs('pdfs', exist_ok=True)
    plt.savefig(f'pdfs/{name}.pdf', bbox_inches='tight', dpi=300)
    return fig


# Plot predictions vs actual for test set
print("\nGenerating predictions vs actual plots...")
fig_predictions = plot_predictions_vs_actual(model, test_loader, device,
                                             name=f'predictions_vs_actual_{hyperVar["suffix"]}', n_samples=5)

def Plot_perc_error(percs, name):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(percs)), percs.cpu().numpy())
    plt.xlabel('Parameter Index')
    plt.ylabel('Average Percentage Error (%)')
    plt.title('Percentage Error per Parameter on Test Set')
    plt.grid(True)

    os.makedirs('pdfs', exist_ok=True)
    plt.savefig(f'pdfs/{name}.pdf', bbox_inches='tight', dpi=300)

Plot_perc_error(state.metrics['perc_pp'], name=f'pepp_{hyperVar["suffix"]}')
print(f"\nTotal script time: {(time.time() - start_script_time)/60:.2f} minutes")

def Plot_Parameter_Histograms(diff_histograms, diff_bins, name):
    # Here we plot n_params 1D histogram subplots in a grid
    n_params = len(diff_histograms)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()
    for i in range(n_params):
        ax = axes[i]
        # hist shape: (num_bins,)
        ax.semilogy(diff_bins[i][:-1], diff_histograms[i], drawstyle='steps-post')
        ax.set_title(f'Parameter {i+1} Histogram of Differences')
        ax.set_xlabel('Bins')
        ax.set_ylabel('Counts')

    plt.tight_layout()
    os.makedirs('pdfs', exist_ok=True)
    plt.savefig(f'pdfs/{name}.pdf', bbox_inches='tight', dpi=300)
diff_histograms = state.metrics['param_hists'][0]
diff_bin_edges = state.metrics['param_hists'][1]
Plot_Parameter_Histograms(diff_histograms, diff_bin_edges, name=f'param_histograms_{hyperVar["suffix"]}')